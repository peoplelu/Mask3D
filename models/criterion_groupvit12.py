# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
# Modified for Mask3D
"""
MaskFormer criterion.
"""

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from models.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list

def get_segmented_scores(scores, fg_thresh=0.75, bg_thresh=0.25):
    '''
    :param scores: (N), float, 0~1
    :return: segmented_scores: (N), float 0~1, >fg_thresh: 1, <bg_thresh: 0, mid: linear
    '''
    fg_mask = scores > fg_thresh
    bg_mask = scores < bg_thresh
    interval_mask = (fg_mask == 0) & (bg_mask == 0)

    segmented_scores = (fg_mask > 0).float()
    k = 1 / (fg_thresh - bg_thresh)
    b = bg_thresh / (bg_thresh - fg_thresh)
    segmented_scores[interval_mask] = scores[interval_mask] * k + b

    return segmented_scores

def get_iou(inputs, targets):
    inputs = inputs.sigmoid()
    # thresholding
    binarized_inputs = (inputs >= 0.5).float()
    targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes

def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule

def dice_loss_multi_classes(input,
                            target,
                            epsilon= 1e-5,
                            weight=None):
    r"""
    modify compute_per_channel_dice from https://github.com/wolny/pytorch-3dunet/blob/6e5a24b6438f8c631289c10638a17dea14d42051/unet3d/losses.py
    """
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    # convert the feature channel(category channel) as first
    axis_order = (1, 0) + tuple(range(2, input.dim()))
    input = input.permute(axis_order)
    target = target.permute(axis_order)

    target = target.float()
    # Compute per channel Dice Coefficient
    per_channel_dice = (2 * torch.sum(input * target, dim=1) + epsilon) / \
                       (torch.sum(input * input, dim=1) + torch.sum(target * target, dim=1) + 1e-4 + epsilon)

    loss = 1. - per_channel_dice

    return loss

def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio,
                 class_weights):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes - 1
        self.class_weights = class_weights
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.aux_losses = self.losses[1:]
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.focal_alpha = 0.25
        if self.class_weights != -1:
            assert len(self.class_weights) == self.num_classes, "CLASS WEIGHTS DO NOT MATCH"
            empty_weight[:-1] = torch.tensor(self.class_weights)

        self.register_buffer("empty_weight", empty_weight)

        self.empty_weight1 = empty_weight.clone()
        self.empty_weight1[-1] = 0.1
        self.empty_weight2 = torch.ones_like(self.empty_weight1)
        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def loss_point_semantic(self,epoch,outputs, targets, indices, num_masks, mask_type):
        assert "semantic_scores" in outputs
        pixel_sem_criterion = nn.CrossEntropyLoss(ignore_index=253).cuda()
        semantic_scores = outputs["semantic_scores"].float()
        # semantic_scores: (N, nClass), float32, cuda
        # semantic_labels: (N), long, cuda
        # CELoss
        
        semantic_labels = torch.cat([targets[i]['pointsemlabel'] for i in range(len(targets))],dim=0)
        semantic_labels = torch.where(semantic_labels<0,torch.tensor(18).cuda(),semantic_labels)
        semantic_loss = pixel_sem_criterion(semantic_scores, semantic_labels)
        # multi-classes dice loss
        semantic_labels_ = semantic_labels[semantic_labels != 253]
        semantic_scores_ = semantic_scores[semantic_labels != 253]
        one_hot_labels = F.one_hot(semantic_labels_, num_classes=self.num_classes+1)
        semantic_scores_softmax = F.softmax(semantic_scores_, dim=-1)
        semantic_loss += dice_loss_multi_classes(semantic_scores_softmax, one_hot_labels).mean()
        return  {"semantic_loss":semantic_loss}

    def loss_labels(self,level, outputs, targets, indices, num_masks, mask_type):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()
        #self.empty_weight[-1] = 10
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        #src_logits[idx].softmax(-1)[target_classes_o!=253][torch.arange((target_classes_o!=253).sum()).cuda(),target_classes_o[target_classes_o!=253]]
        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_masks, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]

        losses = {"loss_ce": loss_ce}
        return losses

    def loss_boxes(self,epoch, outputs, targets, indices, num_boxes,mask_type):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        #tgt_mask = [t[mask_type][i] for t, (_, i) in zip(targets, indices)]
        loss_bbox = F.l1_loss(src_boxes[:,:3], target_boxes[:,:3], reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / target_boxes.shape[0]
        
        # nactual_gt = torch.cat([t.sum(1).long() for t in tgt_mask], dim=0)
        #         #angle = torch.zeros_like(out_boxes[:,0])
        # pred_box = get_3d_box_batch_tensor(src_boxes[...,3:],torch.zeros_like(src_boxes[:,0]),src_boxes[...,:3])[None,...]
        # gt_box = get_3d_box_batch_tensor(target_boxes[...,3:],torch.zeros_like(target_boxes[:,0]),target_boxes[...,:3])[None,...]

        #loss_giou = 1 - torch.diag(generalized_box3d_iou_tensor(pred_box,gt_box,nactual_gt,rotated_boxes=False).squeeze(0))
        #losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses
    def get_region_gtmask(self,map,target_mask,target_id):
        unique_target =  target_id.unique()
        unique_mask = target_mask[unique_target]
        region_map = torch.zeros_like(unique_mask).float()
        for i in range(unique_target.shape[0]):
          a = target_id==unique_target[i]
          map1 = map[a]
          region_map[i] = map1.max(0)[0]
        return region_map,unique_mask.float()
    
    def loss_masks(self,level, outputs, targets, indices, num_masks, mask_type):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        loss_masks = []
        loss_dices = []
        loss_score = []
        for batch_id, (map_id, target_id) in enumerate(indices):
            map = outputs["pred_masks"][batch_id][:,map_id].T
            target_mask = targets[batch_id][mask_type][target_id]
            
            if self.num_points != -1:
                point_idx = torch.randperm(target_mask.shape[1],
                                          device=target_mask.device)[:int(self.num_points*target_mask.shape[1])]
            else:
                # sample all points
                point_idx = torch.arange(target_mask.shape[1], device=target_mask.device)

            num_masks = target_mask.shape[0]
            map = map[:, point_idx]
            target_mask = target_mask[:, point_idx].float()
            #tgt_score = torch.zeros_like(pred_score)
            with torch.no_grad():
                tgt_score = get_iou(map, target_mask).unsqueeze(1)
            valid_idx = tgt_score>0.1
            if valid_idx.sum()!=0 and (level==-1 or level == 1 or level==3):
              pred_score = outputs["score"][batch_id][map_id]
              loss_score.append(F.binary_cross_entropy(pred_score[valid_idx], tgt_score[valid_idx]))
            else:
              loss_score.append(torch.tensor(0).cuda().float())
            
            region_map,region_gtmask = self.get_region_gtmask(map,targets[batch_id][mask_type],target_id)
            #if level>=0 and level <=5:
            if level==-20:
              num_masks = region_gtmask.shape[0]
              loss_masks.append(sigmoid_ce_loss_jit(region_map,
                                                    region_gtmask,
                                                    num_masks))
              loss_dices.append(dice_loss_jit(region_map,
                                              region_gtmask,
                                              num_masks))
            else:
              num_masks = target_mask.shape[0]
              loss_masks.append(sigmoid_ce_loss_jit(map,
                                                    target_mask,
                                                    num_masks))
              loss_dices.append(dice_loss_jit(map,
                                              target_mask,
                                              num_masks))
          # del target_mask
        return {
            "loss_mask": torch.sum(torch.stack(loss_masks)),
            "loss_dice": torch.sum(torch.stack(loss_dices)),
            "loss_score": torch.sum(torch.stack(loss_score)),
        }

        # src_idx = self._get_src_permutation_idx(indices)
        # tgt_idx = self._get_tgt_permutation_idx(indices)
        # src_masks = outputs["pred_masks"]
        # src_masks = src_masks[src_idx]
        # masks = [t[mask_type] for t in targets]
        # # TODO use valid to mask invalid areas due to padding in loss
        # target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        # target_masks = target_masks.to(src_masks)
        # target_masks = target_masks[tgt_idx]

        # # No need to upsample predictions as we are using normalized coordinates :)
        # # N x 1 x H x W
        # src_masks = src_masks[:, None]
        # target_masks = target_masks[:, None]

        # with torch.no_grad():
        #     # sample point_coords
        #     point_coords = get_uncertain_point_coords_with_randomness(
        #         src_masks,
        #         lambda logits: calculate_uncertainty(logits),
        #         self.num_points,
        #         self.oversample_ratio,
        #         self.importance_sample_ratio,
        #     )
        #     # get gt labels
        #     point_labels = point_sample(
        #         target_masks,
        #         point_coords,
        #         align_corners=False,
        #     ).squeeze(1)

        # point_logits = point_sample(
        #     src_masks,
        #     point_coords,
        #     align_corners=False,
        # ).squeeze(1)

        # losses = {
        #     "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks, mask_type),
        #     "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks, mask_type),
        # }

        # del src_masks
        # del target_masks
        # return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self,epoch, loss, outputs, targets, indices, num_masks, mask_type):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
            'boxes': self.loss_boxes,
            'point_sem': self.loss_point_semantic,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](epoch,outputs, targets, indices, num_masks, mask_type)

    def forward(self, outputs, targets,epoch, mask_type):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        #outputs_without_aux['downsample']=True
        outputs_without_aux['score'] = outputs_without_aux['score'][-1]

        
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, mask_type)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        level = -1
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(level,loss, outputs_without_aux, targets, indices, num_masks, mask_type))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        torch.cuda.empty_cache()
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                torch.cuda.empty_cache()
                aux_outputs['sampled_coords'] = outputs['sampled_coords']
                if i == 1:
                  aux_outputs['score'] = outputs['score'][0]
                elif i == 3:
                  aux_outputs['score'] = outputs['score'][1]
#                 if i<4:
#                   aux_outputs['downsample']=False
# #                  aux_outputs['num'] = outputs['num']
#                 else:
#                   aux_outputs['downsample']=True
#                  aux_outputs['num'] = outputs['num']
                indices = self.matcher(aux_outputs, targets, mask_type)
                for loss in self.losses:
                    l_dict = self.get_loss(i,loss, aux_outputs, targets, indices, num_masks, mask_type)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
