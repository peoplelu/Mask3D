# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast
import numpy as np
from detectron2.projects.point_rend.point_features import point_sample


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
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
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
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
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule

# from https://github.com/facebookresearch/detectron2/blob/cbbc1ce26473cb2a5cc8f58e8ada9ae14cb41052/detectron2/layers/wrappers.py#L100
def nonzero_tuple(x):
    """
    A 'as_tuple=True' version of torch.nonzero to support torchscript.
    because of https://github.com/pytorch/pytorch/issues/38718
    """
    if torch.jit.is_scripting():
        if x.dim() == 0:
            return x.unsqueeze(0).nonzero().unbind(1)
        return x.nonzero().unbind(1)
    else:
        return x.nonzero(as_tuple=True)

# from https://github.com/facebookresearch/detectron2/blob/cbbc1ce26473cb2a5cc8f58e8ada9ae14cb41052/detectron2/modeling/sampling.py#L9
def subsample_labels(
    labels: torch.Tensor, num_samples: int, positive_fraction: float, bg_label: int
):
    """
    Return `num_samples` (or fewer, if not enough found)
    random samples from `labels` which is a mixture of positives & negatives.
    It will try to return as many positives as possible without
    exceeding `positive_fraction * num_samples`, and then try to
    fill the remaining slots with negatives.
    Args:
        labels (Tensor): (N, ) label vector with values:
            * -1: ignore
            * bg_label: background ("negative") class
            * otherwise: one or more foreground ("positive") classes
        num_samples (int): The total number of labels with value >= 0 to return.
            Values that are not sampled will be filled with -1 (ignore).
        positive_fraction (float): The number of subsampled labels with values > 0
            is `min(num_positives, int(positive_fraction * num_samples))`. The number
            of negatives sampled is `min(num_negatives, num_samples - num_positives_sampled)`.
            In order words, if there are not enough positives, the sample is filled with
            negatives. If there are also not enough negatives, then as many elements are
            sampled as is possible.
        bg_label (int): label index of background ("negative") class.
    Returns:
        pos_idx, neg_idx (Tensor):
            1D vector of indices. The total length of both is `num_samples` or fewer.
    """
    positive = nonzero_tuple((labels != -1) & (labels != bg_label))[0]
    negative = nonzero_tuple(labels == bg_label)[0]

    num_pos = int(num_samples * positive_fraction)
    # protect against not enough positive examples
    num_pos = min(positive.numel(), num_pos)
    num_neg = num_samples - num_pos
    # protect against not enough negative examples
    num_neg = min(negative.numel(), num_neg)

    # randomly select positive and negative examples
    perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
    perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

    pos_idx = positive[perm1]
    neg_idx = negative[perm2]
    return pos_idx, neg_idx

def sample_topk_per_gt(pr_inds, gt_inds, cost, k):
    if len(gt_inds) == 0:
        return pr_inds, gt_inds
    # find topk matches for each gt
    gt_inds2, counts = gt_inds.unique(return_counts=True)
    scores, pr_inds2 = cost[:,gt_inds2].topk(k, largest=False, dim=0)
    map1 = torch.zeros(gt_inds2.max()+1).cuda().float()
    map1 = map1[None,:].scatter(-1,gt_inds2[None,:],torch.arange(gt_inds2.shape[0]).cuda().float()[None,:]).squeeze(0)

    scores_1 = cost[pr_inds,gt_inds]
    scores_2 = scores[-1][map1[gt_inds].long()]
    valid = scores_1<scores_2
    pr_inds3 = pr_inds[valid]
    gt_inds3 = gt_inds[valid]
    # gt_inds2 = gt_inds2[:,None].repeat(1, k)
    
    # # filter to as many matches that gt has
    # pr_inds3 = torch.cat([pr[:c] for c, pr in zip(counts, pr_inds2.transpose(0,1))])
    # gt_inds3 = torch.cat([gt[:c] for c, gt in zip(counts, gt_inds2)])

    return pr_inds3, gt_inds3

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, num_points: int = 0,cost_box=0,cost_point_sem_class=0, panoptic_on=False):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.cost_box = cost_box
        self.cost_point_sem_class = cost_point_sem_class

        self.panoptic_on = panoptic_on

        self.map = {512:0,256:1,128:1,100:2,200:1,196:1}
        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"
        self.prob512 = 0 #fps 0.9782737145498239 ffps 0.968731679781331  0.9864634584757721
        self.prob256 = 0 #fps 0.9452870135624986 ffps 0.9269959061068217  0.9123324419324104 0.9597070511320976 0.9619457869931849
        self.prob5121 = 0 # 0.9888642537511388
        self.prob2561 = 0 # 0.9231702553497081 0.9665586722652817
        self.prob100 = 0 
        self.prob1001 = 0 
        self.prob1002 = 0
        self.prob1003 = 0 
        self.prob1004 = 0
        self.num_points = num_points
        self.thresholds = [0.3, 0.6]
        self.thresholds.insert(0, -float("inf"))
        self.thresholds.append(float("inf"))
        self.labels = [1, -1, 0]
        self.k = 4
    
    def set_low_quality_matches_(self, match_labels, match_quality_matrix):
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth G find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth G.
        This function implements the RPN assignment case (i) in Sec. 3.1.2 of
        :paper:`Faster R-CNN`.
        """
        # For each gt, find the prediction with which it has highest quality
        highest_quality_foreach_gt, _ = match_quality_matrix.min(dim=0)
        # Find the highest quality match available, even if it is low, including ties. 512 * N   N
        # Note that the matches qualities must be positive due to the use of
        # `torch.nonzero`.
        pred_inds_with_highest_quality,_  = nonzero_tuple(
            match_quality_matrix == highest_quality_foreach_gt[None,:]
        )
        # If an anchor was labeled positive only due to a low-quality match
        # with gt_A, but it has larger overlap with gt_B, it's matched index will still be gt_B.
        # This follows the implementation in Detectron, and is found to have no significant impact.
        match_labels[pred_inds_with_highest_quality] = 1

    def _subsample_labels(self, label):
        """
        Randomly sample a subset of positive and negative examples, and overwrite
        the label vector to the ignore value (-1) for all elements that are not
        included in the sample.
        Args:
            labels (Tensor): a vector of -1, 0, 1. Will be modified in-place and returned.
        """
        pos_idx, neg_idx = subsample_labels(
            label, self.batch_size_per_image, self.positive_fraction, 0
        )
        # Fill with the ignore label (-1), then set positive and negative labels
        label.fill_(-1)
        label.scatter_(0, pos_idx, 1)
        label.scatter_(0, neg_idx, 0)
        return label

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets, mask_type):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]
        # if num_queries==100:
        #   num = outputs['num']
        indices = []
        # Iterate through batch size
        for b in range(bs):
            out_prob = outputs["pred_logits"][b].sigmoid()  # [num_queries, num_classes]
            tgt_ids = targets[b]["labels"].clone()
            
            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            valid = (tgt_ids != 253)
            cost_class = -out_prob[:, tgt_ids[valid]]  # for ignore classes pretend perfect match ;) TODO better worst class match?

            
            sample_coord = outputs['sampled_coords'][self.map[out_prob.shape[0]]][b]
            #out_bbox = outputs['pred_boxes'][b][:,:3]
            tgt_coord = targets[b]['coords']
            coord_cost = []
            for i in range(len(tgt_ids)):
              valid_coords = tgt_coord[targets[b]['masks'][i]]
              dis_min = torch.norm((sample_coord[:,None,:] - valid_coords[None,...]), dim=-1).min(-1)[0]
              coord_cost.append(dis_min)
            coord_cost = torch.stack(coord_cost).transpose(0,1)
            coord_cost = torch.where(torch.isnan(coord_cost),torch.tensor(0).cuda().float(),coord_cost)
            coord_cost = coord_cost[:,valid]
            #valid = (coord_cost.min(1)[0]<0.1).cpu().numpy() 
            a = torch.norm((targets[0]['boxes'][:,:3][None,...]-sample_coord[:,None,:]),dim=-1)
            
              # if valid.sum()==0:
              #   print('1')
            # prob = (~torch.isin(coord_cost.min(-1)[1][coord_cost.min(1)[0]<0.1].unique(), torch.where(filter_ignore)[0])).sum()/sum(~filter_ignore)
            # prob1 = (~torch.isin(coord_cost.min(-1)[1].unique(), torch.where(filter_ignore)[0])).sum()/sum(~filter_ignore)
            # prob2 = coord_cost.min(-1)[1][coord_cost.min(1)[0]<0.1].unique().shape[0]/len(tgt_ids)
            # prob3 = coord_cost.min(-1)[1].unique().shape[0]/len(tgt_ids)
            # if self.map[out_prob.shape[0]] == 0:
            #   self.prob512 += prob
            #   self.prob5121 += prob1
            # elif self.map[out_prob.shape[0]] == 1:
            #   self.prob256 += prob
            #   self.prob2561 += prob1
            # elif self.map[out_prob.shape[0]] == 2:
            #   self.prob100 += prob
            #   self.prob1001 += prob1
            #   self.prob1002 += prob2
            #   self.prob1003 += prob3

              # self.prob100 += coord_cost[:num[b]].min(-1)[1][coord_cost[:num[b]].min(1)[0]<0.1].unique().shape[0]/len(tgt_ids)  
              # self.prob1001 += coord_cost[:num[b]].min(-1)[1].unique().shape[0]/len(tgt_ids)  
              # self.prob1002 += coord_cost[:num[b]].argmin(1)[coord_cost[:num[b]].min(1)[0]<0.1].unique().shape[0]/coord_cost[:num[b]].argmin(1)[coord_cost[:num[b]].min(1)[0]<0.1].shape[0]
            out_mask = outputs['pred_masks'][b].T  # [num_queries, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_mask = targets[b][mask_type].to(out_mask)[valid]


            if self.num_points != -1:
                point_idx = torch.randperm(tgt_mask.shape[1],
                                           device=tgt_mask.device)[:int(self.num_points*tgt_mask.shape[1])]
                #point_idx = torch.randint(0, tgt_mask.shape[1], size=(self.num_points,), device=tgt_mask.device)
            else:
                # sample all points
                point_idx = torch.arange(tgt_mask.shape[1], device=tgt_mask.device)


            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                # Compute the focal loss between masks
                cost_mask = batch_sigmoid_ce_loss_jit(out_mask[:, point_idx], tgt_mask[:, point_idx])

                # Compute the dice loss betwen masks
                cost_dice = batch_dice_loss_jit(out_mask[:, point_idx], tgt_mask[:, point_idx])
            #get_iou(map, target_mask).unsqueeze(1)
            # binarized_inputs = (out_mask >= 0.5).float()
            # targets1 = (tgt_mask > 0.5).float()
            # inter = binarized_inputs@targets1.transpose(0,1)
            # point_num = binarized_inputs.sum(1)[:,None] + targets1.sum(1)[None,:]
            # ious = inter / (point_num - inter + 1e-6)
            # Final cost matrix
            C = (
                self.cost_mask * cost_mask
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
                + self.cost_mask * coord_cost
            )           
            
            #matched_labels = torch.zeros(C.shape[0]).cuda()
            matched_vals, matches = C.min(dim=1)
            coords_min = coord_cost[torch.arange(C.shape[0]).cuda(),matches]
            match_labels = matches.new_full(matches.size(), 1, dtype=torch.int8)
            # self.thresholds = [0.3, 0.6]
            for (l, low, high) in zip(self.labels, self.thresholds[:-1], self.thresholds[1:]):
              low_high = (coords_min >= low) & (coords_min < high)
              match_labels[low_high] = l
            self.set_low_quality_matches_(match_labels, C)
            #matched_labels = self._subsample_labels(matched_labels)
            all_pr_inds = torch.arange(C.shape[0])
            pos_pr_inds = all_pr_inds[match_labels == 1]
            pos_gt_inds = matches[pos_pr_inds]
            #pos_cost = C[pos_pr_inds , pos_gt_inds]
            pos_pr_inds, pos_gt_inds = sample_topk_per_gt(pos_pr_inds, pos_gt_inds, C, self.k)
            # 剩余queries与剩余gt做匈牙利
            #pos_pr_inds, pos_gt_inds = pos_pr_inds.to(anchors.device), pos_gt_inds.to(anchors.device)
            indices.append((pos_pr_inds, torch.where(valid)[0][pos_gt_inds]))
        
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]


    @torch.no_grad()
    def forward(self, outputs, targets, mask_type):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets, mask_type)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
