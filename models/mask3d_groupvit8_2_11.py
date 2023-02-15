import torch
import hydra
import torch.nn as nn
import functools
import MinkowskiEngine.MinkowskiOps as me
from MinkowskiEngine.MinkowskiPooling import MinkowskiAvgPooling
import numpy as np
from torch.nn import functional as F
from models.modules.common import conv
from models.position_embedding import PositionEmbeddingCoordsSine
from third_party.pointnet2.pointnet2_utils import furthest_point_sample
from models.modules.helpers_3detr import GenericMLP
from torch_scatter import scatter_mean, scatter_max
from torch.cuda.amp import autocast
import os
from models.tiny_unet import TinyUnet
from third_party.softgroup.ops import (ballquery_batch_p, bfs_cluster, get_mask_iou_on_cluster, get_mask_iou_on_pred,
                   get_mask_label, global_avg_pool, sec_max, sec_min, voxelization,
                   voxelization_idx)
from third_party.pointops2.functions import pointops
def non_max_suppression(ious, scores, threshold):
    ixs = scores.argsort()[::-1]
    pick = []
    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)
        iou = ious[i, ixs[1:]]
        remove_ixs = np.where(iou > threshold)[0] + 1
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
class Mask3D(nn.Module):
    def __init__(self, config, hidden_dim, num_queries, num_heads, dim_feedforward,
                 sample_sizes, shared_decoder, num_classes,
                 num_decoders, dropout, pre_norm,
                 positional_encoding_type, non_parametric_queries, train_on_segments, normalize_pos_enc,
                 use_level_embed, scatter_type, hlevels,
                 use_np_features,
                 voxel_size,
                 max_sample_size,
                 random_queries,
                 gauss_scale,
                 random_query_both,
                 random_normal,
                 ):
        super().__init__()
        self.idx = 0
        self.random_normal = random_normal
        self.random_query_both = random_query_both
        self.random_queries = random_queries
        self.max_sample_size = max_sample_size
        self.gauss_scale = gauss_scale
        self.voxel_size = voxel_size
        self.scatter_type = scatter_type
        self.hlevels = hlevels
        self.use_level_embed = use_level_embed
        self.train_on_segments = train_on_segments
        self.normalize_pos_enc = normalize_pos_enc
        self.num_decoders = num_decoders
        self.num_classes = num_classes
        self.dropout = dropout
        self.pre_norm = pre_norm
        self.shared_decoder = shared_decoder
        self.sample_sizes = sample_sizes
        self.non_parametric_queries = non_parametric_queries
        self.use_np_features = use_np_features
        self.mask_dim = hidden_dim
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.pos_enc_type = positional_encoding_type
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        self.backbone = hydra.utils.instantiate(config.backbone)
        self.num_levels = len(self.hlevels)
        sizes = self.backbone.PLANES[-5:]
        self.iou_dim = 32
        self.mask_features_head = conv(
            self.backbone.PLANES[7], self.mask_dim, kernel_size=1, stride=1, bias=True, D=3
        )
        # self.iou_features_head = conv(
        #     self.backbone.PLANES[7], self.iou_dim, kernel_size=1, stride=1, bias=True, D=3
        # )

        if self.scatter_type == "mean":
            self.scatter_fn = scatter_mean
        elif self.scatter_type == "max":
            self.scatter_fn = lambda mask, p2s, dim: scatter_max(mask, p2s, dim=dim)[0]
        else:
            assert False, "Scatter function not known"

        assert (not use_np_features) or non_parametric_queries, "np features only with np queries"
        self.query_feat = nn.Embedding(100, hidden_dim)
            # learnable query p.e.
        #self.query_pos = nn.Embedding(100, hidden_dim)
        if self.non_parametric_queries:
            self.query_projection = GenericMLP(
                input_dim=self.mask_dim,
                hidden_dims=[self.mask_dim],
                output_dim=self.mask_dim,
                use_conv=True,
                output_use_activation=True,
                hidden_use_bias=True,
            )
            # self.query_projection1 = GenericMLP(
            #     input_dim=self.mask_dim,
            #     hidden_dims=[self.mask_dim],
            #     output_dim=self.mask_dim,
            #     use_conv=True,
            #     output_use_activation=True,
            #     hidden_use_bias=True,
            # )
            #if self.use_np_features:
            self.np_feature_projection = nn.Sequential(
                nn.Linear(sizes[-1], hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        elif self.random_query_both:
            self.query_projection = GenericMLP(
                input_dim=2*self.mask_dim,
                hidden_dims=[2*self.mask_dim],
                output_dim=2*self.mask_dim,
                use_conv=True,
                output_use_activation=True,
                hidden_use_bias=True
            )
        else:
            # PARAMETRIC QUERIES
            # learnable query features
            self.query_feat = nn.Embedding(num_queries, hidden_dim)
            # learnable query p.e.
            self.query_pos = nn.Embedding(num_queries, hidden_dim)

        if self.use_level_embed:
            # learnable scale-level embedding
            self.level_embed = nn.Embedding(self.num_levels, hidden_dim)

        self.mask_embed_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # self.mask_embed_head1 = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim)
        # )
        self.idx = 0
        # self.linear = nn.Sequential(
        #     nn.Linear(self.backbone.PLANES[7], self.backbone.PLANES[7], bias=True),
        #     norm_fn(self.backbone.PLANES[7]),
        #     nn.ReLU(),
        #     nn.Linear(self.backbone.PLANES[7], self.backbone.PLANES[7], bias=True),
        #     norm_fn(self.backbone.PLANES[7]),
        #     nn.ReLU(),
        #     nn.Linear(self.backbone.PLANES[7], self.num_classes, bias=True),
        # )
        # #### offset branch
        # self.offset = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim, bias=True),
        #     norm_fn(hidden_dim),
        #     nn.ReLU()
        # )
        self.sem_coding = nn.Linear(self.num_classes, hidden_dim, bias=True)
        self._bbox_embed = _bbox_embed = MLP(hidden_dim, hidden_dim, 3, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)
        self.class_embed_head = nn.Linear(hidden_dim, self.num_classes)
        #self.class_embed_head1 = nn.Linear(hidden_dim, self.num_classes)
        # self.tiny_unet = TinyUnet(self.iou_dim)
        # self.iou_score_linear = nn.Linear(self.iou_dim,1)
        self.iou_score = GenericMLP(
                input_dim=hidden_dim,
                hidden_dims=[hidden_dim],
                output_dim=1,
                use_conv=False,
                output_use_activation=False,
                hidden_use_bias=True
            )

        if self.pos_enc_type == "legacy":
            self.pos_enc = PositionalEncoding3D(channels=self.mask_dim)
        elif self.pos_enc_type == "fourier":
            self.pos_enc = PositionEmbeddingCoordsSine(pos_type="fourier",
                                                       d_pos=self.mask_dim,
                                                       gauss_scale=self.gauss_scale,
                                                       normalize=self.normalize_pos_enc)
        elif self.pos_enc_type == "sine":
            self.pos_enc = PositionEmbeddingCoordsSine(pos_type="sine",
                                                       d_pos=self.mask_dim,
                                                       normalize=self.normalize_pos_enc)
        else:
            assert False, 'pos enc type not known'

        self.pooling = MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3)

        self.masked_transformer_decoder = nn.ModuleList()
        self.cross_attention = nn.ModuleList()
        self.self_attention = nn.ModuleList()
        self.ffn_attention = nn.ModuleList()
        self.lin_squeeze = nn.ModuleList()

        num_shared = self.num_decoders if not self.shared_decoder else 1
        sizes1 = [self.mask_dim,self.mask_dim]
        for _ in range(num_shared):
            tmp_cross_attention = nn.ModuleList()
            tmp_self_attention = nn.ModuleList()
            tmp_ffn_attention = nn.ModuleList()
            tmp_squeeze_attention = nn.ModuleList()
            for i, hlevel in enumerate(self.hlevels[:2]):
                tmp_cross_attention.append(
                    CrossAttentionLayer(
                        d_model=self.mask_dim,
                        nhead=self.num_heads,
                        dropout=self.dropout,
                        normalize_before=self.pre_norm
                    )
                )

                tmp_squeeze_attention.append(nn.Linear(sizes1[i], self.mask_dim))

                tmp_self_attention.append(
                    SelfAttentionLayer(
                        d_model=self.mask_dim,
                        nhead=self.num_heads,
                        dropout=self.dropout,
                        normalize_before=self.pre_norm
                    )
                )

                tmp_ffn_attention.append(
                    FFNLayer(
                        d_model=self.mask_dim,
                        dim_feedforward=dim_feedforward,
                        dropout=self.dropout,
                        normalize_before=self.pre_norm
                    )
                )

            self.cross_attention.append(tmp_cross_attention)
            self.self_attention.append(tmp_self_attention)
            self.ffn_attention.append(tmp_ffn_attention)
            self.lin_squeeze.append(tmp_squeeze_attention)


        self.assign_cross_attention = nn.ModuleList()
        self.assign_self_attention = nn.ModuleList()
        self.assign_ffn_attention = nn.ModuleList()
        self.assign_squeeze_attention = nn.ModuleList()
        for i, hlevel in enumerate(self.hlevels[2:]):
            self.assign_cross_attention.append(
                CrossAttentionLayer(
                    d_model=self.mask_dim,
                    nhead=self.num_heads,
                    dropout=self.dropout,
                    normalize_before=self.pre_norm
                )
            )

            self.assign_squeeze_attention.append(nn.Linear(sizes[i], self.mask_dim))

            self.assign_self_attention.append(
                SelfAttentionLayer(
                    d_model=self.mask_dim,
                    nhead=self.num_heads,
                    dropout=self.dropout,
                    normalize_before=self.pre_norm
                )
            )

            self.assign_ffn_attention.append(
                FFNLayer(
                    d_model=self.mask_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=self.dropout,
                    normalize_before=self.pre_norm
                )
            )

        self.decoder_norm =nn.LayerNorm(hidden_dim)
        #self.decoder_norm1 =nn.LayerNorm(hidden_dim)

    def get_pos_encs(self, coords):
        pos_encodings_pcd = []

        for i in range(len(coords)):
            pos_encodings_pcd.append([[]])
            for coords_batch in coords[i].decomposed_features:
                scene_min = coords_batch.min(dim=0)[0][None, ...]
                scene_max = coords_batch.max(dim=0)[0][None, ...]

                with autocast(enabled=False):
                    tmp = self.pos_enc(coords_batch[None, ...].float(),
                                       input_range=[scene_min, scene_max])

                pos_encodings_pcd[-1][0].append(tmp.squeeze(0).permute((1, 0)))

        return pos_encodings_pcd
    def get_fps_query(self,x,coordinates,pcd_features,scale,level,queries,query_pos,query_pos_xyz,output_class,iou_score,outputs_mask,point2segment):
        if level!=0:
          batch_size = len(x.decomposed_coordinates)
          sem_score = output_class.sigmoid()
          #valid = sem_score[...,-1]<0.5
          score = sem_score[...,:-1].max(-1)[0]
          #outputs_mask = [outputs_mask[i][:,valid[i]] for i in range(batch_size)]
          coord_mask = [((outputs_mask[i]>0).sum(-1)==0)[point2segment[i]] for i in range(batch_size)]
          extra_fps = [furthest_point_sample(x.decomposed_coordinates[i][coord_mask[i]][None, ...].float(),
                                              self.num_queries[level]).squeeze(0).long()
                        for i in range(batch_size)]
                        
          query_pos = []
          queries_ = []
          sampled_coords = []
          num = []
          for i in range(batch_size):
            proposals_pred_f = (outputs_mask[i]>0).float()
            intersection = torch.mm(proposals_pred_f.t(), proposals_pred_f)  # (nProposal, nProposal), float, cuda
            proposals_pointnum = proposals_pred_f.sum(0)  # (nProposal), float, cuda
            proposals_pn_h = proposals_pointnum.unsqueeze(-1).repeat(1, proposals_pointnum.shape[0])
            proposals_pn_v = proposals_pointnum.unsqueeze(0).repeat(proposals_pointnum.shape[0], 1)
            cross_ious = intersection / (proposals_pn_h + proposals_pn_v - intersection+1e-6)
            pick_idxs = non_max_suppression(cross_ious.cpu().numpy(), score[i].detach().cpu().numpy(), 0.7)
            proposals_pointnum = proposals_pointnum.cpu().numpy()
            pick_idxs = pick_idxs[(proposals_pointnum[pick_idxs]!=0)]
            num.append(pick_idxs.shape[0])
            if pick_idxs.shape[0]>self.num_queries[level]:
              pick_idxs = pick_idxs[:self.num_queries[level]]
            num_2 = self.num_queries[level]-pick_idxs.shape[0]

            if coord_mask[i].sum()!=0:
              pick_idxs = torch.tensor(pick_idxs).cuda()
              pick_idxs1 = extra_fps[i][:num_2]
              sampled_coords.append(torch.cat([query_pos_xyz[i][pick_idxs.long(), :],coordinates.decomposed_features[i][coord_mask[i]][pick_idxs1.long(), :]],dim=0))
              queries_.append(torch.cat([queries[i][pick_idxs.long(), :],self.np_feature_projection(pcd_features.decomposed_features[i][coord_mask[i]][pick_idxs1.long(), :])],dim=0))
            else:
              sum_idx = np.arange(score[i].shape[0])
              pick_idxs = np.append(pick_idxs,sum_idx[~np.isin(sum_idx, pick_idxs)][:num_2])  
              pick_idxs = torch.tensor(pick_idxs).cuda()
              sampled_coords.append(query_pos_xyz[i][pick_idxs.long(), :])
              queries_.append(queries[i][pick_idxs.long(), :])

          sampled_coords = torch.stack(sampled_coords)
          queries = torch.stack(queries_)
          query_pos = self.pos_enc(sampled_coords,input_range=scale)  # Batch, Dim, queries
          query_pos = self.query_projection(query_pos)
          
          #queries = self.np_feature_projection(queries)
          query_pos = query_pos.permute((2, 0, 1))
        else:
          num = []
          #print([i for i in range(len(x.decomposed_coordinates))])
          fps_idx = [furthest_point_sample(x.decomposed_coordinates[i][None, ...].float(),
                                              self.num_queries[level]).squeeze(0).long()
                        for i in range(len(x.decomposed_coordinates))]
          # fps_idx = [furthest_point_sample(torch.cat([pcd_features.decomposed_features[i],x.decomposed_coordinates[i]],dim=-1)[None, ...].float(),
          #                                      self.num_queries[level]).squeeze(0).long()
          #                for i in range(len(x.decomposed_coordinates))]

          sampled_coords = torch.stack([coordinates.decomposed_features[i][fps_idx[i].long(), :]
                                          for i in range(len(fps_idx))])
          
          
          query_pos = self.pos_enc(sampled_coords.float(),
                                      input_range=scale
                                      )  # Batch, Dim, queries
          query_pos = self.query_projection(query_pos)
          queries = torch.stack([pcd_features.decomposed_features[i][fps_idx[i].long(), :]
                                         for i in range(len(fps_idx))])
          queries = self.np_feature_projection(queries)
          #queries = torch.zeros_like(query_pos).permute((0, 2, 1))
          #queries = self.np_feature_projection(queries)
          query_pos = query_pos.permute((2, 0, 1))

        return queries,query_pos,sampled_coords,num

    def forward(self, x,file_names, point2segment=None, raw_coordinates=None, is_eval=False,targets=None):
        pcd_features, aux = self.backbone(x)

        batch_size = len(x.decomposed_coordinates)

        with torch.no_grad():
            coordinates = me.SparseTensor(features=raw_coordinates,
                                          coordinate_manager=aux[-1].coordinate_manager,
                                          coordinate_map_key=aux[-1].coordinate_map_key,
                                          device=aux[-1].device)

            coords = [coordinates]
            for _ in reversed(range(len(aux)-1)):
                coords.append(self.pooling(coords[-1]))

            coords.reverse()

        pos_encodings_pcd = self.get_pos_encs(coords)
        mask_features = self.mask_features_head(pcd_features)
        
        #point_feature = pcd_features.features.detach()
        #semantic_scores = self.linear(pcd_features.features.detach())
        #semantic_encoding = self.sem_coding(semantic_scores)
        torch.cuda.empty_cache()
        # with torch.no_grad():
            
        #     semantic_encoding = me.SparseTensor(features=semantic_encoding,
        #                                   coordinate_manager=aux[-1].coordinate_manager,
        #                                   coordinate_map_key=aux[-1].coordinate_map_key,
        #                                   device=aux[-1].device)

        #     semantic_encoding_ = [semantic_encoding]
        #     for _ in reversed(range(len(aux)-1)):
        #         semantic_encoding_.append(self.pooling(semantic_encoding_[-1]))

        #     semantic_encoding_.reverse()
        torch.cuda.empty_cache()
        if self.train_on_segments:
            mask_segments = []
            mask_seg_coords = []
            mask_segments_semantic = []
            for i, mask_feature in enumerate(mask_features.decomposed_features):
                mask_segments.append(self.scatter_fn(mask_feature, point2segment[i], dim=0))
                mask_seg_coords.append(self.scatter_fn(pos_encodings_pcd[-1][0][i], point2segment[i], dim=0))
                #mask_segments_semantic.append(self.scatter_fn(semantic_encoding_[-1].decomposed_features[i], point2segment[i], dim=0))
                #mask_segments_semantic[i] = self.sem_coding(mask_segments_semantic[i])
        #sampled_coords = None
        #mask_segments_semantic = [self.linear(mask_segments[i]) for i in range(batch_size)]
        # pt_offsets_feats = [self.offset(mask_segments[i]) for i in range(batch_size)]
        # mask_segments_offset  = [self.offset_linear(pt_offsets_feats[i]) for i in range(batch_size)]
        
        torch.cuda.empty_cache()
        predictions_class = []
        predictions_mask = []
        out_boxes = []
        quert_pos_list = []
        predictions_iou_score = []
        mins = torch.stack([coordinates.decomposed_features[i].min(dim=0)[0] for i in range(len(coordinates.decomposed_features))])
        maxs = torch.stack([coordinates.decomposed_features[i].max(dim=0)[0] for i in range(len(coordinates.decomposed_features))])
        scale = [mins,maxs]
        decomposed_aux = mask_segments
        decomposed_coord = mask_seg_coords
        #decomposed_coord = [mask_seg_coords[i]+mask_segments_semantic[i] for i in range(batch_size)]
        queries = None
        query_pos_xyz = None
        query_pos = None
        output_class = None
        iou_score = None
        outputs_mask = None
        self.hlevels1 = self.hlevels[:2]
        for decoder_counter in range(self.num_decoders):
            torch.cuda.empty_cache()
            queries,query_pos,query_pos_xyz,num = self.get_fps_query(x,coordinates,pcd_features,scale,decoder_counter,queries,query_pos,query_pos_xyz,output_class,iou_score,outputs_mask,point2segment)
            quert_pos_list.append(query_pos_xyz)
            if self.shared_decoder:
                decoder_counter = 0
            for i, hlevel in enumerate(self.hlevels1):
                torch.cuda.empty_cache()
                
                #decomposed_attn = attn_mask.decomposed_features
                curr_sample_size = max([pcd.shape[0] for pcd in decomposed_aux])
                # if not (self.max_sample_size or is_eval):
                #     curr_sample_size = min(curr_sample_size, self.sample_sizes[hlevel])
                rand_idx = []
                mask_idx = []
                for k in range(len(decomposed_aux)):
                    pcd_size = decomposed_aux[k].shape[0]
                    if pcd_size <= curr_sample_size:
                        # we do not need to sample
                        # take all points and pad the rest with zeroes and mask it
                        idx = torch.zeros(curr_sample_size,
                                          dtype=torch.long,
                                          device=queries.device)

                        midx = torch.ones(curr_sample_size,
                                          dtype=torch.bool,
                                          device=queries.device)

                        idx[:pcd_size] = torch.arange(pcd_size,
                                                      device=queries.device)

                        midx[:pcd_size] = False  # attend to first points
                    else:
                        # we have more points in pcd as we like to sample
                        # take a subset (no padding or masking needed)
                        idx = torch.randperm(decomposed_aux[k].shape[0],
                                            device=queries.device)[:curr_sample_size]
                        midx = torch.zeros(curr_sample_size,
                                          dtype=torch.bool,
                                          device=queries.device)  # attend to all

                    rand_idx.append(idx)
                    mask_idx.append(midx)

                batched_aux = torch.stack([
                    decomposed_aux[k][rand_idx[k], :] for k in range(len(rand_idx))
                ])

                # batched_attn = torch.stack([
                #     decomposed_attn[k][rand_idx[k], :] for k in range(len(rand_idx))
                # ])

                batched_pos_enc = torch.stack([
                    decomposed_coord[k][rand_idx[k], :] for k in range(len(rand_idx))
                ])

                #batched_attn.permute((0, 2, 1))[batched_attn.sum(1) == rand_idx[0].shape[0]] = False

                m = torch.stack(mask_idx)[..., None].repeat(1,1,queries.shape[1])
                #batched_attn = torch.logical_or(batched_attn, m[..., None])

                src_pcd = self.lin_squeeze[decoder_counter][i](batched_aux.permute((1, 0, 2)))
                if self.use_level_embed:
                    src_pcd += self.level_embed.weight[i]

                output = self.cross_attention[decoder_counter][i](
                    queries.permute((1, 0, 2)),
                    src_pcd,
                    memory_mask=m.repeat_interleave(self.num_heads, dim=0).permute((0, 2, 1)),
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=batched_pos_enc.permute((1, 0, 2)),
                    query_pos=query_pos
                )

                output = self.self_attention[decoder_counter][i](
                    output, tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=query_pos
                )

                # FFN
                queries = self.ffn_attention[decoder_counter][i](
                    output
                ).permute((1, 0, 2))
                if i==1:
                  output_class, outputs_mask,iou_score = self.mask_module(0,queries,
                                                          pcd_features,
                                                          mask_segments,
                                                          0,
                                                          ret_attn_mask=False,
                                                          ret_iou_score=True,
                                                          point2segment=point2segment,
                                                          coords=coords)
                  
                  new_reference_points,query_pos = self.update_ref_pos(queries,query_pos_xyz,mins,maxs)
                  out_boxes.append(new_reference_points)
                  predictions_class.append(output_class)
                  predictions_mask.append(outputs_mask)
                  predictions_iou_score.append(iou_score)
                else:
                  output_class, outputs_mask = self.mask_module(0,queries,
                                                          pcd_features,
                                                          mask_segments,
                                                          0,
                                                          ret_attn_mask=False,
                                                          ret_iou_score=False,
                                                          point2segment=point2segment,
                                                          coords=coords)
                  
                  new_reference_points,query_pos = self.update_ref_pos(queries,query_pos_xyz,mins,maxs)
                  out_boxes.append(new_reference_points)
                  predictions_class.append(output_class)
                  predictions_mask.append(outputs_mask)
                torch.cuda.empty_cache()
            decomposed_aux = [queries[p] for p in range(batch_size)]
            decomposed_coord = [(query_pos.transpose(0,1))[p] for p in range(batch_size)]
        torch.cuda.empty_cache()

        queries,query_pos,query_pos_xyz,num = self.get_fps_query(x,coordinates,pcd_features,scale,2,queries,query_pos,query_pos_xyz,output_class,iou_score,outputs_mask,point2segment)
        # for idx in range(len(file_names)):
        #   np.save('/ssd/ljh/3d_ins/Mask3D/coords/eval/'+file_names[idx],query_pos_xyz[idx].cpu().detach().numpy())
        #self.idx += 1
        quert_pos_list.append(query_pos_xyz)
        # queries = self.query_feat.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        #queries = torch.zeros_like(query_pos).permute((1, 0, 2))
        self.hlevels2 = self.hlevels[2:]
        torch.cuda.empty_cache()
        for decoder_counter in range(self.num_decoders):
            torch.cuda.empty_cache()
            for i, hlevel in enumerate(self.hlevels2):
                  torch.cuda.empty_cache()
                  output_class, outputs_mask, attn_mask = self.mask_module(1,queries,
                                                            pcd_features,
                                                            mask_segments,
                                                            len(aux) - hlevel - 1,
                                                            ret_attn_mask=True,
                                                            ret_iou_score=False,
                                                            point2segment=point2segment,
                                                            coords=coords)
                  
                  decomposed_aux = aux[hlevel].decomposed_features
                  decomposed_attn = attn_mask.decomposed_features

                  curr_sample_size = max([pcd.shape[0] for pcd in decomposed_aux])

                  if not (self.max_sample_size or is_eval):
                      curr_sample_size = min(curr_sample_size, self.sample_sizes[hlevel])
                  rand_idx = []
                  mask_idx = []
                  for k in range(len(decomposed_aux)):
                      pcd_size = decomposed_aux[k].shape[0]
                      if pcd_size <= curr_sample_size:
                          # we do not need to sample
                          # take all points and pad the rest with zeroes and mask it
                          idx = torch.zeros(curr_sample_size,
                                            dtype=torch.long,
                                            device=queries.device)

                          midx = torch.ones(curr_sample_size,
                                            dtype=torch.bool,
                                            device=queries.device)

                          idx[:pcd_size] = torch.arange(pcd_size,
                                                        device=queries.device)

                          midx[:pcd_size] = False  # attend to first points
                      else:
                          # we have more points in pcd as we like to sample
                          # take a subset (no padding or masking needed)
                          idx = torch.randperm(decomposed_aux[k].shape[0],
                                              device=queries.device)[:curr_sample_size]
                          midx = torch.zeros(curr_sample_size,
                                            dtype=torch.bool,
                                            device=queries.device)  # attend to all

                      rand_idx.append(idx)
                      mask_idx.append(midx)

                  batched_aux = torch.stack([
                      decomposed_aux[k][rand_idx[k], :] for k in range(len(rand_idx))
                  ])

                  batched_attn = torch.stack([
                      decomposed_attn[k][rand_idx[k], :] for k in range(len(rand_idx))
                  ])

                  batched_pos_enc = torch.stack([
                      pos_encodings_pcd[hlevel][0][k][rand_idx[k], :]for k in range(len(rand_idx))
                  ])

                  batched_attn.permute((0, 2, 1))[batched_attn.sum(1) == rand_idx[0].shape[0]] = False

                  m = torch.stack(mask_idx)
                  batched_attn = torch.logical_or(batched_attn, m[..., None])

                  src_pcd = self.assign_squeeze_attention[i](batched_aux.permute((1, 0, 2)))
                  if self.use_level_embed:
                      src_pcd += self.level_embed.weight[i]

                  output = self.assign_cross_attention[i](
                      queries.permute((1, 0, 2)),
                      src_pcd,
                      memory_mask=batched_attn.repeat_interleave(self.num_heads, dim=0).permute((0, 2, 1)),
                      memory_key_padding_mask=None,  # here we do not apply masking on padded region
                      pos=batched_pos_enc.permute((1, 0, 2)),
                      query_pos=query_pos
                  )

                  output = self.assign_self_attention[i](
                      output, tgt_mask=None,
                      tgt_key_padding_mask=None,
                      query_pos=query_pos
                  )
                  
                  # FFN
                  queries = self.assign_ffn_attention[i](
                      output
                  ).permute((1, 0, 2))
                  new_reference_points,query_pos = self.update_ref_pos(queries,query_pos_xyz,mins,maxs)
                  out_boxes.append(new_reference_points)
                  predictions_class.append(output_class)
                  predictions_mask.append(outputs_mask)
        torch.cuda.empty_cache()
        output_class, outputs_mask,iou_score = self.mask_module(1,queries,
                                                      pcd_features,
                                                      mask_segments,
                                                      0,
                                                      ret_attn_mask=False,
                                                      ret_iou_score=True,
                                                      point2segment=point2segment,
                                                      coords=coords)
        new_reference_points,query_pos = self.update_ref_pos(queries,query_pos_xyz,mins,maxs)
        out_boxes.append(new_reference_points)
        predictions_class.append(output_class)
        predictions_mask.append(outputs_mask)
        predictions_iou_score.append(iou_score)                                              

        torch.cuda.empty_cache()                                             
        # if not os.path.exists(path+'/raw'):
        #   os.makedirs(path+'/raw')
        # if not os.path.exists(path+'/sample'):
        #   os.makedirs(path+'/sample')
        # xyz = coordinates.decomposed_features[0].data.cpu().numpy()
        #   #input2,_ = self.rotate_point_cloud_by_angle(input2)
        # sample_xyz = quert_pos_list[0][0].data.cpu().numpy()

        # self.idx +=1
        # np.save(path+'/raw/'+str(self.idx)+'.npy',xyz)
        # np.save(path+'/sample/'+str(self.idx)+'.npy', sample_xyz)
        return {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'score':predictions_iou_score,
            'pred_boxes':out_boxes[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class, predictions_mask,out_boxes=out_boxes
            ),
            'sampled_coords': quert_pos_list,
            #'semantic_scores': semantic_scores
        }
        
    def update_ref_pos(self,queries,refpoint_embed,mins,maxs):

        delta_unsig = self._bbox_embed(queries)
        new_reference_points = (delta_unsig + refpoint_embed)
        #refpoint_embed = new_reference_points.detach()
        query_pos = self.pos_enc(new_reference_points.detach()[...,:3].float(),
                          input_range=[mins, maxs]
                          )
        #query_pos = self.query_projection(query_pos)
        query_pos = query_pos.permute((2, 0, 1))
        return new_reference_points,query_pos

    def get_proposals_iou(self,output_masks,feature,coordinate,spatial_shape,scale):
        iou_scores = []
        for i in range(len(output_masks)):
            torch.cuda.empty_cache()
            mask = output_masks[i].T.sigmoid() > 0.5
            mask[torch.arange(mask.shape[0]).cuda(),output_masks[i].T.sigmoid().max(1)[1]] = True
            # proposals_idx = torch.stack(torch.where(mask)).T
            # proposals_counts = torch.bincount(torch.where(mask)[0])
            # # if (proposals_counts==0).sum()!=0:
            # #   print('1')
            # proposals_offset = torch.cumsum(proposals_counts, dim=0)
            # proposals_offset = torch.cat([torch.tensor([0]).cuda(),proposals_offset],dim=0).int()
            # batch_idx = proposals_idx[:, 0].long()
            # c_idxs = proposals_idx[:, 1].cuda()
            # feats = feature[i][c_idxs.long()]
            # coords = coordinate[i][c_idxs.long()]
            # coords_min = sec_min(coords, proposals_offset)
            # coords_max = sec_max(coords, proposals_offset)
            # torch.cuda.empty_cache()
            # mask = ((coordinate[i][None,...]>=coords_min[:,None,:]).sum(-1)==3) & ((coordinate[i][None,...]<=coords_max[:,None,:]).sum(-1)==3)
            torch.cuda.empty_cache()
            proposals_idx = torch.stack(torch.where(mask)).T
            proposals_counts = torch.bincount(torch.where(mask)[0])
            # if (proposals_counts==0).sum()!=0:
            #   print('1')
            proposals_offset = torch.cumsum(proposals_counts, dim=0)
            proposals_offset = torch.cat([torch.tensor([0]).cuda(),proposals_offset],dim=0).int()
            batch_idx = proposals_idx[:, 0].long()
            c_idxs = proposals_idx[:, 1].cuda()
            feats = feature[i][c_idxs.long()]
            coords = coordinate[i][c_idxs.long()]
            batch_new_offset = (torch.arange(0,proposals_offset.shape[0])*100).cuda().int()
            index = pointops.furthestsampling(coords, proposals_offset,batch_new_offset).long()
            coords = coords[index]
            feats = feats[index]
            batch_idx = batch_idx[index]
            proposals_idx = proposals_idx[index]
            coords_min = sec_min(coords, batch_new_offset)
            coords_max = sec_max(coords, batch_new_offset)

            clusters_scale = 1 / ((coords_max - coords_min) / spatial_shape).max(1)[0] - 0.01
            clusters_scale = torch.clamp(clusters_scale, min=None, max=scale)
            coords_min = coords_min * clusters_scale[:, None]
            coords_max = coords_max * clusters_scale[:, None]
            clusters_scale = clusters_scale[batch_idx]
            coords = coords * clusters_scale[:, None]
            coords_min = coords_min[batch_idx]
            coords -= coords_min
            assert coords.shape.numel() == ((coords >= 0) * (coords < spatial_shape)).sum()
            coords = coords.long()
            coords = torch.cat([proposals_idx[:, 0].view(-1, 1).long().cpu(), coords.cpu()], 1)
            out_coords, inp_map, out_map = voxelization_idx(coords, int(proposals_idx[-1, 0]) + 1)
            out_feats = voxelization(feats, out_map.cuda())
            voxelization_feats = me.SparseTensor(out_feats,
                                                     out_coords.int().cuda())
            voxelization_feats = self.tiny_unet(voxelization_feats)
            voxelization_feats = self.global_pool(voxelization_feats)
            iou_scores.append(self.iou_score_linear(voxelization_feats).sigmoid())
        iou_scores = torch.stack(iou_scores)
        return iou_scores

    def global_pool(self, x, expand=False):
        indices = x.coordinates[:, 0]
        batch_counts = torch.bincount(indices)
        batch_offset = torch.cumsum(batch_counts, dim=0)
        pad = batch_offset.new_full((1,), 0)
        batch_offset = torch.cat([pad, batch_offset]).int()
        x_pool = global_avg_pool(x.features, batch_offset)
        if not expand:
            return x_pool
        x_pool_expand = x_pool[indices.long()]
        x.features = torch.cat((x.features, x_pool_expand), dim=1)
        return x

    def mask_module(self,lvl, query_feat, mask_features, mask_segments, num_pooling_steps, ret_attn_mask=True,ret_iou_score=False,
                                 point2segment=None, coords=None):
        
        query_feat = self.decoder_norm(query_feat)
        mask_embed = self.mask_embed_head(query_feat)
        outputs_class = self.class_embed_head(query_feat)
      
        #iou_score = self.out_score(query_feat).sigmoid()
        output_masks = []
        torch.cuda.empty_cache()
        if point2segment is not None:
            output_segments = []
            for i in range(len(mask_segments)):
                output_segments.append(mask_segments[i] @ mask_embed[i].T)
                output_masks.append(output_segments[-1][point2segment[i]])
        else:
            for i in range(mask_features.C[-1, 0] + 1):
                output_masks.append(mask_features.decomposed_features[i] @ mask_embed[i].T)
        if ret_iou_score:
          score_feature = []
          # iou_features = self.iou_features_head(mask_features.detach())
          # iou_score = self.get_proposals_iou(output_masks,iou_features.decomposed_features,coords[-1].decomposed_features,20,3)
          iou_score = self.iou_score(query_feat.detach()).sigmoid()
          

        output_masks = torch.cat(output_masks)
        

        outputs_mask = me.SparseTensor(features=output_masks,
                                       coordinate_manager=mask_features.coordinate_manager,
                                       coordinate_map_key=mask_features.coordinate_map_key)
        torch.cuda.empty_cache()
        if ret_attn_mask:
            attn_mask = outputs_mask
            for _ in range(num_pooling_steps):
                attn_mask = self.pooling(attn_mask.float())
                #mask_features = self.pooling(mask_features.float())
            # mask = []
            # for i in range(len(attn_mask.decomposed_features)):
            #     
            #     significance_score = self.score_assignment_step(attn_mask.decomposed_features[i].T,mask_features.decomposed_features[i])
            #     torch.cuda.empty_cache()
            #     mask.append(self.inverse_transform_sampling(significance_score,significance_score.shape[-1], 128*4**(2-num_pooling_steps)))
            # mask = torch.cat(mask)
            # attn_mask = me.SparseTensor(features= (mask!=1),
            #                             coordinate_manager=attn_mask.coordinate_manager,
            #                             coordinate_map_key=attn_mask.coordinate_map_key)
            attn_mask = me.SparseTensor(features=(F.one_hot(attn_mask.F.argmax(1),num_classes=query_feat.shape[1])!=1),
                                        coordinate_manager=attn_mask.coordinate_manager,
                                        coordinate_map_key=attn_mask.coordinate_map_key)

            if ret_iou_score:
                return outputs_class, output_segments, attn_mask,iou_score
            else:
                return outputs_class, output_segments, attn_mask

        if ret_iou_score:
            return outputs_class, output_segments,iou_score
        else:
            return outputs_class, output_segments

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks,out_boxes=None,out_score=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
              {"pred_logits": a, "pred_masks": b, "pred_boxes": c}
              for a, b ,c in zip(outputs_class[:-1], outputs_seg_masks[:-1],out_boxes[:-1])
        ]



class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        self.orig_ch = channels
        super(PositionalEncoding3D, self).__init__()
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor, input_range=None):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        pos_x, pos_y, pos_z = tensor[:, :, 0], tensor[:, :, 1], tensor[:, :, 2]
        sin_inp_x = torch.einsum("bi,j->bij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("bi,j->bij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("bi,j->bij", pos_z, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)

        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)

        emb = torch.cat((emb_x, emb_y, emb_z), dim=-1)
        return emb[:, :, :self.orig_ch].permute((0, 2, 1))


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask = None,
                     tgt_key_padding_mask= None,
                     query_pos = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask= None,
                    tgt_key_padding_mask = None,
                    query_pos = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt,
                tgt_mask = None,
                tgt_key_padding_mask = None,
                query_pos = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask = None,
                     memory_key_padding_mask = None,
                     pos = None,
                     query_pos = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask = None,
                    memory_key_padding_mask = None,
                    pos = None,
                    query_pos = None):
        tgt2 = self.norm(tgt)

        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask = None,
                memory_key_padding_mask = None,
                pos = None,
                query_pos = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)

class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
