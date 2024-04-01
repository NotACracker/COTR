
import torch, torch.nn as nn, torch.nn.functional as F
from mmcv.runner import BaseModule
from mmdet.models import HEADS
from mmcv.cnn import xavier_init

@HEADS.register_module()
class MaskPredictorHead(BaseModule):
    def __init__(
        self, nbr_classes=20, in_dims=64, hidden_dims=128, 
        out_dims=None, scale_h=2, scale_w=2, scale_z=2, 
        use_checkpoint=True, mask_dims=256, mask_classification=True
    ):
        super().__init__()
        self.scale_h = scale_h
        self.scale_w = scale_w
        self.scale_z = scale_z
        self.mask_classification = mask_classification

        out_dims = in_dims if out_dims is None else out_dims

        self.decoder = nn.Sequential(
            nn.Linear(in_dims, hidden_dims),
            nn.Softplus(),
            nn.Linear(hidden_dims, out_dims)
        )
        
        self.classes = nbr_classes
        self.use_checkpoint = use_checkpoint

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(in_dims, nbr_classes + 1) # need to add a no-object class
        self.mask_embed = MLP(in_dims, hidden_dims, mask_dims, 3)
        self.init_weights()
    
    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        xavier_init(self.class_embed, distribution='uniform', bias=0.)

    def forward(self, occ_feature, query):
        """
        input: 
            occ_feature: output of occ encoder/decoder
                size:[bs, c, w, h, z]
        output:
            pred_logits: pred class for each mask query
                size:[bs, N, numclass+1]
            pred_masks: pred mask for each query
                size:[bs, N, W, H, Z]
            aux: pred class & mask for each decoder layer
                when deep_supervision = True
        """
        occ_feature = occ_feature.permute(0, 2, 3, 4, 1) # [bs, c, w, h, z] -> [bs, w, h, z, c]
        if self.use_checkpoint:
            occ_feature = torch.utils.checkpoint.checkpoint(self.decoder, occ_feature)
            # logits = torch.utils.checkpoint.checkpoint(self.classifier, fused)
        else:
            occ_feature = self.decoder(occ_feature)
            # logits = self.classifier(fused)
        # logits = logits.permute(0, 4, 1, 2, 3) # [bs, c, w, h, z]
        occ_feature = occ_feature.permute(0, 4, 1, 2, 3) # [bs, c, w, h, z]

        query = query.permute(0, 2, 1, 3)
        if self.mask_classification:
            output_class = self.class_embed(query)
            # pred_logits = output_class[-1]
        
        # [layer, bs, N, c]
        mask_embed = self.mask_embed(query)
        outputs_seg_masks = torch.einsum("lbnc,bcwhz->lbnwhz", mask_embed, occ_feature)

        out = {"cls_preds": output_class, "mask_preds": outputs_seg_masks}
        
        # tpv_occ = self.classifier(fused.permute(0,2,3,4,1))
    
        return occ_feature, out

class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

@HEADS.register_module()
class MaskPredictorHead_Group(BaseModule):
    def __init__(
        self, nbr_classes=20, in_dims=64, hidden_dims=128, 
        out_dims=None, scale_h=2, scale_w=2, scale_z=2, 
        use_checkpoint=True, mask_dims=256, mask_classification=True,
        group_detr=1, group_classes=[17]
    ):
        super().__init__()
        self.scale_h = scale_h
        self.scale_w = scale_w
        self.scale_z = scale_z
        self.mask_classification = mask_classification

        out_dims = in_dims if out_dims is None else out_dims

        self.decoder = nn.Sequential(
            nn.Linear(in_dims, hidden_dims),
            nn.Softplus(),
            nn.Linear(hidden_dims, out_dims)
        )

        self.classes = nbr_classes
        self.use_checkpoint = use_checkpoint

        self.group_detr = group_detr
        # output FFNs
        self.class_embed = nn.ModuleList()
        if self.mask_classification:
            for i in range(self.group_detr):
                self.class_embed.append(nn.Linear(in_dims, group_classes[i] + 1)) # need to add a no-object class

        self.mask_embed = MLP(in_dims, hidden_dims, mask_dims, 3)
        self.init_weights()
        # do we need to split mask_embed to different group ?
        # self.mask_embed = MLP(in_dims, hidden_dims, mask_dims, 3)
    
    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for group_index in range(self.group_detr):
            xavier_init(self.class_embed[group_index], distribution='uniform', bias=0.)

    def forward(self, occ_feature, query):
        """
        input: 
            occ_feature: output of occ encoder/decoder
                size:[bs, c, w, h, z]
        output:
            pred_logits: pred class for each mask query
                size:[bs, N, numclass+1]
            pred_masks: pred mask for each query
                size:[bs, N, W, H, Z]
            aux: pred class & mask for each decoder layer
                when deep_supervision = True
        """
        occ_feature = occ_feature.permute(0, 2, 3, 4, 1) # [bs, c, w, h, z] -> [bs, w, h, z, c]
        if self.use_checkpoint:
            occ_feature = torch.utils.checkpoint.checkpoint(self.decoder, occ_feature)
            # logits = torch.utils.checkpoint.checkpoint(self.classifier, fused)
        else:
            occ_feature = self.decoder(occ_feature)
            # logits = self.classifier(fused)
        # logits = logits.permute(0, 4, 1, 2, 3) # [bs, c, w, h, z]
        occ_feature = occ_feature.permute(0, 4, 1, 2, 3) # [bs, c, w, h, z]

        query = query.permute(0, 2, 1, 3)
        num_query = query.shape[-2]
        num_query_per_group = num_query // self.group_detr
        output_classes = []
        if self.mask_classification:
            if not self.training:
                output_class = self.class_embed[0](query)
                output_classes.append(output_class)
            else:
                for group_index in range(self.group_detr):
                    group_query_start = group_index * num_query_per_group
                    group_query_end = (group_index+1) * num_query_per_group
                    group_query = query[:, :, group_query_start:group_query_end, :]
                    output_class = self.class_embed[group_index](group_query)
                    output_classes.append(output_class)
            # for group_index in range(self.group_detr):
            #     group_query_start = group_index * num_query_per_group
            #     group_query_end = (group_index+1) * num_query_per_group
            #     group_query = query[:, :, group_query_start:group_query_end, :]
            #     output_class = self.class_embed[group_index](group_query)
            #     output_classes.append(output_class)
            # pred_logits = output_class[-1]
        
        # [layer, bs, N, c]
        mask_embed = self.mask_embed(query)
        outputs_seg_masks = torch.einsum("lbnc,bcwhz->lbnwhz", mask_embed, occ_feature)
        
        out = {"cls_preds": output_classes, "mask_preds": outputs_seg_masks}
        # tpv_occ = self.classifier(fused.permute(0,2,3,4,1))
        return occ_feature, out