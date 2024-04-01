# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn

from mmdet.models.builder import LOSSES


def proxy_contrastive_iou_loss(feature, target, label, proxy, scale=12, reduction='mean'):
    """Calculate the proxy contrastive loss of semantic prototype 
        and query pred.

    Args:
        feature (torch.Tensor): the query it self with shape [n, c]
        target (torch.Tensor): Cls targets (gt) with shape [n, ]
        label (torch.Tensor): label mask with shape [c, n]
        proxy (torch.Tensor): semantic proxy with shape [n, c]

    Returns:
        torch.Tensor: Proxy contrastive loss between predictions and targets.
    """
    num = feature.shape[0]
    pred = F.linear(feature, F.normalize(proxy, p=2, dim=1)) # (N, c)
    pred = torch.masked_select(pred.transpose(1, 0), label)  # N,
    pred = pred.unsqueeze(1)  # (N, 1)

    feature = torch.matmul(feature, feature.transpose(1, 0))  # (N, N)
    label_matrix = target.unsqueeze(1) == target.unsqueeze(0)  # (N, N)
    
    index_label = torch.LongTensor([i for i in range(feature.shape[0])])  # generate index label
    index_matrix = index_label.unsqueeze(1) == index_label.unsqueeze(0)  # get index matrix
    
    feature = feature * ~label_matrix  # get negative matrix
    feature = feature.masked_fill(feature < 1e-6, -np.inf)  # (N, N)
    
    logits = torch.cat([pred, feature], dim=1)  # (N, 1+N)
    label = torch.zeros(logits.size(0), dtype=torch.long).to(feature.device)
    loss = F.nll_loss(F.log_softmax(scale * logits, dim=1), label, reduction=reduction)

    # return loss / num
    return loss


@LOSSES.register_module()
class ProxyContrastiveLoss(nn.Module):
    """Calculate the proxy contrastive loss of semantic prototype 
        and query pred.

    Args:
        reduction (str): Method to reduce losses.
            The valid reduction method are none, sum or mean.
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.
    """

    def __init__(self, num_classes, scale=12,
                 reduction='mean', loss_weight=1.0):
        super(ProxyContrastiveLoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.label = torch.LongTensor([i for i in range(num_classes)])
        self.scale = scale

    def forward(self,
                feature,
                target,
                proxy,
                **kwargs):
        """Forward function of loss calculation.

        Args:
            feature (torch.Tensor): the query it self with shape [n, c]
            target (torch.Tensor): Cls targets (gt) with shape [n, ]
            proxy (torch.Tensor): semantic proxy with shape [c, c]

        Returns:
            torch.Tensor: Proxy contrastive loss between predictions and targets.
        """
        reduction = self.reduction
        label = (self.label.unsqueeze(1).to(feature.device) == target.unsqueeze(0))  # (C, N)
        
        return proxy_contrastive_iou_loss(
            feature,
            target,
            label,
            proxy,
            self.scale,
            reduction=reduction) * self.loss_weight
