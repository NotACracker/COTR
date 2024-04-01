# Copyright (c) OpenMMLab. All rights reserved.
"""copy from
https://github.com/ZwwWayne/K-Net/blob/main/knet/det/mask_pseudo_sampler.py."""

import torch

from mmdet.core.bbox.builder import BBOX_SAMPLERS
from mmdet.core.bbox.samplers.base_sampler import BaseSampler
from mmdet.core.bbox.samplers.sampling_result import SamplingResult

class MaskSamplingResult(SamplingResult):
    """Mask sampling result."""

    def __init__(self, pos_inds, neg_inds, masks, gt_masks, assign_result,
                 gt_flags):
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        self.pos_masks = masks[pos_inds]
        self.neg_masks = masks[neg_inds]
        self.pos_is_gt = gt_flags[pos_inds]

        self.num_gts = gt_masks.shape[0]
        self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        if gt_masks.numel() == 0:
            # hack for index error case
            assert self.pos_assigned_gt_inds.numel() == 0
            self.pos_gt_masks = torch.empty_like(gt_masks)
        else:
            self.pos_gt_masks = gt_masks[self.pos_assigned_gt_inds, :]

        if assign_result.labels is not None:
            self.pos_gt_labels = assign_result.labels[pos_inds]
        else:
            self.pos_gt_labels = None

    @property
    def masks(self):
        """torch.Tensor: concatenated positive and negative boxes"""
        return torch.cat([self.pos_masks, self.neg_masks])

    def __nice__(self):
        data = self.info.copy()
        data['pos_masks'] = data.pop('pos_masks').shape
        data['neg_masks'] = data.pop('neg_masks').shape
        parts = [f"'{k}': {v!r}" for k, v in sorted(data.items())]
        body = '    ' + ',\n    '.join(parts)
        return '{\n' + body + '\n}'

    @property
    def info(self):
        """Returns a dictionary of info about the object."""
        return {
            'pos_inds': self.pos_inds,
            'neg_inds': self.neg_inds,
            'pos_masks': self.pos_masks,
            'neg_masks': self.neg_masks,
            'pos_is_gt': self.pos_is_gt,
            'num_gts': self.num_gts,
            'pos_assigned_gt_inds': self.pos_assigned_gt_inds,
        }
 
@BBOX_SAMPLERS.register_module()
class MaskPseudoSamplerEn(BaseSampler):
    """A pseudo sampler that does not do sampling actually."""

    def __init__(self, use_camera_mask, use_lidar_mask, **kwargs):
        self.use_camera_mask = use_camera_mask
        self.use_lidar_mask = use_lidar_mask

    def _sample_pos(self, **kwargs):
        """Sample positive samples."""
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        """Sample negative samples."""
        raise NotImplementedError

    def sample(self, assign_result, masks, gt_masks, **kwargs):
        """Directly returns the positive and negative indices  of samples.

        Args:
            assign_result (:obj:`AssignResult`): Assigned results
            masks (torch.Tensor): Bounding boxes
            gt_masks (torch.Tensor): Ground truth boxes
        Returns:
            :obj:`SamplingResult`: sampler results
        """
        if self.use_camera_mask:
            camera_mask = kwargs['mask_camera']
            masks = masks[:, camera_mask]
            gt_masks = gt_masks[:, camera_mask]
        elif self.use_lidar_mask:
            lidar_mask = kwargs['mask_lidar']
            masks = masks[:, lidar_mask]
            gt_masks = gt_masks[:, lidar_mask]
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        gt_flags = masks.new_zeros(masks.shape[0], dtype=torch.uint8)
        sampling_result = MaskSamplingResult(pos_inds, neg_inds, masks,
                                             gt_masks, assign_result, gt_flags)
        return sampling_result
