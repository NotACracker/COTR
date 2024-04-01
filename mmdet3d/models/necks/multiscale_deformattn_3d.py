import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair, _triple
from torch.utils.checkpoint import checkpoint
from mmcv.cnn import Conv3d, ConvModule, caffe2_xavier_init, normal_init, xavier_init
from mmcv.cnn.bricks.transformer import (build_positional_encoding,
                                         build_transformer_layer_sequence)
# from projects.mmdet3d_plugin.utils.point_generator import MlvlPointGenerator
from mmcv.cnn import build_norm_layer

# from mmdet.models.utils.transformer import MultiScaleDeformableAttention
from ..model_utils.multi_scale_deform_attn_3d import MultiScaleDeformableAttention3D

from mmdet.models import NECKS
from mmcv.runner import BaseModule, ModuleList

import pdb

class MlvlPointGenerator:
    """Standard points generator for multi-level (Mlvl) feature maps in 2D
    points-based detectors.

    Args:
        strides (list[int] | list[tuple[int, int]]): Strides of anchors
            in multiple feature levels in order (w, h).
        offset (float): The offset of points, the value is normalized with
            corresponding stride. Defaults to 0.5.
    """

    def __init__(self, strides, offset=0.5):
        self.strides = [_triple(stride) for stride in strides]
        self.offset = offset

    @property
    def num_levels(self):
        """int: number of feature levels that the generator will be applied"""
        return len(self.strides)

    @property
    def num_base_priors(self):
        """list[int]: The number of priors (points) at a point
        on the feature grid"""
        return [1 for _ in range(len(self.strides))]

    def _meshgrid(self, x, y, z):
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        
        return zz.reshape(-1), yy.reshape(-1), xx.reshape(-1)

    def grid_priors(self,
                    featmap_sizes,
                    dtype=torch.float32,
                    device='cuda',
                    with_stride=False):
        """Generate grid points of multiple feature levels.

        Args:
            featmap_sizes (list[tuple]): List of feature map sizes in
                multiple feature levels, each size arrange as
                as (h, w).
            dtype (:obj:`dtype`): Dtype of priors. Default: torch.float32.
            device (str): The device where the anchors will be put on.
            with_stride (bool): Whether to concatenate the stride to
                the last dimension of points.

        Return:
            list[torch.Tensor]: Points of  multiple feature levels.
            The sizes of each tensor should be (N, 2) when with stride is
            ``False``, where N = width * height, width and height
            are the sizes of the corresponding feature level,
            and the last dimension 2 represent (coord_x, coord_y),
            otherwise the shape should be (N, 4),
            and the last dimension 4 represent
            (coord_x, coord_y, stride_w, stride_h).
        """

        assert self.num_levels == len(featmap_sizes)
        multi_level_priors = []
        for i in range(self.num_levels):
            priors = self.single_level_grid_priors(
                featmap_sizes[i],
                level_idx=i,
                dtype=dtype,
                device=device,
                with_stride=with_stride)
            multi_level_priors.append(priors)
        return multi_level_priors

    def single_level_grid_priors(self,
                                 featmap_size,
                                 level_idx,
                                 dtype=torch.float32,
                                 device='cuda',
                                 with_stride=False):
        """Generate grid Points of a single level.

        Note:
            This function is usually called by method ``self.grid_priors``.

        Args:
            featmap_size (tuple[int]): Size of the feature maps, arrange as
                (h, w).
            level_idx (int): The index of corresponding feature map level.
            dtype (:obj:`dtype`): Dtype of priors. Default: torch.float32.
            device (str, optional): The device the tensor will be put on.
                Defaults to 'cuda'.
            with_stride (bool): Concatenate the stride to the last dimension
                of points.

        Return:
            Tensor: Points of single feature levels.
            The shape of tensor should be (N, 2) when with stride is
            ``False``, where N = width * height, width and height
            are the sizes of the corresponding feature level,
            and the last dimension 2 represent (coord_x, coord_y),
            otherwise the shape should be (N, 4),
            and the last dimension 4 represent
            (coord_x, coord_y, stride_w, stride_h).
        """
        
        assert with_stride == False
        
        feat_X, feat_Y, feat_Z = featmap_size
        stride_X, stride_Y, stride_Z = self.strides[level_idx]
        shift_x = (torch.arange(0, feat_X, device=device) + self.offset) * stride_X
        # keep featmap_size as Tensor instead of int, so that we
        # can convert to ONNX correctly
        shift_x = shift_x.to(dtype)

        shift_y = (torch.arange(0, feat_Y, device=device) +
                   self.offset) * stride_Y
        # keep featmap_size as Tensor instead of int, so that we
        # can convert to ONNX correctly
        shift_y = shift_y.to(dtype)
        
        shift_z = (torch.arange(0, feat_Z, device=device) +
                   self.offset) * stride_Z
        # keep featmap_size as Tensor instead of int, so that we
        # can convert to ONNX correctly
        shift_z = shift_z.to(dtype)
        shift_zz, shift_yy, shift_xx = self._meshgrid(shift_x, shift_y, shift_z)
        
        # grid-sample use reverse-ordered coordinates\
        shifts = torch.stack([shift_zz, shift_yy, shift_xx], dim=-1)
        all_points = shifts.to(device)
        
        # tmp = all_points.view(*featmap_size, -1)
        
        return all_points

    def valid_flags(self, featmap_sizes, pad_shape, device='cuda'):
        """Generate valid flags of points of multiple feature levels.

        Args:
            featmap_sizes (list(tuple)): List of feature map sizes in
                multiple feature levels, each size arrange as
                as (h, w).
            pad_shape (tuple(int)): The padded shape of the image,
                 arrange as (h, w).
            device (str): The device where the anchors will be put on.

        Return:
            list(torch.Tensor): Valid flags of points of multiple levels.
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_flags = []
        for i in range(self.num_levels):
            point_stride = self.strides[i]
            feat_h, feat_w = featmap_sizes[i]
            h, w = pad_shape[:2]
            valid_feat_h = min(int(np.ceil(h / point_stride[1])), feat_h)
            valid_feat_w = min(int(np.ceil(w / point_stride[0])), feat_w)
            flags = self.single_level_valid_flags((feat_h, feat_w),
                                                  (valid_feat_h, valid_feat_w),
                                                  device=device)
            multi_level_flags.append(flags)
        return multi_level_flags

    def single_level_valid_flags(self,
                                 featmap_size,
                                 valid_size,
                                 device='cuda'):
        """Generate the valid flags of points of a single feature map.

        Args:
            featmap_size (tuple[int]): The size of feature maps, arrange as
                as (h, w).
            valid_size (tuple[int]): The valid size of the feature maps.
                The size arrange as as (h, w).
            device (str, optional): The device where the flags will be put on.
                Defaults to 'cuda'.

        Returns:
            torch.Tensor: The valid flags of each points in a single level \
                feature map.
        """
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        return valid

    def sparse_priors(self,
                      prior_idxs,
                      featmap_size,
                      level_idx,
                      dtype=torch.float32,
                      device='cuda'):
        """Generate sparse points according to the ``prior_idxs``.

        Args:
            prior_idxs (Tensor): The index of corresponding anchors
                in the feature map.
            featmap_size (tuple[int]): feature map size arrange as (w, h).
            level_idx (int): The level index of corresponding feature
                map.
            dtype (obj:`torch.dtype`): Date type of points. Defaults to
                ``torch.float32``.
            device (obj:`torch.device`): The device where the points is
                located.
        Returns:
            Tensor: Anchor with shape (N, 2), N should be equal to
            the length of ``prior_idxs``. And last dimension
            2 represent (coord_x, coord_y).
        """
        height, width = featmap_size
        x = (prior_idxs % width + self.offset) * self.strides[level_idx][0]
        y = ((prior_idxs // width) % height +
             self.offset) * self.strides[level_idx][1]
        prioris = torch.stack([x, y], 1).to(dtype)
        prioris = prioris.to(device)
        return prioris

# designed for multi-scale deformable attention: 
# each pixel within every level will access multi-scale features
@NECKS.register_module()
class MSDeformAttnPixelDecoder3D(BaseModule):
    def __init__(
            self,
            in_channels=[256, 512, 1024, 2048],
            strides=[4, 8, 16, 32],
            feat_channels=256,
            out_channels=256,
            ms_out_channels=[64, 128, 256],
            num_outs=3,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention3D',
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=False,
                        norm_cfg=None,
                        init_cfg=None),
                    feedforward_channels=1024,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm')),
                init_cfg=None),
            positional_encoding=dict(
                type='SinePositionalEncoding3D',
                num_feats=128,
                normalize=True),
            init_cfg=None,
            reverse=False,
            size=(50, 50, 16),
            with_cp=True):
        
        super().__init__(init_cfg=init_cfg)
        
        self.reverse = reverse
        self.size = size
        self.ms_out_channels = ms_out_channels
        self.strides = strides
        self.num_input_levels = len(in_channels)
        self.num_encoder_levels = encoder.transformerlayers.attn_cfgs.num_levels
        assert self.num_encoder_levels >= 1
        
        # build input conv for channel adapation
        # from top to down (low to high resolution)
        input_conv_list = []
        for i in range(self.num_input_levels - 1,
                self.num_input_levels - self.num_encoder_levels - 1, -1):
            input_conv = ConvModule(
                in_channels[i],
                feat_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                conv_cfg=conv_cfg,
                act_cfg=None,
                bias=True)
            input_conv_list.append(input_conv)

        self.input_convs = ModuleList(input_conv_list)
        self.encoder = build_transformer_layer_sequence(encoder)
        self.postional_encoding = build_positional_encoding(positional_encoding)
        self.level_encoding = nn.Embedding(self.num_encoder_levels, feat_channels)
        
        # fpn-like structure
        self.lateral_convs = ModuleList()
        self.output_convs = ModuleList()
        self.use_bias = norm_cfg is None
        # from top to down (low to high resolution)
        # fpn for the rest features that didn't pass in encoder
        for i in range(self.num_input_levels - self.num_encoder_levels - 1, -1, -1):
            lateral_conv = ConvModule(
                in_channels[i],
                feat_channels,
                kernel_size=1,
                bias=self.use_bias,
                norm_cfg=norm_cfg,
                conv_cfg=conv_cfg,
                act_cfg=None)
            
            output_conv = ConvModule(
                feat_channels,
                feat_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=self.use_bias,
                norm_cfg=norm_cfg,
                conv_cfg=conv_cfg,
                act_cfg=act_cfg)
            
            self.lateral_convs.append(lateral_conv)
            self.output_convs.append(output_conv)

        self.mask_feature = Conv3d(
            feat_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.num_outs = num_outs
        self.point_generator = MlvlPointGenerator(strides)

        if self.reverse:
            self.with_cp = with_cp
            self.fuse_conv = ConvModule(
                out_channels + sum(self.ms_out_channels),
                ms_out_channels[-1],
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                conv_cfg=dict(type='Conv3d'),
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                act_cfg=dict(type='ReLU',inplace=True))
            self.reduce_convs = nn.ModuleList([
                ConvModule(out_channels, ms_out_channel, kernel_size=1, stride=1,
                    padding=0, bias=False, conv_cfg=dict(type='Conv3d'),
                    norm_cfg=dict(type='SyncBN', requires_grad=True),
                    act_cfg=dict(type='ReLU',inplace=True))
                        for ms_out_channel in ms_out_channels])
    
    def init_weights(self):
        """Initialize weights."""
        for i in range(0, self.num_encoder_levels):
            xavier_init(
                self.input_convs[i].conv,
                gain=1,
                bias=0,
                distribution='uniform')

        for i in range(0, self.num_input_levels - self.num_encoder_levels):
            caffe2_xavier_init(self.lateral_convs[i].conv, bias=0)
            caffe2_xavier_init(self.output_convs[i].conv, bias=0)

        caffe2_xavier_init(self.mask_feature, bias=0)
        normal_init(self.level_encoding, mean=0, std=1)
        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

        for layer in self.encoder.layers:
            for attn in layer.attentions:
                if isinstance(attn, MultiScaleDeformableAttention3D):
                    attn.init_weights()
                
        if self.reverse:
            caffe2_xavier_init(self.fuse_conv, bias=0)
            for i in range(len(self.ms_out_channels)):
                caffe2_xavier_init(self.reduce_convs[i], bias=0)
    
    def forward(self, feats):
        batch_size = feats[0].shape[0]
        encoder_input_list = []
        padding_mask_list = []
        level_positional_encoding_list = []
        spatial_shapes = []
        reference_points_list = []
        
        for i in range(self.num_encoder_levels):
            # 从最后一层输入开始
            level_idx = self.num_input_levels - i - 1
            feat = feats[level_idx]
            feat_projected = self.input_convs[i](feat)
            X, Y, Z = feat.shape[-3:]
            
            # no padding
            padding_mask_resized = feat.new_zeros((batch_size, ) + feat.shape[-3:], dtype=torch.bool)
            pos_embed = self.postional_encoding(padding_mask_resized)
            level_embed = self.level_encoding.weight[i]
            level_pos_embed = level_embed.view(1, -1, 1, 1, 1) + pos_embed
            
            # (h_i * w_i * d_i, 2)
            reference_points = self.point_generator.single_level_grid_priors(
                feat.shape[-3:], level_idx, device=feat.device)
            
            # normalize points to [0, 1]
            factor = feat.new_tensor([[Z, Y, X]]) * self.strides[level_idx]
            reference_points = reference_points / factor
            
            # shape (batch_size, c, x_i, y_i, z_i) -> (x_i * y_i * z_i, batch_size, c)
            feat_projected = feat_projected.flatten(2).permute(2, 0, 1)
            level_pos_embed = level_pos_embed.flatten(2).permute(2, 0, 1)
            padding_mask_resized = padding_mask_resized.flatten(1)

            encoder_input_list.append(feat_projected)
            padding_mask_list.append(padding_mask_resized)
            level_positional_encoding_list.append(level_pos_embed)
            spatial_shapes.append(feat.shape[-3:])
            reference_points_list.append(reference_points)
        
        # shape (batch_size, total_num_query),
        # total_num_query=sum([., x_i * y_i * z_i, .])
        padding_masks = torch.cat(padding_mask_list, dim=1)
        # shape (total_num_query, batch_size, c)
        encoder_inputs = torch.cat(encoder_input_list, dim=0)
        level_positional_encodings = torch.cat(level_positional_encoding_list, dim=0)
        
        device = encoder_inputs.device
        # shape (num_encoder_levels, 2), from low
        # resolution to high resolution
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=device)
        # shape (0, h_0*w_0, h_0*w_0+h_1*w_1, ...)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        reference_points = torch.cat(reference_points_list, dim=0)
        reference_points = reference_points[None, :, None].repeat(
            batch_size, 1, self.num_encoder_levels, 1)
        valid_radios = reference_points.new_ones(
            (batch_size, self.num_encoder_levels, 2))

        
        # shape (num_total_query, batch_size, c)
        memory = self.encoder(
            query=encoder_inputs,
            key=None,
            value=None,
            query_pos=level_positional_encodings,
            key_pos=None,
            attn_masks=None,
            key_padding_mask=None,
            query_key_padding_mask=padding_masks,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_radios=valid_radios)
        
        # (num_total_query, batch_size, c) -> (batch_size, c, num_total_query)
        memory = memory.permute(1, 2, 0)

        # from low resolution to high resolution
        num_query_per_level = [e[0] * e[1] * e[2] for e in spatial_shapes]
        outs = torch.split(memory, num_query_per_level, dim=-1)
        outs = [
            x.reshape(batch_size, -1, spatial_shapes[i][0],
                spatial_shapes[i][1], 
                spatial_shapes[i][2]) for i, x in enumerate(outs)]
        
        # build FPN path
        for i in range(self.num_input_levels - self.num_encoder_levels - 1, -1,
                       -1):
            x = feats[i]
            cur_feat = self.lateral_convs[i](x)
            
            y = cur_feat + F.interpolate(
                outs[-1],
                size=cur_feat.shape[-3:],
                mode='trilinear',
                align_corners=False,
            )
            
            y = self.output_convs[i](y)
            outs.append(y)
        
        outs[-1] = self.mask_feature(outs[-1])
        if self.reverse:
            ms_out_features = []
            for index in range(len(outs)):
                if index != 0:
                    outs[index] = self.reduce_convs[index-1](outs[index])
                    ms_out_features.append(outs[index])
                outs[index] = F.interpolate(outs[index], size=self.size,
                                mode='trilinear', align_corners=True).permute(0, 1, 4, 3, 2)
            x = torch.cat(outs, dim=1)
            if self.with_cp:
                x = checkpoint(self.fuse_conv, x)
            else:
                x = self.fuse_conv(x)
            return x, (ms_out_features[2].permute(0, 1, 4, 3, 2), 
                       ms_out_features[1].permute(0, 1, 4, 3, 2), outs[-3])
                
        return outs[::-1]