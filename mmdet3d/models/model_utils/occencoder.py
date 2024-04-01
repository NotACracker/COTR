# ---------------------------------------------
#  Modified by Qihang Ma
# ---------------------------------------------

from .custom_base_transformer_layer import MyCustomBaseTransformerLayer
import copy
import warnings
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.runner import force_fp32, auto_fp16
import numpy as np
import torch
import cv2 as cv
import mmcv
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.utils import ext_loader
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class OccEncoder(TransformerLayerSequence):

    """
    Attention with both self and cross
    Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default:
            `LN`.
    """

    def __init__(self, *args, pc_range=None, grid_config=None, data_config=None,
                 fix_bug=False, return_intermediate=False, dataset_type='nuscenes',
                 **kwargs):

        super(OccEncoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        
        if grid_config is not None:
            self.x_bound = grid_config['x']
            self.y_bound = grid_config['y']
            self.z_bound = grid_config['z']
        if data_config is not None:
            self.final_dim = data_config['input_size']
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.fix_bug = fix_bug

    @staticmethod
    def get_reference_points(H, W, Z, bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W, Z: spatial shape of volume.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        
        zs = torch.linspace(0.5, Z - 0.5, Z, dtype=dtype,
                            device=device).view(Z, 1, 1).expand(Z, H, W) / Z
        xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                            device=device).view(1, 1, W).expand(Z, H, W) / W
        ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                            device=device).view(1, H, 1).expand(Z, H, W) / H
        ref_3d = torch.stack((xs, ys, zs), -1)
        ref_3d = ref_3d.permute(3, 0, 1, 2).flatten(1).permute(1, 0)
        ref_3d = ref_3d[None, None].repeat(bs, 1, 1, 1)
        return ref_3d

    # This function must use fp32!!!
    @force_fp32(apply_to=('reference_points', 'img_metas'))
    def point_sampling(self, reference_points, pc_range,  img_metas, cam_params=None):
        sensor2ego = img_metas[0]['sensor2ego']
        intrin = img_metas[0]['intrin']
        ego2sensor = sensor2ego.clone()
        ego2img = sensor2ego.clone()
        bs, num_cam = sensor2ego.shape[:2]
        for b in range(bs):
            for cam in range(num_cam):
                rot_inv = sensor2ego[b, cam, :3, :3].T
                trans = (-sensor2ego[b, cam, :3, 3].T).T
                ego2sensor[b, cam, :3, :3] = rot_inv
                ego2sensor[b, cam, :3, 3] = torch.matmul(rot_inv, trans.unsqueeze(-1)).squeeze(1)
                intrinsic = intrin[b][cam]
                viewpad = np.eye(4)
                viewpad = reference_points.new_tensor(viewpad)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                ego2img[b][cam] = (viewpad @ ego2sensor[b][cam].T)

        ego2img = reference_points.new_tensor(ego2img) # (B, N, 4, 4)

        reference_points = reference_points.clone()

        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (pc_range[5] - pc_range[2]) + pc_range[2]

        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1)

        reference_points = reference_points.permute(1, 0, 2, 3) #shape: (num_points_in_pillar,bs,h*w,4)
        D, B, num_query = reference_points.size()[:3] # D=num_points_in_pillar , num_query=h*w
        num_cam = ego2img.size(1)

        reference_points = reference_points.view(
            D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)  #shape: (num_points_in_pillar,bs,num_cam,h*w,4)

        ego2img = ego2img.view(
            1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)
        reference_points_cam = torch.matmul(ego2img.to(torch.float32),reference_points.to(torch.float32)).squeeze(-1)
        eps = 1e-5

        occ_mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)

        reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
        reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]

        occ_mask = (occ_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            occ_mask = torch.nan_to_num(occ_mask)
        else:
            occ_mask = occ_mask.new_tensor(
                np.nan_to_num(occ_mask.cpu().numpy()))
        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4) #shape: (num_cam,bs,h*w,num_points_in_pillar,2)

        # (1, B, N, z*h*w, 1) --> (N, B, z*h*w, 1, 1) --> (N, B, z*h*w, 1)
        occ_mask = occ_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        return reference_points_cam, occ_mask

    def get_reference_points_fixbug(self,H, W, Z=8, bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        X = torch.arange(*self.x_bound, dtype=torch.float) + self.x_bound[-1]/2
        Y = torch.arange(*self.y_bound, dtype=torch.float) + self.y_bound[-1]/2
        Z = torch.arange(*self.z_bound, dtype=torch.float) + self.z_bound[-1]/2
        Y, X, Z = torch.meshgrid([Y, X, Z])
        coords = torch.stack([X, Y, Z], dim=-1)
        coords = coords.to(dtype).to(device)
        # frustum = torch.cat([coords, torch.ones_like(coords[...,0:1])], dim=-1) #(x, y, z, 4)
        return coords


    @force_fp32(apply_to=('reference_points', 'cam_params', 'img_metas'))
    def point_sampling_fixbug(self, reference_points, pc_range, img_metas, cam_params=None):
        sensor2ego, ego2global, cam2imgs, post_rots, post_trans, bda = cam_params
        B, N, _, _ = sensor2ego.shape
        eps = 1e-5
        reference_points = reference_points[None, None].repeat(B, N, 1, 1, 1, 1)
        reference_points = torch.inverse(bda).view(B, 1, 1, 1, 1, 3,
                          3).matmul(reference_points.unsqueeze(-1)).squeeze(-1)
        reference_points -= sensor2ego[:,:,:3,3].view(B, N, 1, 1, 1, 3)
        combine = sensor2ego[:,:,:3,:3].matmul(torch.inverse(cam2imgs)).inverse()
        reference_points_cam = combine.view(B, N, 1, 1, 1, 3, 3).matmul(reference_points.unsqueeze(-1)).squeeze(-1)
        reference_points_cam = torch.cat([reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3])*eps),  reference_points_cam[..., 2:3]], 5
            )
        reference_points_cam = post_rots.view(B, N, 1, 1, 1, 3, 3).matmul(reference_points_cam.unsqueeze(-1)).squeeze(-1)
        reference_points_cam += post_trans.view(B, N, 1, 1, 1, 3) 
        reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
        reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]
        mask = (reference_points_cam[..., 2:3] > eps)
        mask = (mask & (reference_points_cam[..., 0:1] > 0.0) 
                 & (reference_points_cam[..., 0:1] < 1.0) 
                 & (reference_points_cam[..., 1:2] > 0.0) 
                 & (reference_points_cam[..., 1:2] < 1.0))

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            mask = torch.nan_to_num(mask)
        else:
            mask = mask.new_tensor(
                np.nan_to_num(mask.cpu().numpy()))

        B, N, W, H, D, _ = reference_points_cam.shape
        reference_points_cam = reference_points_cam.permute(1, 0, 4, 3, 2, 5).reshape(N, B, D*H*W, 3)
        reference_points_cam = reference_points_cam[:, :, :, None, :]   # shape: (num_cam,bs,z*h*w,num_level,2)
        # (B, N, W, H, D, 1) --> (N, B, D*H*W, 1)
        mask = mask.permute(1, 0, 4, 3, 2, 5).reshape(N, B, D*H*W, 1)

        return reference_points_cam[..., :2], mask, reference_points_cam[..., 2:3]


    @auto_fp16()
    def forward(self,
                occ_query,
                key,
                value,
                *args,
                occ_h=None,
                occ_w=None,
                occ_z=None,
                occ_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                cam_params=None,
                valid_ratios=None,
                prev_occ=None,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            bev_query (Tensor): Input BEV query with shape
                `(num_query, bs, embed_dims)`.
            key & value (Tensor): Input multi-cameta features with shape
                (num_cam, num_value, bs, embed_dims)
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """

        output = occ_query
        intermediate = []

        if self.fix_bug:
            ref_3d = self.get_reference_points_fixbug(
                occ_h, occ_w, occ_z, bs=occ_query.size(1), 
                device=occ_query.device, dtype=occ_query.dtype)
            reference_points_cam, occ_mask, _ = self.point_sampling_fixbug(
                ref_3d, self.pc_range, kwargs['img_metas'], cam_params=cam_params)
        else:
            ref_3d = self.get_reference_points(
                occ_h, occ_w, occ_z, bs=occ_query.size(1), 
                device=occ_query.device, dtype=occ_query.dtype
            )
            reference_points_cam, occ_mask = self.point_sampling(
                ref_3d, self.pc_range, kwargs['img_metas'])
        
        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
        occ_query = occ_query.permute(1, 0, 2)
        occ_pos = occ_pos.permute(1, 0, 2)
        if self.fix_bug:
            ref_3d = ref_3d.permute(3, 2, 1, 0).flatten(1).permute(1, 0) 
            ref_3d = ref_3d[None, None].repeat(occ_query.shape[0], 1, 1, 1) # [w, h, z, 3] -> [bs, num_level, z*h*w, 3]

        for lid, layer in enumerate(self.layers):
            output = layer(
                occ_query,
                key,
                value,
                *args,
                occ_pos=occ_pos,
                ref_3d=ref_3d, # [bs, num_level, z*h*w, 3]
                occ_h=occ_h,
                occ_w=occ_w,
                occ_z=occ_z,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam, # [num_cam, bs, num_level, 2]
                occ_mask=occ_mask,                         # [num_cam, bs, num_level, 1]
                prev_occ=prev_occ,
                **kwargs)

            occ_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


@TRANSFORMER_LAYER.register_module()
class OccFormerLayer(MyCustomBaseTransformerLayer):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default: None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default: 2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(OccFormerLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])

    def forward(self,
                query,
                key=None,
                value=None,
                occ_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_3d=None,
                occ_h=None,
                occ_w=None,
                occ_z=None,
                reference_points_cam=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_occ=None,
                occ_mask=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            # deformable self attention
            if layer == 'self_attn':

                query = self.attentions[attn_index](
                    query,
                    query,
                    query,
                    identity=query,
                    query_pos=occ_pos,
                    key_pos=occ_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=ref_3d.permute(0, 2, 1, 3),
                    spatial_shapes=torch.tensor(
                        [[occ_h, occ_w, occ_z]], device=query.device),
                    level_start_index=torch.tensor([0], device=query.device),
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            # spaital cross attention
            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    bev_mask=occ_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query
