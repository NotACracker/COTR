# Copyright (c) Phigent Robotics. All rights reserved.
from .bevdet import BEVStereo4D

import torch
from mmdet.models import DETECTORS
from mmdet.models.builder import build_loss
from mmcv.cnn.bricks.conv_module import ConvModule
from mmcv.runner import force_fp32, auto_fp16
from torch import nn
import numpy as np

from mmdet.models.backbones.resnet import ResNet
from mmdet3d.models.backbones import VoVNet
from mmdet3d.models.necks import MSDeformAttnPixelDecoder3D


@DETECTORS.register_module()
class COTR(BEVStereo4D):

    def __init__(self,
                 **kwargs):
        super(COTR, self).__init__(**kwargs)
        self.align_after_view_transfromation = False
    
    @force_fp32()
    def bev_encoder(self, x):
        if isinstance(self.img_bev_encoder_neck,
                      MSDeformAttnPixelDecoder3D):
            x = x.permute(0, 1, 4, 3, 2)
        x = self.img_bev_encoder_backbone(x)
        x, feats = self.img_bev_encoder_neck(x)
        # if type(x) in [list, tuple]:
        #     x = x[0]
        return x, feats
    
    def image_encoder(self, img, stereo=False):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        x = self.img_backbone(imgs)
        if isinstance(self.img_backbone,VoVNet):
            tx = []
            for stage in self.img_backbone._out_features:
                tx.append(x[stage])
            x = tuple(tx)
        stereo_feat = None
        if stereo:
            stereo_feat = x[0]
            x = x[1:]
        if self.with_img_neck:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x, stereo_feat

    def extract_stereo_ref_feat(self, x):
        B, N, C, imH, imW = x.shape
        x = x.view(B * N, C, imH, imW)
        if isinstance(self.img_backbone,ResNet):
            if self.img_backbone.deep_stem:
                x = self.img_backbone.stem(x)
            else:
                x = self.img_backbone.conv1(x)
                x = self.img_backbone.norm1(x)
                x = self.img_backbone.relu(x)
            x = self.img_backbone.maxpool(x)
            for i, layer_name in enumerate(self.img_backbone.res_layers):
                res_layer = getattr(self.img_backbone, layer_name)
                x = res_layer(x)
                return x
        elif isinstance(self.img_backbone,VoVNet):
            x = self.img_backbone.stem(x)
            for i, stage_name in enumerate(self.img_backbone.stage_names):
                stage = getattr(self.img_backbone, stage_name)
                x = stage(x)
                return x
        else:
            x = self.img_backbone.patch_embed(x)
            hw_shape = (self.img_backbone.patch_embed.DH,
                        self.img_backbone.patch_embed.DW)
            if self.img_backbone.use_abs_pos_embed:
                x = x + self.img_backbone.absolute_pos_embed
            x = self.img_backbone.drop_after_pos(x)

            for i, stage in enumerate(self.img_backbone.stages):
                x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
                out = out.view(-1,  *out_hw_shape,
                               self.img_backbone.num_features[i])
                out = out.permute(0, 3, 1, 2).contiguous()
                return out

    def prepare_bev_feat(self, img, sensor2keyego, ego2global, intrin,
                         post_rot, post_tran, bda, mlp_input, feat_prev_iv,
                         k2s_sensor, extra_ref_frame):
        if extra_ref_frame:
            stereo_feat = self.extract_stereo_ref_feat(img)
            return None, None, stereo_feat, None
        x, stereo_feat = self.image_encoder(img, stereo=True)
        metas = dict(k2s_sensor=k2s_sensor,
                     intrins=intrin,
                     post_rots=post_rot,
                     post_trans=post_tran,
                     frustum=self.img_view_transformer.cv_frustum.to(x),
                     cv_downsample=4,
                     downsample=self.img_view_transformer.downsample,
                     grid_config=self.img_view_transformer.grid_config,
                     cv_feat_list=[feat_prev_iv, stereo_feat])
        bev_feat, depth = self.img_view_transformer(
            [x, sensor2keyego, ego2global, intrin, post_rot, post_tran, bda,
             mlp_input], metas)
        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]
        return bev_feat, depth, stereo_feat, x

    def extract_img_feat(self,
                         img,
                         img_metas,
                         pred_prev=False,
                         sequential=False,
                         **kwargs):
        if sequential:
            # Todo
            assert False
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
        bda, curr2adjsensor = self.prepare_inputs(img, stereo=True)
        # we only need key frame params for back project
        cam_params = [sensor2keyegos[0], ego2globals[0], intrins[0], 
                      post_rots[0], post_trans[0], bda]
        """Extract features of images."""
        bev_feat_list = []
        img_feature_key_frame = None
        depth_key_frame = None
        feat_prev_iv = None
        for fid in range(self.num_frame-1, -1, -1):
            img, sensor2keyego, ego2global, intrin, post_rot, post_tran = \
                imgs[fid], sensor2keyegos[fid], ego2globals[fid], intrins[fid], \
                post_rots[fid], post_trans[fid]
            key_frame = fid == 0
            extra_ref_frame = fid == self.num_frame-self.extra_ref_frames
            if key_frame or self.with_prev:
                if self.align_after_view_transfromation:
                    sensor2keyego, ego2global = sensor2keyegos[0], ego2globals[0]
                mlp_input = self.img_view_transformer.get_mlp_input(
                    sensor2keyegos[0], ego2globals[0], intrin,
                    post_rot, post_tran, bda)
                inputs_curr = (img, sensor2keyego, ego2global, intrin,
                               post_rot, post_tran, bda, mlp_input,
                               feat_prev_iv, curr2adjsensor[fid],
                               extra_ref_frame)
                if key_frame:
                    bev_feat, depth, feat_curr_iv, img_f = \
                        self.prepare_bev_feat(*inputs_curr)
                    depth_key_frame = depth
                    img_feature_key_frame = img_f
                    img_metas[0]['sensor2ego'] = sensor2keyego
                    img_metas[0]['intrin'] = intrin
                    img_metas[0]['img_shape'] = [img.shape[-2:]]
                else:
                    with torch.no_grad():
                        bev_feat, depth, feat_curr_iv, img_f = \
                            self.prepare_bev_feat(*inputs_curr)
                if not extra_ref_frame:
                    bev_feat_list.append(bev_feat)
                feat_prev_iv = feat_curr_iv
        if pred_prev:
            # Todo
            assert False
        if not self.with_prev:
            bev_feat_key = bev_feat_list[0]
            if len(bev_feat_key.shape) ==4:
                b,c,h,w = bev_feat_key.shape
                bev_feat_list = \
                    [torch.zeros([b,
                                  c * (self.num_frame -
                                       self.extra_ref_frames - 1),
                                  h, w]).to(bev_feat_key), bev_feat_key]
            else:
                b, c, z, h, w = bev_feat_key.shape
                bev_feat_list = \
                    [torch.zeros([b,
                                  c * (self.num_frame -
                                       self.extra_ref_frames - 1), z,
                                  h, w]).to(bev_feat_key), bev_feat_key]
        if self.align_after_view_transfromation:
            for adj_id in range(self.num_frame-2):
                bev_feat_list[adj_id] = \
                    self.shift_feature(bev_feat_list[adj_id],
                                       [sensor2keyegos[0],
                                        sensor2keyegos[self.num_frame-2-adj_id]],
                                       bda)
        bev_feat = torch.cat(bev_feat_list, dim=1)
        x, feats = self.bev_encoder(bev_feat)
        return [x, feats, img_feature_key_frame], depth_key_frame, img_metas, cam_params

    def extract_feat(self, points, img, img_metas, **kwargs):
        """Extract features from images and points."""
        img_feats, depth, img_metas, cam_params = self.extract_img_feat(img, img_metas, **kwargs)
        pts_feats = None
        return (img_feats, pts_feats, depth, img_metas, cam_params)

    @force_fp32(apply_to=('pts_feats'))
    def forward_pts_train(self,
                          pts_feats,
                          depth,
                          voxel_semantics,
                          gt_classes,
                          sem_mask,
                          mask_camera,
                          mask_lidar,
                          img_metas,
                          cam_params,
                          **kwargs,):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats, depth, img_metas, 
                                  cam_params, **kwargs)
        loss_inputs = [voxel_semantics, gt_classes, sem_mask,
                       mask_camera, mask_lidar, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        
        return losses

    def generate_mask(self, semantics):
        """Convert semantics to semantic mask for each instance
        Args:
            semantics: [W, H, Z]
        Return:
            classes: [N]
                N unique class in semantics
            masks: [N, W, H, Z]
                N instance masks
        """
        
        w, h, z = semantics.shape
        classes = torch.unique(semantics)
        # # remove ignore region
        # if self.ignore_label is not None:
        #     classes = classes[classes != self.ignore_label]
        gt_classes = classes.long()

        masks = []
        for class_id in classes:
            masks.append(semantics == class_id)
        
        if len(masks) == 0:
            masks = torch.zeros(0, w, h, z)
        else:
            masks = torch.stack([x.clone() for x in masks])

        return gt_classes, masks.long()

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, pts_feats, depth, img_metas, cam_params = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        voxel_semantics = kwargs['voxel_semantics']
        mask_camera = kwargs['mask_camera']
        mask_lidar = kwargs['mask_lidar']
        gt_classes = []
        sem_mask = []
        for voxel_semantic in voxel_semantics:
            gt_class, sem_masks = self.generate_mask(voxel_semantic)
            gt_classes.append(gt_class.to(voxel_semantic.device))
            sem_mask.append(sem_masks.to(voxel_semantic.device))

        losses = dict()
        loss_occ = self.forward_pts_train(img_feats, depth, voxel_semantics=voxel_semantics,
                                          gt_classes=gt_classes, sem_mask=sem_mask,
                                          mask_camera=mask_camera, mask_lidar=mask_lidar,
                                          img_metas=img_metas, cam_params=cam_params)
        losses.update(loss_occ)
        gt_depth = kwargs['gt_depth']
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses['loss_depth'] = loss_depth
        
        return losses

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        img_feats, _, depth, img_metas, cam_params = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        
        outs = self.pts_bbox_head(img_feats, depth, img_metas=img_metas, 
                                  cam_params=cam_params)
        occ = self.pts_bbox_head.get_occ(outs, img_metas=img_metas)
        # occ = occ.squeeze(dim=0).cpu().numpy().astype(np.uint8)

        return [occ]


@DETECTORS.register_module()
class COTR_Group(COTR):

    def __init__(self,
                 group_split=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.group_split = torch.tensor(group_split, dtype=torch.uint8)
    
    def generate_group(self, voxel_semantics):
        group_classes = []
        group_masks = []
        for i in range(len(self.group_split)+1):
            gt_classes = []
            sem_masks = []
            for voxel_semantic in voxel_semantics:
                if not i < 1:
                    w, h, z = voxel_semantic.shape
                    group_split = self.group_split[i-1].to(voxel_semantic)
                    voxel_semantic = group_split[voxel_semantic.flatten().long()].reshape(w, h, z)
                gt_class, sem_mask = self.generate_mask(voxel_semantic)
                gt_classes.append(gt_class.to(voxel_semantic.device))
                sem_masks.append(sem_mask.to(voxel_semantic.device))
            
            group_classes.append(gt_classes)
            group_masks.append(sem_masks)

        return group_classes, group_masks

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, pts_feats, depth, img_metas, cam_params = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        voxel_semantics = kwargs['voxel_semantics']
        mask_camera = kwargs['mask_camera']
        mask_lidar = kwargs['mask_lidar']
        
        gt_classes, sem_mask = self.generate_group(voxel_semantics)

        losses = dict()
        loss_occ = self.forward_pts_train(img_feats, depth, voxel_semantics=voxel_semantics,
                                          gt_classes=gt_classes, sem_mask=sem_mask,
                                          mask_camera=mask_camera, mask_lidar=mask_lidar,
                                          img_metas=img_metas, cam_params=cam_params)
        losses.update(loss_occ)
        gt_depth = kwargs['gt_depth']
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses['loss_depth'] = loss_depth
        
        return losses
    
    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        img_feats, _, depth, img_metas, cam_params = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        
        outs = self.pts_bbox_head(img_feats, depth, img_metas=img_metas, 
                                  cam_params=cam_params)
        occ = self.pts_bbox_head.get_occ(outs, img_metas=img_metas)

        return [occ]
    
    def forward_dummy(self, points=None, img_metas=None, img_inputs=None, **kwargs):
        img_feats, _, depth, img_metas, cam_params = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        
        outs = self.pts_bbox_head(img_feats, depth, img_metas=img_metas, 
                                  cam_params=cam_params)
        return outs