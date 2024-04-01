# align_after_view_transfromation=False


_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']
# Global
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

# nus occ class frequencies
nusc_class_frequencies = [944004, 1897170, 152386, 2391677, 16957802, 724139, 189027,
                          2074468, 413451, 2384460, 5916653, 175883646, 4275424, 51393615,
                          61411620, 105975596, 116424404, 1892500630]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
    'input_size': (256, 704),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

# Model
grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 0.4],
    'depth': [1.0, 45.0, 0.5],
}
occformer_grid_config = {
    'x': [-40, 40, 1.6],
    'y': [-40, 40, 1.6],
    'z': [-1, 5.4, 0.4],
}
occ_h = int((grid_config['y'][1] - grid_config['y'][0]) / grid_config['y'][2])
occ_w = int((grid_config['x'][1] - grid_config['x'][0]) / grid_config['x'][2])
occ_z = int((grid_config['z'][1] - grid_config['z'][0]) / grid_config['z'][2])

point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]
voxel_size = [0.1, 0.1, 0.2]

numC_Trans = 64
_dim_ = 256
_pos_dim_ = [96, 96, 64]
_ffn_dim_ = _dim_*2

# occformer settings
voxel_channels = [128, 256, 512, 1024]
voxel_num_layer = [2, 2, 2, 2]
voxel_strides = [1, 2, 2, 2]
voxel_out_indices = (0, 1, 2, 3)
voxel_out_channels = 192
ms_out_channels = [numC_Trans*4, numC_Trans*2, numC_Trans]
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

# group split
group_split = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2], # front, back, empty
               [0, 0, 1, 2, 0, 3, 4, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 6], # other, front(fine_less), empty
               [0, 1, 0, 0, 2, 0, 0, 3, 4, 0, 5, 0, 0, 0, 0, 0, 0, 6], # other, front(fine_more), empty
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 4], # other, back(fine_less), empty
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 3, 4]] # other, back(fine_more), empty
group_detr = len(group_split) + 1
group_classes = [17] + [group[-1] for group in group_split]

multi_adj_frame_id_cfg = (1, 1+1, 1)

model = dict(
    type='COTR_Group',
    group_split=group_split,
    align_after_view_transfromation=False,
    num_adj=len(range(*multi_adj_frame_id_cfg)),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=_dim_,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='LSSViewTransformerBEVStereo',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=_dim_,
        out_channels=numC_Trans,
        sid=False,
        collapse_z=False,
        loss_depth_weight=0.05,
        depthnet_cfg=dict(use_dcn=False,
                          aspp_mid_channels=96,
                          stereo=True,
                          bias=5.),
        downsample=16),
    img_bev_encoder_backbone=dict(
        type='OccupancyEncoder',
        num_stage=len(voxel_num_layer),
        in_channels=numC_Trans * (len(range(*multi_adj_frame_id_cfg))+1),
        block_numbers=voxel_num_layer,
        block_inplanes=voxel_channels,
        block_strides=voxel_strides,
        out_indices=voxel_out_indices,
        with_cp=True,
        norm_cfg=norm_cfg),
    img_bev_encoder_neck=dict(
        type='MSDeformAttnPixelDecoder3D',
        strides=[2, 4, 8, 16],
        in_channels=voxel_channels,
        feat_channels=voxel_out_channels,
        out_channels=voxel_out_channels,
        norm_cfg=norm_cfg,
        ms_out_channels=ms_out_channels,
        reverse=True,
        size=(50, 50, 16),
        encoder=dict(
            type='DetrTransformerEncoder',
            num_layers=6,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=dict(
                    type='MultiScaleDeformableAttention3D',
                    embed_dims=voxel_out_channels,
                    num_heads=8,
                    num_levels=3,
                    num_points=4,
                    im2col_step=64,
                    dropout=0.0,
                    batch_first=False,
                    norm_cfg=None,
                    init_cfg=None),
                ffn_cfgs=dict(
                    embed_dims=voxel_out_channels),
                feedforward_channels=voxel_out_channels * 4,
                ffn_dropout=0.0,
                operation_order=('self_attn', 'norm', 'ffn', 'norm')),
            init_cfg=None),
        positional_encoding=dict(
            type='SinePositionalEncoding3D',
            num_feats=voxel_out_channels // 3,
            normalize=True),),
    pts_bbox_head=dict(
        type='COTRHead',
        in_channels=numC_Trans,
        embed_dims=_dim_,
        num_query=100,
        group_detr=group_detr,
        group_classes=group_classes,
        num_classes=17,
        transformer=dict(
            type='TransformerMSOcc',
            embed_dims=_dim_,
            num_feature_levels=1,
            encoder=dict(
                type='OccEncoder',
                num_layers=1,
                grid_config=occformer_grid_config,
                data_config=data_config,
                pc_range=point_cloud_range,
                return_intermediate=False,
                fix_bug=True,
                transformerlayers=dict(
                    type='OccFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiScaleDeformableAttention3D',
                            embed_dims=_dim_,
                            num_levels=1,
                            num_points=4),
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=1),
                            embed_dims=_dim_,)
                    ],
                    ffn_embed_dims=_dim_,
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                    'ffn', 'norm')))),
        positional_encoding=dict(
            type='CustomLearnedPositionalEncoding3D',
            num_feats=_pos_dim_,
            row_num_embed=int(occ_h),
            col_num_embed=int(occ_w),
            tub_num_embed=int(occ_z)),
        transformer_decoder=dict(
            type='MaskOccDecoder',
            return_intermediate=True,
            num_layers=1,
            transformerlayers=dict(
                type='MaskOccDecoderLayer',
                attn_cfgs=[
                    dict(
                        type='MultiScaleDeformableAttention3D',
                        embed_dims=_dim_,
                        num_levels=1,
                        num_points=4,),
                    dict(
                        type='GroupMultiheadAttention',
                        group=group_detr,
                        embed_dims=_dim_,
                        num_heads=8,
                        dropout=0.1),
                ],
                feedforward_channels=2*_dim_,
                ffn_dropout=0.1,
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                    'ffn', 'norm'))),
        predictor=dict(
            type='MaskPredictorHead_Group',
            nbr_classes=17,
            group_detr=group_detr,
            group_classes=group_classes,
            in_dims=_dim_,
            hidden_dims=2*_dim_,
            out_dims=_dim_,
            mask_dims=_dim_),
        use_camera_mask=True,
        use_lidar_mask=False,
        cls_freq=nusc_class_frequencies,
        loss_occ=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=10.0),
        loss_cls= dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            reduction='mean',
            class_weight=[1.0] * 17 + [0.1]),
        loss_mask= dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=20.0),
        loss_dice= dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=1.0)),
    pre_process=dict(
        type='CustomResNet3D',
        numC_input=numC_Trans,
        with_cp=False,
        num_layer=[1,],
        num_channels=[numC_Trans,],
        stride=[1,],
        backbone_output_ids=[0,]),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            out_size_factor=4,
            assigner=dict(
                type='MaskHungarianAssigner3D',
                cls_cost=dict(type='MaskClassificationCost', weight=1.0),
                mask_cost=dict(type='MaskFocalLossCost', weight=20.0, binary_input=True),
                dice_cost=dict(type='MaskDiceLossCost', weight=1.0, pred_act=True, eps=1.0),
                use_camera_mask=True,
                use_lidar_mask=False),
            sampler=dict(
                type='MaskPseudoSampler',
                use_camera_mask=True,
                use_lidar_mask=False)
        )),
    test_cfg=dict(
        pts=dict(
            mask_threshold = 0.7,
            overlap_threshold = 0.8,
            occupy_threshold = 0.3,
            inf_merge=True,
            only_encoder=False
        )),
)

# Data
dataset_type = 'NuScenesDatasetOccpancy'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=True),
    dict(type='LoadOccGTFromFile'),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=True),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_depth', 'voxel_semantics',
                                'mask_lidar','mask_camera'])
]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config, sequential=True),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs'])
        ])
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    classes=class_names,
    modality=input_modality,
    stereo=True,
    filter_empty_gt=False,
    img_info_prototype='bevdet4d',
    multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
)

test_data_config = dict(
    pipeline=test_pipeline,
    ann_file=data_root + 'bevdetv2-nuscenes_infos_val.pkl')

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'bevdetv2-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=test_data_config,
    test=test_data_config,
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

for key in ['val', 'train', 'test']:
    data[key].update(share_data_config)

# Optimizer
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=1e-2) # lr 2e-4 for total batch size of 8 = 1 * 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[100,])
evaluation = dict(interval=4, pipeline=test_pipeline)
runner = dict(type='EpochBasedRunner', max_epochs=24)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
    dict(
        type='SyncbnControlHook',
        syncbn_start_epoch=0,
    ),
]

load_from="ckpts/bevdet-r50-4d-stereo-cbgs.pth"
# fp16 = dict(loss_scale='dynamic')