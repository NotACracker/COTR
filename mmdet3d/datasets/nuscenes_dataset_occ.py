# Copyright (c) OpenMMLab. All rights reserved.
import os
import mmcv
import torch
import cv2
import numpy as np
from tqdm import tqdm

from .builder import DATASETS
from .nuscenes_dataset import NuScenesDataset
from .occ_metrics import Metric_mIoU, Metric_FScore, Metric_SSC

colors_map = np.array(
    [
        [  0,   0,   0, 255],       # other                Black
        [112, 128, 144, 255],       # barrier              Slategrey
        [220,  20,  60, 255],       # bicycle              Crimson
        [255, 127,  80, 255],       # bus                  Coral
        [255, 158,   0, 255],       # car                  Orange
        [233, 150,  70, 255],       # construction_vehicle Darksalmon
        [255,  61,  99, 255],       # motorcycle           Red
        [  0,   0, 230, 255],       # pedestrian           Blue
        [ 47,  79,  79, 255],       # traffic_cone         Darkslategrey
        [255, 140,   0, 255],       # trailer              Darkorange
        [255,  99,  71, 255],       # truck                Tomato                
        [  0, 207, 191, 255],       # driveable_surface    nuTonomy green
        [175,   0,  75, 255],       # other_flat           dark red
        [ 75,   0,  75, 255],       # sidewalk             
        [112, 180,  60, 255],       # terrain                        
        [222, 184, 135, 255],       # manmade              Burlywood
        [  0, 175,   0, 255],       # vegetation
    ])



@DATASETS.register_module()
class NuScenesDatasetOccpancy(NuScenesDataset):
    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        input_dict = super(NuScenesDatasetOccpancy, self).get_data_info(index)
        # standard protocol modified from SECOND.Pytorch
        input_dict['occ_gt_path'] = self.data_infos[index]['occ_path']
        return input_dict

    def evaluate(self, occ_results, runner=None, show_dir=None, logger=None,
                 save_path=None, **eval_kwargs):
        logger = runner.logger if runner is not None else logger
        self.occ_eval_metrics = Metric_SSC(
            num_classes=18,
            use_lidar_mask=False,
            use_image_mask=True,
            logger=logger)

        print('\nStarting Evaluation...')
        if logger is not None:
            logger.info('Starting Evaluation...')
        for index, occ_pred in enumerate(tqdm(occ_results)):
            info = self.data_infos[index]

            occ_gt = np.load(os.path.join(info['occ_path'],'labels.npz'))
            gt_semantics = occ_gt['semantics']
            mask_lidar = occ_gt['mask_lidar'].astype(bool)
            mask_camera = occ_gt['mask_camera'].astype(bool)
            
            if isinstance(occ_pred, dict):
                occ_pred = occ_pred['occ']

            self.occ_eval_metrics.add_batch(occ_pred, gt_semantics, mask_lidar, mask_camera)

            if index%100==0 and show_dir is not None:
                gt_vis = self.vis_occ(gt_semantics)
                pred_vis = self.vis_occ(occ_pred)
                mmcv.imwrite(np.concatenate([gt_vis, pred_vis], axis=1),
                             os.path.join(show_dir + "%d.jpg"%index))
        return self.occ_eval_metrics.count_miou()

    def vis_occ(self, semantics):
        # simple visualization of result in BEV
        semantics_valid = np.logical_not(semantics == 17)
        d = np.arange(16).reshape(1, 1, 16)
        d = np.repeat(d, 200, axis=0)
        d = np.repeat(d, 200, axis=1).astype(np.float32)
        d = d * semantics_valid
        selected = np.argmax(d, axis=2)

        selected_torch = torch.from_numpy(selected)
        semantics_torch = torch.from_numpy(semantics)

        occ_bev_torch = torch.gather(semantics_torch, dim=2,
                                     index=selected_torch.unsqueeze(-1))
        occ_bev = occ_bev_torch.numpy()

        occ_bev = occ_bev.flatten().astype(np.int32)
        occ_bev_vis = colors_map[occ_bev].astype(np.uint8)
        occ_bev_vis = occ_bev_vis.reshape(200, 200, 4)[::-1, ::-1, :3]
        occ_bev_vis = cv2.resize(occ_bev_vis,(400,400))
        return occ_bev_vis