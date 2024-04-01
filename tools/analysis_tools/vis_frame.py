
from mayavi import mlab
import mayavi
# mlab.options.offscreen = True
print("Set mlab.options.offscreen={}".format(mlab.options.offscreen))

import argparse, torch, os, json
import shutil
import numpy as np
import mmcv
from mmcv import Config, DictAction
from collections import OrderedDict
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from mmdet3d.datasets import build_dataset
from nuscenes.nuscenes import NuScenes
import torch.nn.functional as F
try:
    # If mmdet version > 2.23.0, compat_cfg would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg
# from pyvirtualdisplay import Display
# display = Display(visible=False, size=(2560, 1440))
# display.start()


def gridcloud3d(B, Z, Y, X, device='cpu'):
    # we want to sample for each location in the grid
    grid_z, grid_y, grid_x = meshgrid3d(B, Z, Y, X, device=device)
    x = torch.reshape(grid_x, [B, -1])
    y = torch.reshape(grid_y, [B, -1])
    z = torch.reshape(grid_z, [B, -1])

    # pdb.set_trace()
    # these are B x N
    xyz = torch.stack([x, y, z], dim=2)
    # here is stack in order with xyz
    # this is B x N x 3

    # pdb.set_trace()
    return xyz


def meshgrid3d(B, Z, Y, X, stack=False, device='cuda'):
    # returns a meshgrid sized B x Z x Y x X

    grid_z = torch.linspace(0.0, Z-1, Z, device=device)
    grid_z = torch.reshape(grid_z, [1, Z, 1, 1])
    grid_z = grid_z.repeat(B, 1, Y, X)

    grid_y = torch.linspace(0.0, Y-1, Y, device=device)
    grid_y = torch.reshape(grid_y, [1, 1, Y, 1])
    grid_y = grid_y.repeat(B, Z, 1, X)

    grid_x = torch.linspace(0.0, X-1, X, device=device)
    grid_x = torch.reshape(grid_x, [1, 1, 1, X])
    grid_x = grid_x.repeat(B, Z, Y, 1)
    # here repeat is in the order with ZYX

    if stack:
        # note we stack in xyz order
        # (see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample)
        grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        return grid
    else:
        return grid_z, grid_y, grid_x


def draw(
    voxels,          # semantic occupancy predictions
    vox_origin,      #
    voxel_size=0.4,  # voxel size in the real world
    pc_range=[-40, -40, -1, 40, 40, 5.4],        # point cloud range
    grid=None,       # voxel coordinates of point cloud
    pt_label=None,   # label of point cloud
    voxels_lidar=None,
    save_dirs=None,
    cam_positions=None,
    focal_positions=None,
    cam_names=None,
    timestamp=None,
    mode=0,          # mode:0 pred, 1 gt
):
    h, w, z = voxels.shape
    if grid is not None:
        grid = grid.astype(np.int)


    X_org, Y_org, Z_org = voxels.shape
    x_offset, y_offset, z_offset = (X_org - 200) // 2, (Y_org - 200) // 2, (Z_org - 16) // 2
    voxels = voxels[x_offset:x_offset+200, y_offset:y_offset+200, z_offset:z_offset+16]
    voxels_lidar = voxels_lidar[x_offset:x_offset+200, y_offset:y_offset+200, z_offset:z_offset+16]

    xyz = gridcloud3d(1, 16, 200, 200, device='cpu')
    xyz_min = np.array(pc_range[:3])
    xyz_max = np.array(pc_range[3:])
    occ_size = np.array([200, 200, 16])
    xyz = xyz / occ_size * (xyz_max - xyz_min) + xyz_min + 0.5 * voxel_size
    xyz = xyz.reshape(16, 200, 200, 3).permute(2, 1, 0, 3).numpy()
    xyz_lidar = np.concatenate([xyz, voxels_lidar[..., None]], axis=-1).reshape(-1, 4)
    xyz = np.concatenate([xyz, voxels[..., None]], axis=-1).reshape(-1, 4)

    # if mode == 0: # occupancy pred
    # grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T
    grid_coords = xyz
    grid_coords[grid_coords[:, 3] == 17, 3] = 20
    grid_coords_lidar = xyz_lidar
    grid_coords_lidar[grid_coords_lidar[:, 3] == 17, 3] = 20

    # Get the voxels inside FOV
    fov_grid_coords = grid_coords
    fov_grid_coords_lidar = grid_coords_lidar

    # Remove empty and unknown voxels
    fov_voxels = fov_grid_coords[
        (fov_grid_coords[:, 3] >= 0) & (fov_grid_coords[:, 3] < 20)
    ]
    fov_voxels_lidar = fov_grid_coords_lidar[
        (fov_grid_coords_lidar[:, 3] >= 0) & (fov_grid_coords_lidar[:, 3] < 20)
    ]
    
    figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
    # Draw occupied inside FOV voxels
    plt_plot_fov = mlab.points3d(
        fov_voxels[:, 0],
        fov_voxels[:, 1],
        fov_voxels[:, 2],
        fov_voxels[:, 3],
        colormap="viridis",
        scale_factor=0.95 * voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=0,
        vmax=19, # 19
    )

    colors = np.array(
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
            # [175,   0,  75, 255],       # other_flat           dark red
            [175,   0,  75, 255],       # other_flat           
            [ 75,   0,  75, 255],       # sidewalk             
            [112, 180,  60, 255],       # terrain                        
            [222, 184, 135, 255],       # manmade              Burlywood
            [  0, 175,   0, 255],       # vegetation           
            [  0, 255, 127, 255],       # ego car              dark cyan
            [255,  99,  71, 255],       # ego car
            [  0, 191, 255, 255]        # ego car
        ]
    ).astype(np.uint8)

    plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors
    scene = figure.scene

    for i, cam_name in enumerate(cam_names):
        # if cam_name != 'CAM_FRONT_LEFT':
        #     continue
        scene.camera.position = cam_positions[i]
        scene.camera.focal_point = focal_positions[i] 
        scene.camera.view_angle = 35 if i != 3 else 60
        scene.camera.view_up = [0.0, 0.0, 1.0]
        scene.camera.clipping_range = [0.01, 300.]
        scene.camera.compute_view_plane_normal()
        scene.render()
        if mode == 0:
            save_path = os.path.join(save_dirs, f'pred_{cam_name}.png')
        elif mode == 1:
            save_path = os.path.join(save_dirs, f'gt_{cam_name}.png')
        print(f"save_path:{save_path}")
        mlab.savefig(save_path)
    mlab.close(scene=None, all=False)
    
    figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
    plt_plot_fov_lidar = mlab.points3d(
        fov_voxels_lidar[:, 0],
        fov_voxels_lidar[:, 1],
        fov_voxels_lidar[:, 2],
        fov_voxels_lidar[:, 3],
        colormap="viridis",
        scale_factor=0.95 * voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=0,
        vmax=19, # 19
    )
    plt_plot_fov_lidar.glyph.scale_mode = "scale_by_vector"
    plt_plot_fov_lidar.module_manager.scalar_lut_manager.lut.table = colors
    scene = figure.scene

    scene.camera.position = [0.75131739, -35.08337438,  16.71378558]
    scene.camera.focal_point = [0.75131739, -34.21734897,  16.21378558]
    scene.camera.view_angle = 40.0
    scene.camera.view_up = [0.0, 0.0, 1.0]
    scene.camera.clipping_range = [0.01, 300.]
    scene.camera.compute_view_plane_normal()
    scene.render()
    
    if mode==0:
        save_path = os.path.join(save_dirs, 'pred_normal.png')
    elif mode==1:
        save_path = os.path.join(save_dirs, 'gt_normal.png')
    print(f"save_path:{save_path}")
    mlab.savefig(save_path)
    
    scene.camera.position = [ 0.75131739,  0.78265103, 93.21378558]
    scene.camera.focal_point = [ 0.75131739,  0.78265103, 92.21378558]
    scene.camera.view_angle = 40.0
    scene.camera.view_up = [0., 1., 0.]
    scene.camera.clipping_range = [0.01, 400.]
    scene.camera.compute_view_plane_normal()
    scene.render()
    if mode==0:
        save_path = os.path.join(save_dirs, 'pred_bev.png')
    elif mode==1:
        save_path = os.path.join(save_dirs, 'gt_bev.png')
    print(f"save_path:{save_path}")
    mlab.savefig(save_path)
    mlab.close(scene=None, all=False)
    #mlab.show()

def rt2mat(translation, quaternion=None, inverse=False, rotation=None):
    R = Quaternion(quaternion).rotation_matrix if rotation is None else rotation
    T = np.array(translation)
    if inverse:
        R = R.T
        T = -R @ T
    mat = np.eye(4)
    mat[:3, :3] = R
    mat[:3, 3] = T
    return mat

val_scene = ['scene-0003', 'scene-0012', 'scene-0013', 'scene-0014', 'scene-0015', 'scene-0016', 'scene-0017', 'scene-0018',
     'scene-0035', 'scene-0036', 'scene-0038', 'scene-0039', 'scene-0092', 'scene-0093', 'scene-0094', 'scene-0095',
     'scene-0096', 'scene-0097', 'scene-0098', 'scene-0099', 'scene-0100', 'scene-0101', 'scene-0102', 'scene-0103',
     'scene-0104', 'scene-0105', 'scene-0106', 'scene-0107', 'scene-0108', 'scene-0109', 'scene-0110', 'scene-0221',
     'scene-0268', 'scene-0269', 'scene-0270', 'scene-0271', 'scene-0272', 'scene-0273', 'scene-0274', 'scene-0275',
     'scene-0276', 'scene-0277', 'scene-0278', 'scene-0329', 'scene-0330', 'scene-0331', 'scene-0332', 'scene-0344',
     'scene-0345', 'scene-0346', 'scene-0519', 'scene-0520', 'scene-0521', 'scene-0522', 'scene-0523', 'scene-0524',
     'scene-0552', 'scene-0553', 'scene-0554', 'scene-0555', 'scene-0556', 'scene-0557', 'scene-0558', 'scene-0559',
     'scene-0560', 'scene-0561', 'scene-0562', 'scene-0563', 'scene-0564', 'scene-0565', 'scene-0625', 'scene-0626',
     'scene-0627', 'scene-0629', 'scene-0630', 'scene-0632', 'scene-0633', 'scene-0634', 'scene-0635', 'scene-0636',
     'scene-0637', 'scene-0638', 'scene-0770', 'scene-0771', 'scene-0775', 'scene-0777', 'scene-0778', 'scene-0780',
     'scene-0781', 'scene-0782', 'scene-0783', 'scene-0784', 'scene-0794', 'scene-0795', 'scene-0796', 'scene-0797',
     'scene-0798', 'scene-0799', 'scene-0800', 'scene-0802', 'scene-0904', 'scene-0905', 'scene-0906', 'scene-0907',
     'scene-0908', 'scene-0909', 'scene-0910', 'scene-0911', 'scene-0912', 'scene-0913', 'scene-0914', 'scene-0915',
     'scene-0916', 'scene-0917', 'scene-0919', 'scene-0920', 'scene-0921', 'scene-0922', 'scene-0923', 'scene-0924',
     'scene-0925', 'scene-0926', 'scene-0927', 'scene-0928', 'scene-0929', 'scene-0930', 'scene-0931', 'scene-0962',
     'scene-0963', 'scene-0966', 'scene-0967', 'scene-0968', 'scene-0969', 'scene-0971', 'scene-0972', 'scene-1059',
     'scene-1060', 'scene-1061', 'scene-1062', 'scene-1063', 'scene-1064', 'scene-1065', 'scene-1066', 'scene-1067',
     'scene-1068', 'scene-1069', 'scene-1070', 'scene-1071', 'scene-1072', 'scene-1073']
val_night = ['scene-1059', 'scene-1060', 'scene-1061', 'scene-1062', 
            'scene-1063', 'scene-1064', 'scene-1065', 'scene-1066', 
            'scene-1067', 'scene-1068', 'scene-1069', 'scene-1070', 
            'scene-1071', 'scene-1072', 'scene-1073']

if __name__ == "__main__":
    import sys; sys.path.insert(0, os.path.abspath('.'))

    device = torch.device('cuda:0')
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('load_path', default='pred output pkl file')
    parser.add_argument('config', default='test config file path')
    parser.add_argument('--save-path', type=str, default='vis_results/scene')
    parser.add_argument('--scene-idx', type=int, default=None, nargs='+', 
                        help='idx of scene to visualize, scene idx must in the val scene list.')
    parser.add_argument('--frame-idx', type=int, default=None, nargs='+', 
                        help='idx of frame to visualize, the idx corresponds to the order in pkl file.')
    parser.add_argument('--vis-gt', action='store_true', help='vis gt or not')

    args = parser.parse_args()
    print(args)

    cfg = Config.fromfile(args.config)

    cfg = compat_cfg(cfg)
    dataset = build_dataset(cfg.data.test)
    nusc = NuScenes(version='v1.0-trainval', dataroot='/data2/mqh/code/BEVDet/data/nuscenes',
                    verbose=True)

    res = mmcv.load(args.load_path)

    for index in range(len(dataset)):
        info = dataset.data_infos[index]
        scene_name = nusc.get('scene', info['scene_token'])['name']
        if args.scene_idx is not None and int(scene_name[6:]) not in args.scene_idx:
            continue
        if args.frame_idx is not None and index not in args.frame_idx:
            continue
        frame_dir = os.path.join(args.save_path, scene_name, str(index))
        if os.path.exists(frame_dir):
            continue
        os.makedirs(frame_dir, exist_ok=True)
        ego2cam_rts = []
        cam_positions = []
        focal_positions = []
        cam_names = []
        
        for cam_type, cam_info in info['cams'].items():
            cam_names.append(cam_type)
            cam2ego = rt2mat(cam_info['sensor2ego_translation'], cam_info['sensor2ego_rotation'])
            f = 0.0055
            cam_position = cam2ego @ np.array([0., 0., 0., 1.]).reshape([4, 1])
            cam_positions.append(cam_position.flatten()[:3])
            focal_position = cam2ego @ np.array([0., 0., f, 1.]).reshape([4, 1])
            focal_positions.append(focal_position.flatten()[:3])

        occ_gt = np.load(os.path.join(info['occ_path'],'labels.npz'))

        gt_vox = occ_gt['semantics']
        visible_mask = occ_gt['mask_lidar'].astype(bool)
        for cls in range(11): # foreground do not use visible mask
            mask = gt_vox == cls
            visible_mask[mask] = True

        pred_vox = res[index]
        if isinstance(pred_vox, dict):
            pred_vox = pred_vox['occ']
        pred_vox[~visible_mask] = 17

        voxel_origin = [-40, -40, -1.0]
        voxel_max = [40.0, 40.0, 5.4]
        grid_size = [200, 200, 16]
        resolution = [(e - s) / l for e, s, l in zip(voxel_max, voxel_origin, grid_size)]

        if args.vis_gt:
            for cam_type, cam_info in info['cams'].items():
                cam_path = cam_info['data_path']
                shutil.copy(cam_path, os.path.join(frame_dir, str(cam_type)+'.jpg'))

        # trans to lidar cord
        gt_vox_lidar = gt_vox.copy().transpose(2, 0, 1)  # h w z -> z h w
        gt_vox_lidar = np.rot90(gt_vox_lidar, 1, [1, 2])
        gt_vox_lidar = gt_vox_lidar.transpose(1, 2, 0)
        pred_vox_lidar = pred_vox.copy().transpose(2, 0, 1)  # h w z -> z h w
        pred_vox_lidar = np.rot90(pred_vox_lidar, 1, [1, 2])
        pred_vox_lidar = pred_vox_lidar.transpose(1, 2, 0)
        draw(pred_vox,
             voxel_origin,
             voxels_lidar=pred_vox_lidar,
             save_dirs=frame_dir,
             cam_positions=cam_positions,
             focal_positions=focal_positions,
             cam_names=cam_names,
             mode=0)
        if args.vis_gt:
            draw(gt_vox,
                voxel_origin,
                voxels_lidar=gt_vox_lidar,
                save_dirs=frame_dir,
                cam_positions=cam_positions,
                focal_positions=focal_positions,
                cam_names=cam_names,
                mode=1)

