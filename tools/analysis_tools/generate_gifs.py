import os, os.path as osp
import PIL.Image as Image
import cv2
import argparse

def cat_images(frame_dir, cam_img_size, pred_img_size, pred_img_size2, spacing):
    cam_imgs = []
    pred_imgs = []
    cam_files = [osp.join(frame_dir, fn) for fn in os.listdir(frame_dir) if fn.startswith('CAM')]
    cam_files = sorted(cam_files)
    cam_imgs.extend([
        Image.open(fn).resize(cam_img_size, Image.BILINEAR) for fn in cam_files
    ])
    pred_files = [osp.join(frame_dir, fn) for fn in os.listdir(frame_dir) if fn.startswith('pred_CAM')]
    pred_files = sorted(pred_files)
    pred_imgs.extend([
        Image.open(fn).resize(cam_img_size, Image.BILINEAR) for fn in pred_files 
    ])
    fn = osp.join(frame_dir, 'pred_normal.png')
    pred_imgs.extend([Image.open(fn).resize(pred_img_size, Image.BILINEAR)])
    fn = osp.join(frame_dir, 'pred_bev.png')
    pred_imgs.extend([Image.open(fn).resize(pred_img_size, Image.BILINEAR).crop([460, 0, 1460, 1080])])

    cam_w, cam_h = cam_img_size
    pred_w, pred_h = pred_img_size
    result_w = cam_w * 6 + 5 * spacing
    result_h = cam_h * 2 + pred_h + 2 * spacing
    
    result = Image.new(pred_imgs[0].mode, (result_w, result_h), (0, 0, 0))
    result.paste(cam_imgs[3], box=(1*cam_w+1*spacing, 0))
    result.paste(cam_imgs[5], box=(2*cam_w+2*spacing, 0))
    result.paste(cam_imgs[4], box=(0, 0))
    result.paste(cam_imgs[0], box=(1*cam_w+1*spacing, 1*cam_h+1*spacing))
    result.paste(cam_imgs[2], box=(0, 1*cam_h+1*spacing))
    result.paste(cam_imgs[1], box=(2*cam_w+2*spacing, 1*cam_h+1*spacing))

    result.paste(pred_imgs[3], box=(4*cam_w+4*spacing, 0))
    result.paste(pred_imgs[5], box=(5*cam_w+5*spacing, 0))
    result.paste(pred_imgs[4], box=(3*cam_w+3*spacing, 0))
    result.paste(pred_imgs[0], box=(4*cam_w+4*spacing, 1*cam_h+1*spacing))
    result.paste(pred_imgs[2], box=(3*cam_w+3*spacing, 1*cam_h+1*spacing))
    result.paste(pred_imgs[1], box=(5*cam_w+5*spacing, 1*cam_h+1*spacing))

    result.paste(pred_imgs[6], box=(0, 2*cam_h+2*spacing))
    result.paste(pred_imgs[7], box=(1*pred_w+1*spacing, 2*cam_h+2*spacing))

    result_path = osp.join(frame_dir, 'cat.png')
    result.save(result_path)

    return result


if __name__ == "__main__":
    parse = argparse.ArgumentParser('')
    parse.add_argument('--scene-dir', type=str, default='vis_results/scene/scene-630', 
                       help='directory of the scene outputs')
    args = parse.parse_args()

    scene_dir = args.scene_dir
    frame_dirs = os.listdir(scene_dir)
    list.sort(frame_dirs)
    cam_img_size = [480, 270]
    pred_img_size = [1920, 1080]
    pred_img_size2 = [1000, 1080]
    spacing = 10

    cat_imgs = []
    for dir in frame_dirs:
        if dir.endswith('.gif'):
            continue
        print(f'processing {osp.join(scene_dir, dir)}')
        cat_img = cat_images(osp.join(scene_dir, dir), cam_img_size, pred_img_size, pred_img_size2, spacing)
        cat_imgs.append(cat_img.resize((cat_img.width//2, cat_img.height//2), resample=Image.LANCZOS))

    cat_imgs[0].save(osp.join(scene_dir, 'video_resize.gif'),
                     save_all=True,
                     append_images=cat_imgs[1:],
                     optimize=True,
                     duration=150,
                     loop=0)
    
    