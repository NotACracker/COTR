import torch
import torch.nn.functional as F


def multiscale_supervision(gt_occ, ratio, gt_shape):
    """
    原论文的下采样gt方法, 不适用于比赛的gt
    change ground truth shape as (B, W, H, Z) for each level supervision
    params:
        gt_occ: 原论文生成的gt_occ , shape = [N,4], N为occupied voxel数, 4表示(xyz + label id)
        ratio: downsample ratio, choice list =[1,2,4,8]
        gt_shape: torch.Size([1, 17, 25, 25, 2]), [bs, cls_num, W, H , Z]
    return:
        gt: shape = (B, W//ratio, H//ratio, Z)
    """

    gt = torch.zeros([gt_shape[0], gt_shape[2], gt_shape[3], gt_shape[4]]).to(gt_occ.device).type(gt_occ.dtype)
    for i in range(gt.shape[0]):
        coords = gt_occ[i][:, :3].type(torch.long) // ratio
        gt[i, coords[:, 0], coords[:, 1], coords[:, 2]] = gt_occ[i][:, 3]

    return gt


def multiscale_supervision_cvpr(gt_occ, ratio, gt_shape, mask_camera=None):
    """
    change ground truth shape as (B, W, H, Z) for each level supervision
    用于cvpr2023-occ比赛的label asign
    """
    start = int(ratio // 2)
    gt = gt_occ[:, start::ratio, start::ratio, start::ratio]
    # gt = gt_occ[:, ::ratio, ::ratio, ::ratio]

    if mask_camera is not None:
        mask_camera = mask_camera.type(torch.bool)[:, ::ratio, ::ratio, ::ratio]
    return gt, mask_camera


def geo_scal_loss(pred, ssc_target, semantic=True, mask_camera=None):
    """
    Args:
        pred: shape = [bs,cls_nums,H,W,Z]
        ssc_target: shape=[bs,H,W,Z]
        mask_camera: shape=[bs,H,W,Z]
    """
    free_idx = 17
    # Get softmax probabilities
    if semantic:
        pred = F.softmax(pred, dim=1)

        # Compute empty and nonempty probabilities
        empty_probs = pred[:, free_idx, :, :, :]  # 原本是0, 比赛是-1
    else:
        empty_probs = 1 - torch.sigmoid(pred)
    nonempty_probs = 1 - empty_probs

    # Remove unknown voxels
    mask = ssc_target != 255
    if mask_camera is not None:
        mask = mask * mask_camera
    nonempty_target = ssc_target != free_idx
    nonempty_target = nonempty_target[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]

    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / max(nonempty_probs.sum(), 1)
    recall = intersection / max(nonempty_target.sum(), 1)
    spec = ((1 - nonempty_target) * (empty_probs)).sum() / max((1 - nonempty_target).sum(), 1)

    # if not ((1 - nonempty_target).sum() > 0 and nonempty_probs.sum() > 0 and nonempty_target.sum() > 0):
    #     rank, world_size = get_dist_info()
    #     if rank == 0:
    #         pdb.set_trace()
    #     else:
    #         pdb.set_trace()
    return (
        F.binary_cross_entropy(precision, torch.ones_like(precision))
        + F.binary_cross_entropy(recall, torch.ones_like(recall))
        + F.binary_cross_entropy(spec, torch.ones_like(spec))
    )


def sem_scal_loss(pred, ssc_target, mask_camera=None):
    """
    Args:
        pred: shape = [bs,cls_nums,H,W,Z]
        ssc_target: shape=[bs,H,W,Z]
        mask_camera: shape=[bs,H,W,Z]
    """
    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)
    loss = 0
    count = 0
    mask = ssc_target != 255
    if mask_camera is not None:
        mask = mask * mask_camera
    n_classes = pred.shape[1]
    for i in range(0, n_classes):

        # Get probability of class i
        p = pred[:, i, :, :, :]

        # Remove unknown voxels
        target_ori = ssc_target
        p = p[mask]
        target = ssc_target[mask]

        completion_target = torch.ones_like(target)
        completion_target[target != i] = 0
        completion_target_ori = torch.ones_like(target_ori).float()
        completion_target_ori[target_ori != i] = 0
        if torch.sum(completion_target) > 0:
            count += 1.0
            nominator = torch.sum(p * completion_target)
            loss_class = 0
            if torch.sum(p) > 0:
                precision = nominator / (torch.sum(p))
                loss_precision = F.binary_cross_entropy(precision, torch.ones_like(precision))
                loss_class += loss_precision
            if torch.sum(completion_target) > 0:
                recall = nominator / (torch.sum(completion_target))
                loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))
                loss_class += loss_recall
            if torch.sum(1 - completion_target) > 0:
                specificity = torch.sum((1 - p) * (1 - completion_target)) / (torch.sum(1 - completion_target))
                loss_specificity = F.binary_cross_entropy(specificity, torch.ones_like(specificity))
                loss_class += loss_specificity
            loss += loss_class
    return loss / count


def compute_ce_loss_with_mask(criterion, pred, gt_occ, mask_camera, num_classes):
    """
    Args:
        pred: shape = bs, w, h, z, cls_nums
        gt_occ: shape = bs, w, h, z
        mask_c
    """
    gt_occ = gt_occ.reshape(-1)
    pred = pred.permute(0, 2, 3, 4, 1)  # bs, cls_nums, h,w,z -> bs, h,w,z, cls_nums
    pred = pred.reshape(-1, num_classes)
    mask_camera = mask_camera.reshape(-1)
    num_total_samples = mask_camera.sum()
    loss_occ = criterion(pred, gt_occ, mask_camera, avg_factor=num_total_samples)

    return loss_occ
