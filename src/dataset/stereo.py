# 复现新增：TartanDrive 双目一致性软权重；矛盾像素降权，一致与未知像素保留。
from pathlib import Path

import cv2
import numpy as np
import torch


def load_depth_quality(depth_path: Path, shape, conflict_weight=.25):
    """Return an H×W quality tensor; cache native-resolution conflict masks per scene."""
    scene = depth_path.parent.parent
    cache = scene / 'stereo_conflict' / depth_path.name
    if cache.exists():
        conflict = cv2.imread(str(cache), cv2.IMREAD_GRAYSCALE) > 0
    else:
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
        left = cv2.imread(str(scene / 'left' / depth_path.name), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(scene / 'right' / depth_path.name), cv2.IMREAD_GRAYSCALE)
        matcher = cv2.StereoSGBM_create(minDisparity=-127, numDisparities=128, blockSize=5,
            P1=200, P2=800, disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100,
            speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        right_disp = matcher.compute(right, left).astype(np.float32) / 16
        valid = np.isfinite(depth) & (depth >= .1) & (depth <= 65.5)
        # fB 来自 TartanDrive Multisense 标定右相机 P[3] 的负值，与现有左深度一致。
        disp = np.divide(100.14994812011719, depth, out=np.zeros_like(depth), where=valid)
        x = np.rint(np.arange(depth.shape[1])[None, :] - disp).astype(np.int32)
        sampled = np.take_along_axis(right_disp, x.clip(0, depth.shape[1] - 1), axis=1)
        checked = valid & (x >= 0) & (x < depth.shape[1]) & (sampled < 0) & (sampled >= -127)
        conflict = checked & (np.abs(disp + sampled) > 1.)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(cache), conflict.astype(np.uint8) * 255)
    conflict = cv2.resize(conflict.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(np.where(conflict, conflict_weight, 1).astype(np.float32))
