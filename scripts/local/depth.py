# LOCAL ADDITION: project data/baseline tool; not upstream BEV-Patch-PF.
"""Generate rectified stereo depth in uint16 millimeters, without changing RGB."""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene-dir', type=Path, required=True)
    args = parser.parse_args()
    records = {}
    calibration = Path(__file__).resolve().parents[2] / 'third_party/tartandrive/calibration/multisense_intrinsics.txt'
    for part in calibration.read_text().split('---'):
        lines = part.strip().splitlines()
        if lines:
            records[lines[0]] = yaml.safe_load('\n'.join(lines[1:]))
    left_p = records['/multisense/left/camera_info']['P']
    right_p = records['/multisense/right/image_rect/camera_info']['P']
    fx = left_p[0]
    baseline = -right_p[3] / right_p[0]
    matcher = cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=5,
        P1=200, P2=800, disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100,
        speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    cv2.setNumThreads(4)
    output = args.scene_dir / 'depth'
    output.mkdir(exist_ok=True)
    images = sorted((args.scene_dir / 'color').glob('*.png'))
    timestamps = np.atleast_1d(np.loadtxt(args.scene_dir / 'timestamps.txt'))
    if len(images) != len(timestamps):
        raise ValueError('Image/timestamp count mismatch')
    valid_fractions = []
    start = time.monotonic()
    for i, image in enumerate(images):
        left = cv2.imread(str(args.scene_dir / 'left' / image.name), 0)
        right = cv2.imread(str(args.scene_dir / 'right' / image.name), 0)
        if left is None or right is None or left.shape != right.shape:
            raise ValueError(f'Invalid stereo pair: {image.name}')
        disparity = matcher.compute(left, right).astype(np.float32) / 16
        depth = np.zeros_like(disparity)
        good = disparity > 0
        depth[good] = fx * baseline / disparity[good]
        good &= (depth >= .1) & (depth <= 65.5)
        depth[~good] = 0
        mm = np.round(depth * 1000).astype(np.uint16)
        target = output / image.name
        temporary = target.with_name(target.stem + '.tmp.png')
        if not cv2.imwrite(str(temporary), mm):
            raise RuntimeError(f'Cannot save {temporary}')
        check = cv2.imread(str(temporary), -1)
        if check.dtype != np.uint16 or not np.array_equal(check, mm):
            raise RuntimeError(f'Depth verification failed: {image.name}')
        temporary.replace(target)
        valid_fractions.append(float(good.mean()))
        if i % 50 == 0 or i == len(images) - 1:
            print(f'Depth {i+1}/{len(images)}, valid={good.mean():.3f}', flush=True)
    if {p.name for p in output.glob('*.png')} != {p.name for p in images}:
        raise ValueError('Depth filenames do not match RGB')
    report = dict(method='OpenCV SGBM', frames=len(images), fx=fx, baseline_m=baseline,
        calibration=str(calibration), units='uint16 millimeters; invalid=0',
        valid_fraction_quantiles=np.quantile(valid_fractions, [0, .5, 1]).tolist(),
        runtime_s=time.monotonic()-start)
    (args.scene_dir / 'depth_manifest.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
