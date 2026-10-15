# 复现适配：配置双目相机并运行 cuVSLAM，保存左相机局部位姿与逐帧跟踪状态。
"""Headless rectified stereo cuVSLAM; exports one row per input frame."""
import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import cuvslam
import numpy as np
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene-dir', required=True, type=Path)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    records = {}
    calibration = Path(__file__).resolve().parents[2] / 'third_party/tartandrive/calibration/multisense_intrinsics.txt'
    for part in calibration.read_text().split('---'):
        lines = part.strip().splitlines()
        if lines:
            records[lines[0]] = yaml.safe_load('\n'.join(lines[1:]))
    projections = [records[key]['P'] for key in
        ['/multisense/left/camera_info', '/multisense/right/image_rect/camera_info']]
    cameras = [cuvslam.Camera(), cuvslam.Camera()]
    for camera, projection in zip(cameras, projections):
        camera.size = (1024, 544)
        camera.focal = [projection[0], projection[5]]
        camera.principal = [projection[2], projection[6]]
    cameras[1].rig_from_camera.translation[0] = -projections[1][3] / projections[1][0]
    if hasattr(cuvslam, 'Odometry'):
        config = cuvslam.Odometry.Config(async_sba=False, rectified_stereo_camera=True)
        tracker = cuvslam.Tracker(cuvslam.Rig(cameras), cuvslam.Tracker.Mode.OdometryOnlyOffline, config)
    else:
        config = cuvslam.Tracker.OdometryConfig(async_sba=False, rectified_stereo_camera=True)
        tracker = cuvslam.Tracker(cuvslam.Rig(cameras), odom_config=config)
    timestamps = np.atleast_1d(np.loadtxt(args.scene_dir / 'timestamps.txt'))
    images = sorted((args.scene_dir / 'color').glob('*.png'))
    if len(images) != len(timestamps) or np.any(np.diff(timestamps) <= 0):
        raise ValueError('Invalid timestamps')
    if args.limit:
        images = images[:args.limit]
        timestamps = timestamps[:args.limit]
    output = args.scene_dir / ('pycuvslam_probe.csv' if args.limit else 'pycuvslam.csv')
    temporary = output.with_suffix('.tmp.csv')
    status = []
    last_pose = None
    started = time.monotonic()
    with temporary.open('w') as handle:
        writer = csv.writer(handle)
        writer.writerow(['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
        for i, (image, stamp) in enumerate(zip(images, timestamps)):
            stereo = [cv2.imread(str(args.scene_dir / side / image.name), 0) for side in ['left', 'right']]
            if any(p is None for p in stereo):
                raise ValueError(f'Cannot read pair {image.name}')
            estimate, _ = tracker.track(int(round((stamp-timestamps[0])*1e9)), stereo)
            tracked = estimate.world_from_rig is not None
            if tracked:
                pose = estimate.world_from_rig.pose
                last_pose = [*pose.translation, *pose.rotation]
            if last_pose is None:
                last_pose = [0, 0, 0, 0, 0, 0, 1]
            writer.writerow([stamp, *last_pose])
            status.append({'frame': i, 'tracked': tracked})
            if i % 50 == 0 or i == len(images)-1:
                handle.flush()
                print(f'cuVSLAM {i+1}/{len(images)}, tracked={tracked}', flush=True)
    temporary.replace(output)
    report = dict(frames=len(images), tracked_frames=sum(s['tracked'] for s in status),
        runtime_s=time.monotonic()-started, coordinate='world_from_left_optical_rig; xyzw',
        failure_policy='hold last pose; preserve frame count; explicit status',
        calibration=str(calibration), statuses=status)
    output.with_suffix('.json').write_text(json.dumps(report, indent=2))
    print(f'Saved {output}; tracked {report["tracked_frames"]}/{len(images)}')

if __name__ == '__main__':
    main()
