# 复现适配：评估 PF、cuVSLAM 和预计算 Super Odometry 的二维位置误差。
"""LOCAL ADDITION: reference-initialized VO baselines and direct XY error metrics."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def read(path):
    data = np.atleast_2d(np.loadtxt(path, delimiter=',', skiprows=1))
    if len(data) < 1 or not np.isfinite(data).all() or np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError(f'Invalid trajectory: {path}')
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=['pf', 'cuvslam', 'superodom'], required=True)
    p.add_argument('--scene-dir', type=Path, required=True)
    p.add_argument('--extracted', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    reference = read(a.scene_dir / 'odom.csv')
    if a.limit:
        reference = reference[:a.limit]
    alignment = 'none; direct saved absolute positions'
    if a.method == 'pf':
        prediction = read(a.output / 'pf_traj.csv')
        indices = np.searchsorted(reference[:, 0], prediction[:, 0])
        if np.any(indices >= len(reference)) or not np.allclose(reference[indices, 0], prediction[:, 0], atol=1e-6, rtol=0):
            raise ValueError('PF timestamps do not match reference')
        reference = reference[indices]
    elif a.method == 'cuvslam':
        from dataset.tartandrive import TartanDriveDataset
        from dataset.utils import se3_poses_from_rows, compute_se3_actions
        from manifpy import SE2
        vo = read(a.scene_dir / ('pycuvslam_probe.csv' if a.limit else 'pycuvslam.csv'))
        if len(vo) != len(reference) or not np.allclose(vo[:, 0], reference[:, 0], atol=1e-6, rtol=0):
            raise ValueError('VO timestamps do not match reference')
        actions = compute_se3_actions(se3_poses_from_rows(vo[:, 1:8], camera_frame=True, cam2enu=TartanDriveDataset.CAM2ENU))
        pose = SE2(*reference[0, 1:4])
        prediction = []
        for stamp, action in zip(reference[:, 0], actions):
            pose = pose * SE2(*action)
            prediction.append([stamp, *pose.translation(), pose.angle()])
        prediction = np.asarray(prediction)
        alignment = 'same official camera-to-ENU actions; reference first pose only'
    else:
        from scipy.spatial.transform import Rotation, Slerp
        if a.extracted is None:
            p.error('--extracted is required for superodom')
        lidar = read(a.extracted / 'lidar_odom.csv')
        reference = reference[(reference[:, 0] >= lidar[0, 0]) & (reference[:, 0] <= lidar[-1, 0])]
        if len(reference) < 2:
            raise ValueError('Insufficient shared coverage')
        ts = reference[:, 0]
        xy = np.column_stack([np.interp(ts, lidar[:, 0], lidar[:, k]) for k in [1, 2]])
        rotation = Slerp(lidar[:, 0] - lidar[0, 0], Rotation.from_quat(lidar[:, 4:8]))(ts - lidar[0, 0]).as_matrix()
        yaw = np.arctan2(rotation[:, 1, 0], rotation[:, 0, 0])
        angle = reference[0, 3] - yaw[0]
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        xy = (xy - xy[0]) @ rot.T + reference[0, 1:3]
        prediction = np.column_stack([ts, xy, np.arctan2(np.sin(yaw + angle), np.cos(yaw + angle))])
        alignment = 'precomputed lidar_odom; first shared reference pose only, no scale/trajectory fitting'
    error = np.linalg.norm(prediction[:, 1:3] - reference[:, 1:3], axis=1)
    report = dict(method=a.method, frames=len(error), RMSE_m=float(np.sqrt(np.mean(error ** 2))),
                  mean_m=float(error.mean()), median_m=float(np.median(error)),
                  reference='bag /odom', alignment=alignment)
    a.output.mkdir(parents=True, exist_ok=True)
    if a.method != 'pf':
        np.savetxt(a.output / 'trajectory.csv', prediction, delimiter=',', header='timestamp,x,y,angle', comments='', fmt='%.9f')
    np.savetxt(a.output / 'position_errors.csv', np.column_stack([prediction[:, 0], error]), delimiter=',', header='timestamp,error_m', comments='')
    (a.output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    origin = reference[0, 1:3]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(reference[:, 1]-origin[0], reference[:, 2]-origin[1], label='/odom reference')
    ax.plot(prediction[:, 1]-origin[0], prediction[:, 2]-origin[1], label=a.method)
    ax.set(xlabel='East from start (m)', ylabel='North from start (m)')
    ax.set_aspect('equal', adjustable='datalim'); ax.legend(); ax.grid(); fig.tight_layout()
    fig.savefig(a.output / 'trajectory_comparison.png', dpi=150); plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
