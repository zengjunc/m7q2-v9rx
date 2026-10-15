# 复现适配：复用图像/深度，整理参考、TartanVO 和真实 IMU 为 BEVLoc 输入格式。
"""Adapt an already processed scene to BEVLoc's official TartanDrive loader.

Images and millimeter depth are hardlinked. Full GPS/VO poses and real Novatel
IMU messages are retained; the official loader handles GPS/IMU synchronization.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from rosbags.highlevel import AnyReader


def prepare(processed, extracted, bags, output):
    ts = np.loadtxt(processed / "timestamps.txt")
    colors = sorted((processed / "color").glob("*.png"))
    depths = sorted((processed / "depth").glob("*.png"))
    if len(colors) != len(ts) or len(depths) != len(ts):
        raise ValueError("Image/depth/timestamp counts differ")
    output.mkdir(parents=True, exist_ok=True)
    for folder, sources in [("image_left_color", colors), ("depth_left", depths)]:
        target = output / folder
        target.mkdir(exist_ok=True)
        for i, source in enumerate(sources):
            dest = target / f"{i:06d}.png"
            if dest.exists():
                if not os.path.samefile(source, dest):
                    raise ValueError(f"Existing input differs: {dest}")
            else:
                os.link(source, dest)
    np.savetxt(output / "image_left_color" / "timestamps.txt", ts, fmt="%.9f")

    gps = np.loadtxt(extracted / "odom.csv", delimiter=",", skiprows=1)
    target = output / "gps_odom"
    target.mkdir(exist_ok=True)
    np.save(target / "odometry.npy", gps[:, 1:])
    np.savetxt(target / "timestamps.txt", gps[:, 0], fmt="%.9f")

    vo = np.loadtxt(extracted / "tartanvo_odom.csv", delimiter=",", skiprows=1)
    vo = vo[np.argsort(vo[:, 0])]
    _, unique = np.unique(vo[:, 0], return_index=True)
    vo = vo[unique]
    # Official loader prepends identity for frame zero. Store remaining poses.
    query = ts[1:]
    if query[0] < vo[0, 0] or query[-1] > vo[-1, 0]:
        raise ValueError("VO does not cover the image times; refusing extrapolation")
    xyz = np.column_stack([np.interp(query, vo[:, 0], vo[:, k]) for k in range(1, 4)])
    quat = Slerp(vo[:, 0], Rotation.from_quat(vo[:, 4:8]))(query).as_quat()
    target = output / "tartanvo_odom"
    target.mkdir(exist_ok=True)
    np.save(target / "odometry.npy", np.column_stack([xyz, quat]))

    target = output / "novatel_imu"
    target.mkdir(exist_ok=True)
    data_path, time_path = target / "imu.npy", target / "timestamps.txt"
    if not (data_path.exists() and time_path.exists()):
        rows = []
        with AnyReader(sorted(bags.glob("*.bag"))) as reader:
            connections = [c for c in reader.connections if c.topic == "/novatel/imu/data"]
            if not connections:
                raise ValueError("No Novatel IMU topic found")
            for connection, _, raw in reader.messages(connections=connections):
                msg = reader.deserialize(raw, connection.msgtype)
                stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
                a, w = msg.linear_acceleration, msg.angular_velocity
                rows.append([stamp, a.x, a.y, a.z, w.x, w.y, w.z])
        imu = np.asarray(rows)
        imu = imu[np.argsort(imu[:, 0])]
        np.save(data_path, imu[:, 1:])
        np.savetxt(time_path, imu[:, 0], fmt="%.9f")
    report = {"frames": len(ts), "processed": str(processed),
              "extracted": str(extracted), "raw_bags": str(bags),
              "depth_format": "millimeters", "motion_prior": "bag TartanVO",
              "gps_pose_coordinates": "original bag convention (N, -E)",
              "first_vo_pose": "identity prepended by official loader",
              "imu_topic": "/novatel/imu/data", "imu_time_source": "header",
              "imu_samples": int(len(np.load(data_path))),
              "image_storage": "hardlinks", "output": str(output)}
    (output / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    for name in ("processed", "extracted", "bags", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.processed.resolve(), args.extracted.resolve(), args.bags.resolve(), args.output.resolve())
