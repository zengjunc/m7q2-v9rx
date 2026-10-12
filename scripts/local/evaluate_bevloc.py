# LOCAL ADDITION: BEVLoc input/evaluation adapter.
"""Evaluate saved BEVLoc poses against the paired official reference poses."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def evaluate(run, input_frames):
    predicted = list((run / "traj").glob("*_BEVLoc_0.csv"))
    reference = list((run / "traj").glob("*_GPS_0.csv"))
    if len(predicted) != 1 or len(reference) != 1:
        raise ValueError("Expected exactly one estimated and reference trajectory")
    pred = np.atleast_2d(np.loadtxt(predicted[0], delimiter=",", skiprows=1))
    gps = np.atleast_2d(np.loadtxt(reference[0], delimiter=",", skiprows=1))
    for data in [pred, gps]:
        if not np.isfinite(data).all() or np.any(np.diff(data[:, 0]) <= 0):
            raise ValueError("Non-finite or unordered poses")
    idx = np.searchsorted(gps[:, 0], pred[:, 0])
    if (idx >= len(gps)).any() or not np.allclose(gps[idx, 0], pred[:, 0], atol=1e-6, rtol=0):
        raise ValueError("Estimated poses do not have paired reference timestamps")
    ref = gps[idx]
    error = np.linalg.norm(pred[:, 1:3] - ref[:, 1:3], axis=1)
    report = {"input_frames": input_frames, "reference_output_points": len(gps),
              "frames": len(pred), "missing_prediction_points": len(gps)-len(pred),
              "RMSE_m": float(np.sqrt(np.mean(error**2))),
              "mean_m": float(error.mean()), "median_m": float(np.median(error)),
              "final_m": float(error[-1]), "max_m": float(error.max()),
              "reference": "bag /odom, synchronized by official loader",
              "frame": "EPSG:26917 map frame, paired estimated/reference positions",
              "sampling": "one output per batch (8 input images; final batch may be shorter)",
              "alignment": "none: direct paired absolute position error",
              "position_assistance": "official closed_loop prior reset every 5 batches",
              "heading_assistance": "official reference yaw each batch",
        "matching_gate": "official reference pixel-error gate rmse < 300",
              "prediction_csv": str(predicted[0]), "reference_csv": str(reference[0])}
    (run / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savetxt(run / "position_errors.csv", np.column_stack([pred[:, 0], error]),
               delimiter=",", header="timestamp,error_m", comments="", fmt="%.9f")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    origin = gps[0, 1:3]
    axes[0].plot(gps[:, 1]-origin[0], gps[:, 2]-origin[1], label="Reference", color="black")
    axes[0].plot(pred[:, 1]-origin[0], pred[:, 2]-origin[1], label="BEVLoc")
    axes[0].set(xlabel="East from start (m)", ylabel="North from start (m)")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].legend()
    axes[1].plot(pred[:, 0]-pred[0, 0], error)
    axes[1].set(xlabel="Time from first output (s)", ylabel="Position error (m)")
    for ax in axes: ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(run / "trajectory_comparison.png", dpi=150)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--input-frames", required=True, type=int)
    args = parser.parse_args()
    evaluate(args.run.resolve(), args.input_frames)
