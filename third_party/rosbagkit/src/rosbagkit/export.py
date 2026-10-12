from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from rosbagkit.conversions.image import read_image_msg, save_image


def msgs_to_dataframe(
    msgs: Sequence[tuple[float, object]], reader_fn: Callable[[object], dict[str, Any]]
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for ts, msg in tqdm(msgs, leave=False, dynamic_ncols=True):
        if _has_invalid_header_timestamp(ts, msg):
            tqdm.write(f"[WARN] Invalid timestamp {ts} for message {msg}")
            continue

        try:
            row = dict(reader_fn(msg))
            row["timestamp"] = ts
            row = {k: (v.flatten().tolist() if isinstance(v, np.ndarray) else v) for k, v in row.items()}
            records.append(row)
        except Exception as exc:
            tqdm.write(f"[WARN] Failed to convert message {msg}: {exc}")

    df = pd.DataFrame(records)
    if "timestamp" in df.columns:
        df = df[["timestamp", *[c for c in df.columns if c != "timestamp"]]]

    return df


def export_image_msgs(
    msgs: Sequence[tuple[float, object]],
    outdir: str | Path,
    ts_file: str | Path,
    prefix: str = "",
    image_transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> None:
    outdir = Path(outdir)
    ts_file = Path(ts_file)
    timestamps: list[float] = []

    for frame_idx, (ts, msg) in enumerate(tqdm(msgs, desc=f"process {outdir.name}", leave=False, dynamic_ncols=True)):
        if _has_invalid_header_timestamp(ts, msg):
            tqdm.write(f"[WARN] Invalid timestamp {ts} for message {msg}")
            continue

        outfile = outdir / f"{prefix}{frame_idx:06d}.png"

        try:
            image = read_image_msg(msg)
            if image is None:
                tqdm.write(f"[WARN] Empty image for message {msg}")
                continue
            if image_transform is not None:
                image = image_transform(image)
            saved = save_image(image, str(outfile))
        except Exception as exc:
            tqdm.write(f"[WARN] Failed to process/save image {outfile}: {exc}")
            continue

        if not saved:
            tqdm.write(f"[WARN] Failed to save image {outfile}")
            continue

        timestamps.append(ts)

    np.savetxt(ts_file, np.array(timestamps).reshape(-1, 1), fmt="%.6f", delimiter=",")
    tqdm.write(f"[SUCCESS] Saved {len(timestamps)} images to {outdir}")


###
# Helper functions
###


def _has_invalid_header_timestamp(ts: float, msg: object) -> bool:
    return hasattr(msg, "header") and ts < 1e-3
