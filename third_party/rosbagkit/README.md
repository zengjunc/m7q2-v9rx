# rosbagkit

`rosbagkit` is a lightweight toolkit for reading ROS bagfiles and exporting
common sensor streams to disk. It is built around YAML-driven extraction, with
support for image, depth, pointcloud-derived depth, and CSV exports, plus
utilities for TF and camera calibration extraction.

The repository is organized around:

- a Python package in `src/rosbagkit`
- extraction and utility scripts in `scripts/`
- example YAML configs in `config/extract/`

## Requirements

- Python 3.12+
- ROS bagfiles supported by [`rosbags`](https://pypi.org/project/rosbags/)

## Install

Choose one environment setup path.

### Option 1: `uv`

```bash
uv sync
source .venv/bin/activate
```

[uv](https://docs.astral.sh/uv/getting-started/installation/) must be installed
separately before running these commands.

### Option 2: `conda`

```bash
conda create -n rosbagkit python=3.12
conda activate rosbagkit
pip install -e .
```

## Usage

### Extract bag contents from a YAML config

The standard workflow is:

1. Start from a template in `config/extract/`.
2. Update the bag paths, topics, and scenes for your dataset.
3. Run the extractor on that YAML file.

For example:

```bash
python scripts/extract_bagfile.py config/extract/UT-SARA-GQ.yaml
```

Each extraction config follows this shape. `bagfile_root` is the base directory
containing the bag data, and each entry in `scenes.<name>.bagfiles` is a path
relative to that root.

```yaml
bagfile_root: /path/to/bagfiles
output_root: /path/to/output

topics:
  /camera/color/image_raw:
    format: image
    outdir: image_raw
  /gps/fix:
    format: csv
    outpath: gps.csv

scenes:
  example_scene:
    bagfiles:
      - example_run/example_0.bag
    start: 0
    end: -1
```

Supported topic output formats:

- `image`
- `depth`
- `pointcloud_depth`
- `csv`

Useful starting points in `config/extract/` include:

- `UT-SARA-GQ.yaml` for stereo extraction with rectification
- `arl_lonebot.yaml` for a RealSense-style RGB-D layout
- `tartandrive2.yaml` for another multi-sensor extraction example

### Optional synchronized filtering

Use `sync` when multiple topics should be filtered to matched timestamps before
they are exported.

```yaml
sync:
  enabled: true
  topics:
    - /camera/color/image_raw/compressed
    - /camera/depth/image_rect_raw
  threshold: 0.005
```

The first topic in `sync.topics` is the reference topic. A reference timestamp
is kept only when every other listed topic has a nearest match within the
configured threshold. This filtering happens before raw export, undistortion,
and rectification.

### Optional image undistortion

Use topic-level `undistortion` when an image topic should be exported as
undistorted frames instead of raw frames.

```yaml
topics:
  /camera/color/image_raw/compressed:
    format: image
    outdir: image_raw
    undistortion:
      enabled: true
      calib: /path/to/color_intrinsics.yaml
```

### Optional stereo rectification

Use a top-level `rectification` block when a stereo pair should be
synchronized and rectified before writing images to disk.

```yaml
rectification:
  enabled: true
  left_topic: /stereo/left/image_raw
  right_topic: /stereo/right/image_raw
  left_calib: /path/to/left_intrinsics.yaml
  right_calib: /path/to/right_intrinsics.yaml
  extrinsics: /path/to/left_to_right.yaml
  threshold: 0.005
  output_dir: 2d_rect
  left_subdir: cam_left
  right_subdir: cam_right
  timestamp_file: timestamps.txt
```

The configured stereo topics are rectified in memory, and the raw export for
that pair is skipped. Only the rectified image pair is written under
`output_dir`. See
[`config/extract/UT-SARA-GQ.yaml`](config/extract/UT-SARA-GQ.yaml) for a public
example.

## Additional Utilities

### Extract a TF transform from a bagfile

```bash
python scripts/extract_tf.py /path/to/bag --src source_frame --tgt target_frame
```

This writes `tf_<src>_to_<tgt>.yaml` next to the bagfile with the resolved
transform chain.

### Extract camera intrinsics from `camera_info`

```bash
python scripts/extract_camera_info.py /path/to/bag
```

This scans for `camera_info` topics and writes ROS-style `*_intrinsics.yaml`
files next to the bagfile.

### Rewrite bagfiles for release

```bash
python scripts/release_bagfile.py /path/to/release_config.yaml
```

This runs bag rewriting jobs from a release config with `input_root`,
`output_root`, `topics_keep`, and `scenes`.
