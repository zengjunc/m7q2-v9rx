import argparse
import logging
import re
from pathlib import Path

import yaml

from rosbagkit.bagreader import get_topics_from_bagfile, read_bagfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class _CalibrationDumper(yaml.SafeDumper):
    pass


def _represent_flow_style_list(dumper: _CalibrationDumper, data: list):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


_CalibrationDumper.add_representer(list, _represent_flow_style_list)


def camera_info_to_calibration(msg) -> dict:
    frame_id = getattr(getattr(msg, "header", None), "frame_id", "")
    distortion = [float(x) for x in msg.d]

    return {
        "frame_id": frame_id,
        "width": int(msg.width),
        "height": int(msg.height),
        "camera_matrix": {
            "rows": 3,
            "cols": 3,
            "data": [float(x) for x in msg.k],
        },
        "distortion_model": msg.distortion_model,
        "distortion_coefficients": {
            "rows": 1,
            "cols": len(distortion),
            "data": distortion,
        },
        "rectification_matrix": {
            "rows": 3,
            "cols": 3,
            "data": [float(x) for x in msg.r],
        },
        "projection_matrix": {
            "rows": 3,
            "cols": 4,
            "data": [float(x) for x in msg.p],
        },
    }


def calibration_to_yaml_text(camera_info: dict) -> str:
    return yaml.dump(camera_info, Dumper=_CalibrationDumper, sort_keys=False)


def extract_camera_info(bagfile: str):
    topics = get_topics_from_bagfile(bagfile)
    cam_info_topics = {topic for topic in topics if "camera_info" in topic}

    if not cam_info_topics:
        logger.warning(f"[TOPIC] No camera_info topics found in {bagfile}")
        return

    logger.info(f"[FOUND] {len(cam_info_topics)} camera_info topics found.")
    topics_to_msgs = read_bagfile(bagfile, cam_info_topics, early_return=True)

    for topic in sorted(cam_info_topics):
        msgs = topics_to_msgs.get(topic, [])
        if not msgs:
            logger.warning(f"[EMPTY] No messages for topic: {topic}")
            continue

        msg = msgs[0][1]
        camera_info = camera_info_to_calibration(msg)
        outfile = Path(bagfile).parent / topic_to_intrinsics_filename(topic)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        outfile.write_text(calibration_to_yaml_text(camera_info))

        logger.info(f"[SAVED] {topic} -> {outfile}")


def topic_to_intrinsics_filename(topic: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_]+", "_", topic.strip("/"))
    if name.endswith("_camera_info"):
        name = name[: -len("_camera_info")] + "_intrinsics"
    return name + ".yaml"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract CameraInfo from ROS bagfile")
    parser.add_argument("bagfile", help="Path to bagfile")
    args = parser.parse_args()

    extract_camera_info(args.bagfile)
