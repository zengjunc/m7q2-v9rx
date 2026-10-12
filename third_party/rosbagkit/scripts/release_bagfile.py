"""Process UT-SARA-GQ bagfiles for public release."""

import argparse
from pathlib import Path

import yaml

from rosbagkit import rewrite_bagfile


def run_release_jobs(config: dict) -> None:
    required_keys = {"input_root", "output_root", "topics_keep", "scenes"}
    missing_keys = sorted(required_keys - config.keys())
    if missing_keys:
        raise KeyError(f"Missing release config keys: {missing_keys}")

    input_root = Path(config["input_root"])
    output_root = Path(config["output_root"])
    topics_keep = config["topics_keep"]
    topics_exclude = config.get("topics_exclude", [])

    if not topics_keep:
        raise ValueError("'topics_keep' must contain at least one topic")

    for scene_name, scene_cfg in config["scenes"].items():
        start = float(scene_cfg.get("start", 0.0))
        end = float(scene_cfg.get("end", -1.0))
        if end >= 0 and start > end:
            raise ValueError(f"Scene {scene_name} has start > end: {start} > {end}")

        for bagfile in scene_cfg.get("bagfiles", []):
            relpath = Path(bagfile)
            input_path = input_root / relpath
            output_path = output_root / relpath.with_suffix("")

            if not input_path.exists():
                print(f"[WARN] Missing source bagfile for scene={scene_name}: {input_path}", flush=True)
                continue

            print(
                f"[WRITE] scene={scene_name} source={input_path} output={output_path} window=({start}, {end})",
                flush=True,
            )

            success = rewrite_bagfile(
                source_path=input_path,
                output_path=output_path,
                topics_keep=topics_keep,
                topics_exclude=topics_exclude,
                start=start,
                end=end,
            )

            if not success:
                print(f"[WARN] Failed to write bagfile for source={input_path}", flush=True)
                continue

            print(f"[DONE] scene={scene_name} output={output_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite public-safe ROS bagfiles from a release config.")
    parser.add_argument("config", help="Path to a bag release YAML config.")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_release_jobs(config)


if __name__ == "__main__":
    main()
