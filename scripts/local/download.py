#!/usr/bin/env python3
# 复现适配：按官方清单下载单条轨迹的全部 bag 分片，保留部分下载以便继续。
"""Download one TartanDrive 2.0 run without a graphical desktop.

Uses the same public endpoint and assets/files.yaml as tartandrive_gui.py.
Partial downloads are retained as .part files for the next invocation.
"""

import argparse
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import time
from urllib.parse import quote

import boto3
from botocore import UNSIGNED
from botocore.client import Config
import yaml


ROOT = Path(__file__).resolve().parents[2] / "third_party/tartandrive"
ENDPOINT = "https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/AUTH_ac8533a83cff4d48bc8c608ad222d330"
BUCKET = "tartandrive2"
DEFAULT_SCENE = "2023-11-02-16-00-43_down_meadows"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument("--list", action="store_true", help="List scenes without network access")
    parser.add_argument("--dry-run", action="store_true", help="Show remote sizes without downloading")
    args = parser.parse_args()
    manifest = yaml.safe_load((ROOT / "assets/files.yaml").read_text())["bags"]
    if args.list:
        for name, entry in sorted(manifest.items()):
            if isinstance(entry, dict):
                print(name.removeprefix("bags/").rstrip("/"))
        return
    prefix = f"bags/{args.scene}/"
    if prefix not in manifest:
        parser.error("Unknown scene; use --list to see available scenes")
    keys = manifest[prefix]["files"]
    for key in keys:
        path = PurePosixPath(key)
        if path.is_absolute() or ".." in path.parts or not key.startswith(prefix):
            raise ValueError(f"Invalid manifest path: {key}")
    client = boto3.client(
        "s3", endpoint_url=ENDPOINT,
        config=Config(signature_version=UNSIGNED, connect_timeout=15,
                      read_timeout=30, retries={"max_attempts": 2},
                      s3={"addressing_style": "path"}),
    )
    sizes = {}
    for key in keys:
        size = client.head_object(Bucket=BUCKET, Key=key)["ContentLength"]
        sizes[key] = size
        print(f"{size / 1024**3:8.3f} GiB  {key}", flush=True)
    print(f"Total: {sum(sizes.values()) / 1024**3:.3f} GiB", flush=True)
    print(f"Output: {args.output.resolve()}", flush=True)
    if args.dry_run:
        return
    if shutil.which("curl") is None:
        raise RuntimeError("curl is required for resumable downloads")
    args.output.mkdir(parents=True, exist_ok=True)
    needed = 0
    for key, size in sizes.items():
        target = args.output / key
        partial = target.with_name(target.name + ".part")
        if target.exists():
            if target.stat().st_size != size:
                raise RuntimeError(f"Existing file has unexpected size; move it aside: {target}")
        else:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > size:
                raise RuntimeError(f"Partial file is larger than source: {partial}")
            needed += size - offset
    if shutil.disk_usage(args.output).free < needed + 1024**3:
        raise RuntimeError("Insufficient free space (including 1 GiB reserve)")
    for key, size in sizes.items():
        target = args.output / key
        if target.exists():
            print(f"Skip complete file: {target.name}", flush=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        print(f"Downloading: {key}", flush=True)
        if not partial.exists() or partial.stat().st_size != size:
            command = [
                "curl", "--http1.1", "--fail", "--location", "--retry", "3",
                "--connect-timeout", "120", "--speed-limit", "1024",
                "--speed-time", "120", "--continue-at", "-",
                "--output", str(partial),
                f"{ENDPOINT}/{BUCKET}/{quote(key, safe='/')}",
            ]
            # Restart curl with -C - to recalculate the offset after a broken stream.
            retryable = {5, 6, 7, 18, 28, 35, 52, 55, 56, 92}
            for attempt in range(1, 6):
                result = subprocess.run(command)
                if result.returncode == 0:
                    break
                if result.returncode not in retryable or attempt == 5:
                    raise RuntimeError(
                        f"curl exited {result.returncode}; partial data kept at {partial}. "
                        "Rerun the same command to resume."
                    )
                offset = partial.stat().st_size if partial.exists() else 0
                print(f"Connection interrupted; resuming from {offset} bytes "
                      f"(retry {attempt}/4)", flush=True)
                time.sleep(5)
        if partial.stat().st_size != size:
            raise RuntimeError(f"Incomplete download: {partial}; rerun to resume")
        partial.rename(target)
    print("Download complete; all file sizes verified.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted; rerun the same command to resume.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        sys.exit(1)
