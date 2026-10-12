from __future__ import annotations

import math
from pathlib import Path

from rosbags.convert.converter import create_connections_converters
from rosbags.highlevel import AnyReader
from rosbags.rosbag2 import StoragePlugin, Writer
from rosbags.rosbag2.enums import CompressionFormat, CompressionMode
from tqdm import tqdm

from rosbagkit.utils.misc import format_bytes, get_path_size


def rewrite_bagfile(
    source_path: str | Path,
    output_path: str | Path,
    topics_keep: list[str],
    topics_exclude: list[str] | None = None,
    start: float = 0.0,
    end: float = -1.0,
) -> bool:
    source_path = Path(source_path)
    output_path = Path(output_path)

    if output_path.exists():
        tqdm.write(f"[WARN] Output bagfile already exists, skipping: {output_path}")
        return False

    if not topics_keep:
        raise ValueError("'topics_keep' must contain at least one topic")

    keep_rules = list(dict.fromkeys(topics_keep))
    exclude_rules = list(dict.fromkeys(topics_exclude or []))

    start_ns = math.floor(start * 1e9) if start > 0 else None
    stop_ns = math.floor(end * 1e9) + 1 if end >= 0 else None
    if start_ns is not None and stop_ns is not None and start_ns >= stop_ns:
        raise ValueError(f"Invalid time window for {source_path}: start={start}, end={end}")

    tqdm.write(f"Opening {source_path} ({format_bytes(get_path_size(source_path))})...")
    with AnyReader([source_path]) as reader:
        selected_connections = []
        for conn in reader.connections:
            if not _topic_matches(conn.topic, keep_rules):
                continue
            if _topic_matches(conn.topic, exclude_rules):
                continue
            selected_connections.append(conn)

        if not selected_connections:
            tqdm.write(f"[WARN] No messages to keep for {source_path}")
            return False

        tqdm.write(f"[INFO] Selected {len(selected_connections)} connections from {source_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        writer = Writer(output_path, version=8, storage_plugin=StoragePlugin.SQLITE3)
        writer.set_compression(CompressionMode.MESSAGE, CompressionFormat.ZSTD)

        with writer:
            connmap, convmap = create_connections_converters(selected_connections, None, reader, writer)

            for conn, timestamp, rawdata in tqdm(
                reader.messages(connections=selected_connections, start=start_ns, stop=stop_ns),
                total=sum(max(conn.msgcount, 0) for conn in selected_connections),
                desc=f"write {source_path.name}",
                unit="msg",
                dynamic_ncols=True,
            ):
                writer.write(connmap[(conn.id, conn.owner)], timestamp, convmap[conn.msgtype](rawdata))

    return True


###
# Helper functions
###


def _topic_matches(topic: str, rules: list[str]) -> bool:
    return any(topic == rule or topic.startswith(f"{rule}/") for rule in rules)
