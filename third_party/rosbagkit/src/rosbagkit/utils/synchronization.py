import bisect
from typing import Any

import numpy as np
from tqdm import tqdm

SYNC_REQUIRED_KEYS = {"topics"}


def _has_invalid_header_timestamp(ts: float, msg: object) -> bool:
    return hasattr(msg, "header") and ts < 1e-3


def _filter_valid_msgs(msgs: list[tuple[float, object]]) -> list[tuple[float, object]]:
    return [(ts, msg) for ts, msg in msgs if not _has_invalid_header_timestamp(ts, msg)]


def sync_indices_closest(
    left_timestamps: list[float] | np.ndarray,
    right_timestamps: list[float] | np.ndarray,
    threshold: float = 0.005,
) -> tuple[list[int], list[int], list[float]]:
    left_ts = np.asarray(left_timestamps, dtype=float)
    right_ts = np.asarray(right_timestamps, dtype=float)

    left_indices: list[int] = []
    right_indices: list[int] = []
    synced_timestamps: list[float] = []
    used_right: set[int] = set()

    for left_idx, left_ts_val in enumerate(left_ts):
        pos = bisect.bisect_left(right_ts, left_ts_val)
        best_idx = None
        best_diff = threshold

        for right_idx in (pos - 1, pos):
            if 0 <= right_idx < len(right_ts) and right_idx not in used_right:
                diff = abs(left_ts_val - right_ts[right_idx])
                if diff < best_diff:
                    best_diff = diff
                    best_idx = right_idx

        if best_idx is None:
            continue

        left_indices.append(left_idx)
        right_indices.append(best_idx)
        synced_timestamps.append(float(left_ts_val))
        used_right.add(best_idx)

    return left_indices, right_indices, synced_timestamps


def build_sync_config(sync_cfg: dict[str, Any] | None, topics_info: dict[str, dict]) -> dict[str, Any] | None:
    if not sync_cfg or not sync_cfg.get("enabled", False):
        return None

    missing = sorted(SYNC_REQUIRED_KEYS - sync_cfg.keys())
    if missing:
        raise KeyError(f"Missing sync config keys: {missing}")

    raw_topics = sync_cfg["topics"]
    if not isinstance(raw_topics, list) or len(raw_topics) < 2:
        raise ValueError("Sync topics must be a list with at least 2 topics")

    topics = list(raw_topics)
    if len(set(topics)) != len(topics):
        raise ValueError("Sync topics must be unique")

    for topic in topics:
        if topic not in topics_info:
            raise KeyError(f"Sync topic not found in topics config: {topic}")

    threshold = float(sync_cfg.get("threshold", 0.005))
    if threshold <= 0:
        raise ValueError("Sync threshold must be > 0")

    cfg = {
        "topics": topics,
        "reference_topic": topics[0],
        "threshold": threshold,
    }
    tqdm.write(
        f"[SYNC] Enabled synchronized filtering for reference={cfg['reference_topic']} "
        f"topics={len(topics)} threshold={threshold:.6f}"
    )
    return cfg


def filter_synced_messages(
    sync: dict[str, Any], topics_to_msgs: dict[str, list[tuple[float, object]]]
) -> dict[str, list[tuple[float, object]]]:
    reference_topic = sync["reference_topic"]
    synchronized_topics = sync["topics"]
    threshold = sync["threshold"]

    aligned_msgs: dict[str, list[tuple[float, object]]] = {
        reference_topic: _filter_valid_msgs(topics_to_msgs.get(reference_topic, []))
    }
    if not aligned_msgs[reference_topic]:
        tqdm.write(f"[SYNC] No valid messages found for reference topic: {reference_topic}")
        for topic in synchronized_topics:
            topics_to_msgs[topic] = []
        return topics_to_msgs

    for topic in synchronized_topics[1:]:
        topic_msgs = _filter_valid_msgs(topics_to_msgs.get(topic, []))
        if not topic_msgs:
            tqdm.write(f"[SYNC] No valid messages found for sync topic: {topic}")
            for synced_topic in synchronized_topics:
                topics_to_msgs[synced_topic] = []
            return topics_to_msgs

        reference_ts = [ts for ts, _ in aligned_msgs[reference_topic]]
        topic_ts = [ts for ts, _ in topic_msgs]
        reference_idx, topic_idx, _ = sync_indices_closest(reference_ts, topic_ts, threshold=threshold)
        if not reference_idx:
            tqdm.write(f"[SYNC] No synchronized messages matched within threshold {threshold:.6f} for topic: {topic}")
            for synced_topic in synchronized_topics:
                topics_to_msgs[synced_topic] = []
            return topics_to_msgs

        for aligned_topic, msgs in list(aligned_msgs.items()):
            aligned_msgs[aligned_topic] = [msgs[i] for i in reference_idx]
        aligned_msgs[topic] = [topic_msgs[i] for i in topic_idx]

    for topic in synchronized_topics:
        topics_to_msgs[topic] = aligned_msgs[topic]

    tqdm.write(
        f"[SYNC] Filtered {len(synchronized_topics)} topics to "
        f"{len(aligned_msgs[reference_topic])} synchronized messages"
    )
    return topics_to_msgs
