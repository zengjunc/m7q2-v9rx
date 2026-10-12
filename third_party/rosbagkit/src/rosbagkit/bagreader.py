import logging
from collections import Counter
from pathlib import Path

from rosbags.highlevel import AnyReader
from tqdm import tqdm

from rosbagkit.utils.misc import format_bytes, get_path_size

logger = logging.getLogger(__name__)


def read_bagfile(
    bagfile: str,
    topics: list[str],
    start_time: float = 0.0,
    end_time: float = float("inf"),
    early_return: bool = False,
    timestamp_source: str = "header",
) -> dict[str, list[tuple[float, object]]]:  # topic: [(timestamp, message), ...]
    if timestamp_source not in ("header", "bag"):
        raise ValueError(f"timestamp_source must be 'header' or 'bag', got {timestamp_source!r}")

    if not topics:
        tqdm.write("[TOPICS] No topics provided. Returning empty result.")
        return {}

    if end_time < 0:
        end_time = float("inf")

    topics_to_msgs = {topic: [] for topic in topics}
    bagfile_path = Path(bagfile)

    tqdm.write(f"Opening {bagfile_path} ({format_bytes(get_path_size(bagfile))})...")
    with AnyReader([bagfile_path]) as reader:
        connections = [c for c in reader.connections if c.topic in topics]
        available_topics = {c.topic for c in connections}
        missing_topics = set(topics) - available_topics

        if not missing_topics and available_topics:
            tqdm.write(f"[TOPICS] All found: {topics}")
        elif not connections:
            tqdm.write(f"[TOPICS] None found: {topics}")
            tqdm.write(f"[TOPICS] Available: {[c.topic for c in reader.connections]}")
            return topics_to_msgs
        else:
            tqdm.write(f"[TOPICS] Missing: {missing_topics}")
            tqdm.write(f"[TOPICS] Found: {available_topics}")

        total_msgs = sum(max(getattr(connection, "msgcount", 0), 0) for connection in connections) or None
        for connection, timestamp, rawdata in tqdm(
            reader.messages(connections=connections),
            total=total_msgs,
            desc=f"read {bagfile_path.name}",
            unit="msg",
            leave=False,
            dynamic_ncols=True,
        ):
            msg = reader.deserialize(rawdata, connection.msgtype)

            header = getattr(msg, "header", None)
            if timestamp_source == "header" and header and hasattr(header, "stamp"):
                ts_sec = header.stamp.sec + header.stamp.nanosec * 1e-9
            else:
                ts_sec = timestamp * 1e-9

            if start_time <= ts_sec <= end_time:
                topics_to_msgs[connection.topic].append((ts_sec, msg))

            if early_return and all(len(msgs) > 0 for msgs in topics_to_msgs.values()):
                tqdm.write("[EARLY RETURN] All topics have at least one message.")
                break

    for msgs in topics_to_msgs.values():
        msgs.sort(key=lambda x: x[0])

    for topic, msgs in topics_to_msgs.items():
        timestamps = [ts for ts, _ in msgs]
        dupes = [ts for ts, count in Counter(timestamps).items() if count > 1]
        if dupes:
            tqdm.write(f"[DUPLICATES] Topic {topic} has {len(dupes)} duplicate ts (e.g. {dupes[:5]})")

    return topics_to_msgs


def get_topics_from_bagfile(bagfile: str) -> set[str]:
    try:
        with AnyReader([Path(bagfile)]) as reader:
            topics = {c.topic for c in reader.connections}
        logger.info(f"[TOPICS] Found {len(topics)} topics in {bagfile}")
        return topics
    except Exception as e:
        logger.exception(f"[ERROR] Failed to read topics from {bagfile}: {e}")
        return set()


def get_topic_frame_ids(bagfile: str, max_msg_per_topic: int = 1) -> dict[str, str]:
    frame_ids = {}

    try:
        with AnyReader([Path(bagfile)]) as reader:
            for connection in reader.connections:
                topic = connection.topic
                msgtype = connection.msgtype

                count = 0
                for _, _, rawdata in reader.messages(connections=[connection]):
                    msg = reader.deserialize(rawdata, msgtype)

                    header = getattr(msg, "header", None)
                    if header and hasattr(header, "frame_id"):
                        frame_ids[topic] = header.frame_id
                        break

                    count += 1
                    if count >= max_msg_per_topic:
                        break

        logger.info(f"[FRAME_ID] Found frame_id for {len(frame_ids)} topics.")
        return frame_ids

    except Exception as e:
        logger.exception(f"[ERROR] Failed to extract frame_id from bagfile: {e}")
        return {}
