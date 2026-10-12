from pathlib import Path


def get_path_size(path: str) -> int:
    path = Path(path)
    if path.is_file():  # ROS1
        return path.stat().st_size
    elif path.is_dir():  # ROS2
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return 0


def format_bytes(size_in_bytes: int) -> str:
    if size_in_bytes >= 1024**3:
        return f"{size_in_bytes / 1024**3:.2f} GB"
    elif size_in_bytes >= 1024**2:
        return f"{size_in_bytes / 1024**2:.2f} MB"
    elif size_in_bytes >= 1024:
        return f"{size_in_bytes / 1024:.2f} KB"
    else:
        return f"{size_in_bytes} bytes"
