import cv2
import numpy as np
from tqdm import tqdm


def read_depth_msg(msg) -> np.ndarray:
    """
    Convert a depth image message to a 2D NumPy array in float32 meters.

    Args:
        msg (sensor_msgs.Image or sensor_msgs.CompressedImage)

    Returns:
        np.ndarray: Depth image in meters (float32)
    """
    if msg.data is None or len(msg.data) == 0:
        return None

    if hasattr(msg, "format") and "compressed" in msg.format.lower():
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Failed to decode compressed depth image.")

        if img.dtype == np.uint16:
            return img.astype(np.float32) / 1000.0  # mm -> m
        elif img.dtype == np.float32:
            return img
        else:
            raise ValueError(f"Unsupported compressed depth dtype: {img.dtype}")

    encoding = getattr(msg, "encoding", "").lower()

    if encoding == "16uc1" or encoding == "mono16":
        img = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width))
        return img.astype(np.float32) / 1000.0  # mm -> m
    elif encoding == "32fc1":
        img = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))
        return img
    else:
        raise ValueError(f"Unsupported depth encoding: {msg.encoding}")


def read_pointcloud_depth_msg(msg, width=None, height=None, P=None, **kwargs) -> np.ndarray:
    """Extract a depth image (in meters) from a PointCloud2 message."""
    # Check if the cloud is ordered (Height> 1 implies 2D structure)
    if msg.height <= 1:
        if width is None or height is None or P is None:
            tqdm.write("[WARN] Unordered cloud detected but missing intrinsics (width/height/P)!")
            return None
        return project_unordered_cloud(msg, width, height, P)

    try:
        # We assume standard float32 (4 bytes) and rely on msg.point_step
        raw_data = np.frombuffer(msg.data, dtype=np.uint8)
        reshaped = raw_data.reshape(msg.height, msg.width, msg.point_step)
        z_offset = next(f.offset for f in msg.fields if f.name == "z")
        z_bytes = reshaped[:, :, z_offset : z_offset + 4]
        # Cast to float32 (meters) and Copy to ensure contiguous memory for OpenCV
        depth_img = z_bytes.view(dtype=np.float32).reshape(msg.height, msg.width).copy()
        return depth_img
    except Exception as e:
        tqdm.write(f"[WARN] Failed to extract depth from cloud: {e}")
        return None


def project_unordered_cloud(msg, width, height, P) -> np.ndarray:
    """Project unordered PointCloud2 to depth with projection matrix P."""
    try:
        # Parse Data
        raw_data = np.frombuffer(msg.data, dtype=np.uint8)

        n_points = msg.width
        points_data = raw_data.reshape(n_points, msg.point_step)

        x_off = next(f.offset for f in msg.fields if f.name == "x")
        y_off = next(f.offset for f in msg.fields if f.name == "y")
        z_off = next(f.offset for f in msg.fields if f.name == "z")
        x = points_data[:, x_off : x_off + 4].copy().view(dtype=np.float32).flatten()
        y = points_data[:, y_off : y_off + 4].copy().view(dtype=np.float32).flatten()
        z = points_data[:, z_off : z_off + 4].copy().view(dtype=np.float32).flatten()

        # Filter: Valid depth only (Z > 0)
        valid_mask = (z > 0) & np.isfinite(z)
        x, y, z = x[valid_mask], y[valid_mask], z[valid_mask]

        if x.size == 0:
            return np.zeros((height, width), dtype=np.float32)

        points_hom = np.vstack((x, y, z, np.ones_like(x)))

        p_mat = np.array(P, dtype=np.float32).reshape(3, 4)
        uvw = p_mat @ points_hom

        w_prime = uvw[2, :]
        np.maximum(w_prime, 1e-6, out=w_prime)
        u = (uvw[0, :] / w_prime).round().astype(int)
        v = (uvw[1, :] / w_prime).round().astype(int)

        valid_uv = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        u, v, z = u[valid_uv], v[valid_uv], z[valid_uv]

        sort_idx = np.argsort(z)[::-1]
        u, v, z = u[sort_idx], v[sort_idx], z[sort_idx]

        depth_img = np.zeros((height, width), dtype=np.float32)
        depth_img[v, u] = z
        return depth_img

    except Exception as e:
        tqdm.write(f"[WARN] Homogeneous projection failed: {e}")
        return None


def save_depth(depth: np.ndarray, filename: str) -> None:
    """
    Save a float32 depth image (in meters) as a 16-bit PNG in millimeters.

    Args:
        depth (np.ndarray): Depth image in meters (float32)
        filename (str): Output PNG path
    """
    if depth.dtype != np.float32:
        raise TypeError("Input depth image must be float32 (meters).")

    # Convert meters to millimeters
    depth_mm = (depth * 1000.0).round().astype(np.uint16)

    return cv2.imwrite(filename, depth_mm)
