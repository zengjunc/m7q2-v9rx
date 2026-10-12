import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def read_image_msg(msg) -> np.ndarray:
    """
    Convert a sensor_msgs/Image or sensor_msgs/CompressedImage to a NumPy array,
    trusting the 'encoding' field for color channel ordering.

    Args:
        msg (sensor_msgs.Image or sensor_msgs.CompressedImage): ROS image message

    Returns:
        np.ndarray: Image in RGB or grayscale format
    """
    if hasattr(msg, "format") and "compressed" in msg.format.lower():
        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

        if image is None:
            raise ValueError("Failed to decode CompressedImage message.")

        if len(image.shape) == 3:
            if image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            elif image.shape[2] == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError(f"Unsupported channel count: {image.shape[2]}")
        else:
            return image

    # sensor_msgs/Image (uncompressed)
    encoding = getattr(msg, "encoding", "").lower()
    np_arr = np.frombuffer(msg.data, dtype=np.uint8)

    if encoding in ("mono8", "8uc1"):
        image = np_arr.reshape((msg.height, msg.width))
    elif encoding == "rgb8":
        image = np_arr.reshape((msg.height, msg.width, 3))
    elif encoding == "bgr8":
        image = np_arr.reshape((msg.height, msg.width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif encoding == "rgba8":
        image = np_arr.reshape((msg.height, msg.width, 4))
    elif encoding == "bgra8":
        image = np_arr.reshape((msg.height, msg.width, 4))
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    else:
        raise NotImplementedError(f"Unsupported image encoding: {encoding}")

    return image


def save_image(img: np.ndarray, filename: str) -> None:
    if len(img.shape) == 3:
        if img.shape[2] == 3:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif img.shape[2] == 4:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")
    else:
        img_bgr = img  # mono8 or mono16

    return cv2.imwrite(filename, img_bgr)
