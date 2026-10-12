from pathlib import Path

import cv2
import numpy as np

from rosbagkit.camera.utils import load_camera_params


class Undistorter:
    def __init__(self, intrinsic: np.ndarray, distortion_coeffs: np.ndarray, image_size: tuple[int, int]):
        self.image_size = tuple(int(v) for v in image_size)
        self.image_shape = (self.image_size[1], self.image_size[0])
        self.map_x, self.map_y = cv2.initUndistortRectifyMap(
            intrinsic, distortion_coeffs, np.eye(3), intrinsic, self.image_size, cv2.CV_32FC1
        )

    def undistort(self, image: np.ndarray) -> np.ndarray:
        if image.shape[:2] != self.image_shape:
            raise ValueError(
                "Image shape does not match undistortion calibration size: "
                f"expected={self.image_shape} got={image.shape[:2]}"
            )
        return cv2.remap(image, self.map_x, self.map_y, cv2.INTER_LINEAR)


def build_undistorter(calib: str | Path) -> Undistorter:
    cam = load_camera_params(calib)
    image_size = tuple(int(v) for v in cam["img_size"][::-1])
    return Undistorter(cam["K"], cam["D"], image_size)
