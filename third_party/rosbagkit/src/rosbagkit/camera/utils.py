from pathlib import Path

import numpy as np
import yaml


def load_extrinsics(extrinsic_file: str | Path) -> np.ndarray:
    with open(extrinsic_file) as f:
        params = yaml.safe_load(f)["extrinsic_matrix"]

    if "R" in params and "T" in params:
        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3, :3] = np.array(params["R"]["data"]).reshape(
            params["rows"], params["cols"]
        )
        extrinsic_matrix[:3, 3] = np.array(params["T"])
        return extrinsic_matrix

    return np.array(params["data"]).reshape(params["rows"], params["cols"])


def load_camera_params(intrinsic_file: str | Path) -> dict[str, np.ndarray]:
    with open(intrinsic_file) as f:
        params = yaml.safe_load(f)

    cam_params: dict[str, np.ndarray] = {}
    if "image_width" in params and "image_height" in params:
        cam_params["img_size"] = np.array([params["image_height"], params["image_width"]])
    elif "width" in params and "height" in params:
        cam_params["img_size"] = np.array([params["height"], params["width"]])
    else:
        raise KeyError("Missing image size in calibration file")

    if "camera_matrix" in params:
        cam_params["K"] = np.array(params["camera_matrix"]["data"]).reshape(
            params["camera_matrix"]["rows"],
            params["camera_matrix"]["cols"],
        )
    else:
        raise KeyError("Missing camera_matrix in calibration file")

    if "distortion_coefficients" in params:
        cam_params["D"] = np.array(params["distortion_coefficients"]["data"])
    else:
        raise KeyError("Missing distortion_coefficients in calibration file")

    return cam_params
