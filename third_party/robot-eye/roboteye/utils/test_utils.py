# Third Party
import numpy as np
from pyquaternion import Quaternion

def create_K(f, img_size):
    return np.array([[f, 0, (img_size[1]-1)/2],
                     [0, f, (img_size[0]-1)/2],
                     [0, 0, 1]])

def create_extrinsics(cam_dx, cam_dy, cam_dz, cam_placement):
    E = np.eye(4)   
    if cam_placement == "left":
        E[:3, -1] = np.array([-cam_dx, 0, cam_dz])
    elif cam_placement == "right":
        E[:3, -1] = np.array([cam_dx, 0, cam_dz])
    # This should be the inverse of the camera locations to transform pts
    return np.linalg.inv(E)

def get_calib(img_size, cam_dx, cam_dy, cam_dz, q_c = Quaternion(1, 0, 0, 0), f = 1, cam_placement = "middle"):
    """
    Create camera calibration for testing purposes.

    \param[in] img_size:      (H,W) of size of image plane (pix)
    \param[in] q_c:           Quaternion describing the rotation of the camera (including change of basis)
    \param[in] cam_dx:        X displacement of camera from world origin (m)
    \param[in] cam_dy:        Y displacement of camera from world origin (m)
    \param[in] cam_dz:        Z displacement of camera from world origin (m)
    \param[in] f:             Focal length used for scaling meters to pixels
    \param[in] cam_placement: Where the camera is relative to the ego frame

    \return: Calibration dictionary for a robot with as single camera.
    """
    calib = {}
    calib["K"] = create_K(f, img_size)
    cam_rot = np.eye(4)
    cam_rot[:3, :3] = q_c.rotation_matrix
    calib["E"] =  cam_rot @ create_extrinsics(cam_dx, cam_dy, cam_dz, cam_placement)
    return calib

def get_calib_K(K, cam_dx, cam_dy, cam_dz, cam_placement="middle"):
    calib     = {}
    calib["K"] = K
    calib["E"] = create_extrinsics(cam_dx, cam_dy, cam_dz)
    return calib

def setup_camera_fov(hfov, wid, hgt, sens_width=13.2):
    """
    Setup a camera that is able to see a certain number of meters up down left and right.

    \param[in] hfov: Horizontal field of view (degrees)
    \param[in] wd: Working distance (mm)
    \param[in] sens_width: Width of the sensor (mm)
    """
    hfov_rad = hfov * np.pi/180
    f = sens_width / (2 * np.tan(hfov_rad/2))
    K = np.array([[f, 0, (wid-1/2)],
                  [0, f, (hgt-1)/2],
                  [0, 0, 1]])
    return K

def create_bbox_pts(obj_bboxes, obj_rots):
    final_pts = []
    for obj_num in range(obj_bboxes.shape[0]):
        # Compute the verticies then rotate them
        obj_bbox = obj_bboxes[obj_num]
        obj_rot  = obj_rots[obj_num]
        xc, yc, zc, dx, dy, dz = obj_bbox
        bounding_pts = []
        for x in [xc-dx/2, xc+dx/2]:
            for y in [yc-dy/2, yc+dy/2]:
                for z in [zc-dz/2, zc+dz/2]:
                    x_r, y_r, z_r = obj_rot.rotate([x, y, z])
                    bounding_pts.append([x_r, y_r, z_r, 1])
        final_pts.append(bounding_pts)
    return np.array(final_pts)

def create_stereo_camera_setup(yaw = 0):
    # Camera 1 transformation matrix
    cam1_loc = np.array([0., 1.5, 0.])
    cam1_rot = Quaternion(angle = yaw, axis = [0, 0, 1])
    T_cam_1  = np.eye(4)
    T_cam_1[0:3, 0:3] = cam1_rot.rotation_matrix
    T_cam_1[0:3, -1]  = -cam1_rot.rotation_matrix@cam1_loc

    # Camera 2 transformation matrix
    cam2_loc = np.array([0., -1.5, 0.])
    cam2_rot = Quaternion(angle = -yaw, axis = [0, 0, 1])
    T_cam_2  = np.eye(4)
    T_cam_2[0:3, 0:3] = cam2_rot.rotation_matrix
    T_cam_2[0:3, -1]  = -cam2_rot.rotation_matrix@cam2_loc
    
    # Create robot frame with two affine cameras
    calib = {}
    calib["0"] = {}
    calib["1"] = {}
    calib["0"]["K"] = np.array([[200, 0, 256],
                                [0, 200, 256],
                                [0, 0, 1]])
    calib["0"]["E"] = T_cam_1
    calib["1"]["K"] = np.array([[200, 0, 256],
                                [0, 200, 256],
                                [0, 0, 1]])
    calib["1"]["E"] = T_cam_2
    return calib

def interpolate_vud(vud_pts, img_wid=512, img_hgt=256, method = "weighted"):
    """
    Raster the v,u points into an image, interpolating the depth between the points.
    We will essentially do an inverse distance weighting for the missing pixels.

    Args:
        vud_pts (np.ndarray): (3, N) [V (x), U (y), depth]
        img_wid (int, optional): _description_. Defaults to IMG_WID.
        img_hgt (_int, optional): _description_. Defaults to IMG_HGT.
    """
    # Instaniate image
    img = np.zeros((img_hgt, img_wid))

    # Bounds of the image
    if vud_pts.shape[1] > 0:
        min_x = np.clip(np.min(vud_pts[0]), 0, img_wid-1).astype("int")
        max_x = np.clip(np.max(vud_pts[0]), 0, img_wid-1).astype("int")
        min_y = np.clip(np.min(vud_pts[1]), 0, img_hgt-1).astype("int")
        max_y = np.clip(np.max(vud_pts[1]), 0, img_hgt-1).astype("int")

        # Interpolate based on the number of points
        for x in range(min_x, max_x+1):
            for y in range(min_y, max_y+1):
                if method == "weighted":
                    # We can weight by distance from all 8 points:
                    dist  = np.linalg.norm(np.array([x, y]).reshape(2, 1) - vud_pts[:2], axis = 0)
                    power = 1
                    wgts  = 1/(dist**power)
                    wgts /= np.sum(wgts)
                    img[y,x] = np.sum(wgts*vud_pts[2])
                elif method == "constant":
                    dist  = np.linalg.norm(np.array([x, y]).reshape(2, 1) - vud_pts[:2], axis = 0)
                    img[y,x] = vud_pts[2, np.argmin(dist)]

    return img
