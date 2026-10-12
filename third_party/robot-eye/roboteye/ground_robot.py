# Third Party
import numpy as np
from pyquaternion import Quaternion
import warnings
from enum import Enum

# In House
from roboteye.geom import pi, inv_pi, transform_points_3d

class Frames(Enum):
    """
    The frames are specified as follows:

    WORLD_FRAME:              World coordinate system

    BODY_FRAME_WORLD_ALIGNED: World frame basis vectors but is translated
                              to the position of the robot
    
    BODY_FRAME:               Robot body frame

    CAM_FRAME:                Camera coordinate system (3D)

    IMG_FRAME:                Pixel coordinate system (2D)              
    """
    WORLD_FRAME              = 0
    BODY_FRAME_WORLD_ALIGNED = 1
    BODY_FRAME               = 2
    CAM_FRAME                = 3
    IMG_FRAME                = 4

class COB(Enum):
    """
    Following conversions are supported:

    NED_TO_CAM: NED body frame to X right Y down Z depth

    Args:
        Enum (_type_): _description_

    Returns:
        _type_: _description_
    """
    NED_TO_CAM = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ])

class GroundRobot:
    """
    The robot class is in charge of tracking a single frame of the robot (keyframe)
    This information will be used and altered when the pose estimation is done.
    """
    def __init__(self, cam_calib = {}, rob_q = None, rob_t = None, cob = None):
        """
        Initialize the robot frame.

        \param[in] cam_calib: Intrinsics and extrinisics matrix/matricies for the cameras attached to the robot
        \param[in] rob_q: Quaternion describing the rotation of the robot relative to world. (R_robot^world)
                          This should include change of basis from w to robot and the rotation of the robot
        \param[in] rob_C: Vector with the tail from the origin and the head towards the robot's centroid
        """
        # Default camera matricies, can be one to many cameras
        self.K = {}
        self.E = {}
        self.q = None
        self.t = None

        # Support change of basis for cameras
        if cob is None:
            self.camera_cob = np.eye(4)
        elif cob in COB:
            self.camera_cob = cob.value
        else:
            self.camera_cob = cob

        # Support for multiple cameras but only one robot pose
        if "K" in cam_calib.keys():
            K = np.array(cam_calib["K"])
            if isinstance(K, np.ndarray):
                self.add_camera_intrinsics(K, "0")
        if "E" in cam_calib.keys():
            E = np.array(cam_calib["E"])
            if isinstance(E, np.ndarray):
                self.add_camera_extrinsics(E, "0")
        else:
            self.set_camera_calibration(cam_calib)

        # Set the rotation and translation of the robot
        self.set_robot_extrinsics(rob_q, rob_t)

    @staticmethod
    def init_intrinsics(f_x, f_y, p_x, p_y, s = 0):
        """
        Create the intrinsics matrix from subparamters.

        \param[in] f_x   Focal length (horizontal)
        \param[in] f_y   Focal length (vertical)
        \param[in] p_x   Principal Point (x)
        \param[in] p_y   Principal Point (y)
        \param[in] s     Skew coefficient of the camera
        https://blog.immenselyhappy.com/post/camera-axis-skew/
        """
        
        return np.array([[f_x,  s,   p_x],
                           [0,  f_y, p_y],
                           [0,  0,   1]])

    def set_intrinsics(self, K):
        """
        Directly set the intrinsics for the robot if it has a camera
        """
        self.K = K

    def set_camera_calibration(self, calib):
        """
        Set the calibration for the cameras on the robot.

        \param[in] calib: Calibration dictionary
        """
        for camera_str in calib.keys():
            # Add the intrinsics information
            cam_calib = calib[camera_str]
            self.add_camera_intrinsics(cam_calib["K"], camera_str)
            self.add_camera_extrinsics(cam_calib["E"], camera_str)

    def add_camera_intrinsics(self, K, camera_str):
        """
        Add camera intrinsics calibration information

        \param[in] K_m_to_pix: [3,3] matrix for calibration
        \param[in] camera_str: Which camera to store the intrinsics for
        """
        self.K[camera_str] = K

    def add_camera_extrinsics(self, E, camera_str):
        """
        Add camera extrinsics calibration information.

        \param[in] E:          [4,4] matrix for calibration
        \param[in] camera_str: Which camera to store the extrinsics for
        """
        self.E[camera_str] = E

    def set_robot_extrinsics(self, rot_quat, position):
        """
        Initialize the robot extrinsics from a quaternion and displacement from world's origin.

        \param[in] rot_quat: Rotation of the robot
        \param[in] position: Position of the robot relative to the world origin
        """
        if rot_quat is not None:
            self.q = Quaternion(w=rot_quat[-1], x=rot_quat[0], y = rot_quat[1], z = rot_quat[2])

        if position is not None:
            self.rob_pos = position

    def get_rotation_matrix(self):
        """
        Convert quaternion to the rotation matrix to get robot's orientation in the world
        """
        return self.q.rotation_matrix
    
    def get_K(self, camera = "0"):
        """
        Get the intrinsic matrix for the given camera.

        \param[in] camera: Camera of interest (not needed if only one camera in system).
        """
        K = np.eye(3)
        if self.K is not None:
            if camera is None:
                K = self.K
            else:
                K = self.K[camera]

        else:
            warnings.warn("K is not initialized properly.")
        return K
    
    def get_E(self, camera = "0"):
        """
        Get the calibration extrinsics matrix for the given camera.

        \param[in] camera: Camera of interest (not needed if only one camera in system).
        """
        E = np.eye(4)
        if self.E is not None and len(self.E) > 0:
            if camera is None:
                E = self.E
            else:
                E = self.E[camera]
        else:
            warnings.warn("E is not initialized properly.")
        return E
    
    def get_extrinsics_matrix(self, w_to_rob = True):
        """
        Find extrinsics matrix depending on the parameter passed in
        Treat this as a passive transformation because the coordinate system is moving.

        | ------------------------------|
        | w_to_c | Equation             |
        | ------------------------------|
        | True   | P_c = R(P_w - C)     |
        | False  | P_w = (R.T P_c) + C  |
        |-------------------------------|

        extrinsics = [R | t ]
                     [0 | 1 ]
        Size: [4,4]

        \param[in] w_to_rob: world to camera matrix, if false, will be camera to world extrinsics

        """
        R_rob_to_w = np.eye(4)
        t_rob_to_w = np.eye(4)

        # Set the rotation and translation then find the extrinsics matrix
        R_rob_to_w[0:3, 0:3] = self.get_rotation_matrix()
        t_rob_to_w[0:3, 3]   = -self.rob_pos
        M                    = R_rob_to_w @ t_rob_to_w

        # Invert the matrix if we are going from world to robot
        if w_to_rob is False:
            M = np.linalg.inv(M)

        return M
    
    def get_P(self, in_frame: Frames, out_frame: Frames, camera="0"):
        """
        Get the projection matrix from in frame to out frame for the camera.

        Args:
            in_frame (Frames): The input frame
            out_frame (Frames): The output frame
            camera (str, optional): Which camera in the system to grab. Defaults to "0".
        """
       # Decide how to transform the points
        direction = 2 * int((in_frame.value - out_frame.value) < 0) - 1
        M_t = np.eye(4)
        M_r = np.eye(4)
        E   = np.eye(4)
        K   = np.eye(3)
        
        for i in range(in_frame.value, out_frame.value + direction, direction):
            # These conditions are for if we need camera extrinsics.
            need_cam_extr = (in_frame.value - i) != 0 and \
                            ((i==2 and direction==-1) or (i==3 and direction==1))
            
            # Move to world center
            if i == 0 and direction == -1 or i == 1 and direction == 1:
                M_t[:3, -1] = -self.rob_pos
            # Aligning points that were w.r.t. robot
            if i == 1 and direction == -1 or i == 2 and direction == 1:
                M_r = self.q.transformation_matrix
            # Translate to camera rot, translation
            elif need_cam_extr:
                E = self.get_E(camera)
            # Use cam intrinsics
            elif i == 4:
                K = self.get_K(camera)

        # Change of basis:
        camera_cob = self.camera_cob

        # Composing rotation and translation of robot
        M = M_r @ M_t

        # We are going towards the world
        if direction == -1:
            M = np.linalg.inv(M)
            E = np.linalg.inv(E)
            K = np.linalg.inv(K)
            camera_cob = np.linalg.inv(camera_cob)

        # In this case, we will need the change of basis
        if (in_frame == Frames.IMG_FRAME or \
           out_frame == Frames.IMG_FRAME or \
           in_frame == Frames.CAM_FRAME or \
           out_frame == Frames.CAM_FRAME) and \
           abs(in_frame.value - out_frame.value) > 1:
            E = camera_cob @ E

        return K, E, M


    def transform_points(self, points, in_frame: Frames, out_frame: Frames, camera="0"):
        """
        Transform points in the camera frame to points in the world frame.

        \param[in] points: 2D or 3D homogenous points, Shape: [n, 4] or [n, 3]
        \param[in] in_frame:  What frame to transfrom the points from.
        \param[in] out_frame: What frame to transform the points to.
        \param[in] camera: String specifying which camera to use for the transform.

        Returns: World points of shape [n, 4]
        """
        # NOTE: Yeah, this has gotten a bit hacky over time. I have a TODO in the future to clean up this logic.

        # Input Checking
        if points.shape[1] != 3 and points.shape[0] == 3 and in_frame == Frames.IMG_FRAME or \
           points.shape[1] != 4 and points.shape[0] == 4 and in_frame != Frames.IMG_FRAME:
            points = points.T

        # Trivial case
        if in_frame == out_frame:
            return points

        # Get the parts of the P matrix (intriniscs and extrinsics)
        K, T = self.get_P(in_frame, out_frame, camera)

        if in_frame == Frames.IMG_FRAME:
            depths  = points[:, 2].reshape(points.shape[0], 1)
            locs    = (points[:, :3] / depths)[:, :2]
            out_pts = inv_pi(K, T, locs, depths.flatten())
        elif out_frame == Frames.IMG_FRAME:
            img_w = K[0, 2] * 2
            img_h = K[1, 2] * 2
            out_pts = pi(K, T, points, img_h, img_w)

        else:
            out_pts = transform_points_3d(T, points)

        # Special Case: Aligning basis vectors to world frame but in the body frame
        # NOTE: This really doesn't have a use case
        # if out_frame == Frames.BODY_FRAME_WORLD_ALIGNED:
        #     # From body frame to world frame
        #     out_pts = self.q.inverse.transformation_matrix @ out_pts

        return out_pts
    
    # def unproject_points(self, points, depth, camera = "0", transform="world"):
    #     """
    #     Project each pixel into a 3D point given the pixel coordinates and the depth

    #     \param[in] points  Homogoenous points:     Shape [n,3]
    #     \param[in] depth   Depths for each points: Shape [n,]
    #     """
    #     # Input handling
    #     if points.shape[1] == 2:
    #         points = np.concatenate((points, np.ones((points.shape[0], 1))), 1)
    #     if len(depth.shape) == 2:
    #         depth = depth.reshape(len(depth))

    #     # Go from pixel space to metric space
    #     K             = np.array(self.K[camera])
    #     camera_points = (np.linalg.inv(K) @ points.T) * depth

    #     # Create homogenous 3D points
    #     if camera_points.shape[0] == 3:
    #         camera_points = np.concatenate((camera_points, np.ones((1, camera_points.shape[1]))))

    #     return self.transform_points(camera_points, camera, transform)
