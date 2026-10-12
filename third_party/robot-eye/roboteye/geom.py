# Third Party
import numpy as np
from scipy.ndimage.interpolation import map_coordinates

# In House
# from src.utils.ground_robot import GroundRobot

# This python module is to be used for geometric CV functions and classes

def pc_to_depth(points, K, T_sens_to_cam, sz):
    """
    Create a depth map from a radar/lidar point cloud.

    \param[in] points:        3D points in the ego frame
    \param[in] K:             3x3 Intrinsic matrix to convert 3D points into points projected on the image plane
    \param[in] T_ego_to_cam:  4x4 Transformation matrix from the ego frame to the camera frame
    \param[in] sz:            Size of the depth map

    \return out_pix_loc: The pixel locations of the perspective projected 3D points
    \return out_depth:   The associated depths at the pixel locations for the points
    """
    # Project into the camera frame then project on to the image plane
    points_xyz = points[:3, :]
    points_xyz_homog = np.concatenate((points_xyz, np.ones((1, points_xyz.shape[1]))))
    points_cam = T_sens_to_cam @ points_xyz_homog
    points_cam /= points_cam[-1, :]
    points_cam = points_cam[0:3, :]

    # At this point, we have the 3D points in the camera coordinate system. Filter out negative depths
    points_mask = np.ones((points_cam.shape[1])).astype('bool')
    points_mask = np.bitwise_and(points_mask, points_cam[-1, :] > 0)

    # Now find the associated pixel locations for each and filter out points not on the image plane
    pixel_locs = K @ points_cam
    pixel_locs /= pixel_locs[-1, :]

    # Filter pixels with negative depth and are not in the image plane
    valid_pix = (pixel_locs[0, :] >= 0) & (pixel_locs[0, :] < sz[0]) & (pixel_locs[1, :] >= 0) & (pixel_locs[1, :] < sz[1])
    points_mask = np.bitwise_and(points_mask, valid_pix)
    
    # Calculate the output locations and associated depths that pass the filter
    out_pix_loc = pixel_locs[:2, points_mask]
    out_depth   = points_cam[-1, points_mask]
    return out_pix_loc, out_depth

def scale_intrinsics(K, sx, sy):
    fx, fy, x0, y0 = split_intrinsics(K)
    fx = fx*sx
    fy = fy*sy
    x0 = x0*sx
    y0 = y0*sy
    K = merge_intrinsics(fx, fy, x0, y0)
    return K

def split_intrinsics(K):
    # K is B x 3 x 3 or B x 4 x 4
    fx = K[:,0,0]
    fy = K[:,1,1]
    x0 = K[:,0,2]
    y0 = K[:,1,2]
    return fx, fy, x0, y0

def merge_intrinsics(fx, fy, x0, y0):
    B = list(fx.shape)[0]
    # K = torch.zeros(B, 4, 4, dtype=torch.float32, device=fx.device)
    K = np.zeros((B, 4, 4), dtype = np.float32)
    K[:,0,0] = fx
    K[:,1,1] = fy
    K[:,0,2] = x0
    K[:,1,2] = y0
    K[:,2,2] = 1.0
    K[:,3,3] = 1.0
    return K

def inv_pi(K_to_met, E_to_rob, locs, depths):
    """
    Implementation of perspective projection from image plane to robot frame.
    Take the robot pose and the location of points and project them into 3D in the robot frame.

    \param[in] K_to_met:    Intrinsic matrix going from pixels to metric.
    \param[in] E_to_rob:    Extrinsic matrix going from anything to a target body/world frame
    \param[in] locs:        Locations of points with registered depth in the frame
    \param[in] depth:       Associated depths in the image frame
    """
    if locs.shape[0] == 2 and locs.shape[1] != 2:
        locs = locs.T

    # Hand calculate which voxels are hit
    pix_locs_homog  = np.concatenate((locs, np.ones((locs.shape[0],1))), axis = 1)

    # This is the step by step of going from the camera locations to the world locations
    cam_points         = (K_to_met @ pix_locs_homog.T) * depths
    cam_points_homog   = np.concatenate((cam_points, np.ones((1,cam_points.shape[1]))), axis = 0)
    rob_points         = E_to_rob @ cam_points_homog
    return rob_points

def in_img_frame(uv_pts, depths, img_wid, img_hgt):
    # See if it's in the image plane
    mask = np.ones((uv_pts.shape[1])).astype("bool")
    if img_hgt is not None and img_wid is not None:
        mask = mask & (uv_pts[0] >= 0) & (uv_pts[0] < img_wid)
        mask = mask & (uv_pts[1] >= 0) & (uv_pts[1] < img_hgt)
    
    # See if the depth is positive:
    if depths is not None:
        mask = mask & (depths > 0) 

    return mask   

def pi(K, E, X, im_hgt = None, im_wid = None, ret_depth = True):
    """
    Implementation of perspective projection of 3D points onto the image plane.
    We keep this pretty ambigious to be more general.
    E can define extrinsics from robot to the camera, world to the camera, etc.
    We will only return points that are in the image plane.

    \param[in] K_to_pix:  Intrinsics to convert pixels to meters, Shape: [3, 3]
    \param[in] E_to_cam:  Extrinsics camera, Shape: [4, 4]
    \param[in] X:         3D points, Shape: [4, N]
    \param[in] im_hgt:    Height of the image in pixels
    \param[in] im_wid:    Width of the iamge in pixels
    """
    # Use extrinsics to have the 3D points relative to the camera of interest
    cam_pts  = transform_points_3d(E, X)
    cam_pts  = (cam_pts / cam_pts[-1])[:3]

    # Project on to image plane (pi)
    uv_pts  = K @ cam_pts
    uv_pts /= uv_pts[-1, :]

    # See if it's in the image plane
    depths = cam_pts[2]
    mask   = in_img_frame(uv_pts, depths, im_wid, im_hgt)
    
    # Concatenate depth if desired
    ret_pts = uv_pts[:-1]
    if ret_depth:
        ret_pts = np.vstack((ret_pts, depths.reshape(1, len(depths))))
    return ret_pts, mask

def transform_points_3d(E, X):
    # Use extrinsics to have the 3D points relative to the camera/body of interest
    if X.shape[1] == 4 and X.shape[0] != 4:
        X = X.T
    tra_pts  = E @ X
    tra_pts /= tra_pts[-1, :]
    return tra_pts

def scroll_3d_volume(grid_res, volumes : list, t: np.ndarray, order: int = 0):
    """
    Translate (scroll) the 3D volume and interpolate the values.

    \param[in] grid_res Voxel dimensions of the grid, Shape: [3,]
    \param[in] volumes  List of 3D/4D volumes
    \param[in] t:       Translation vectorm Shape: [3,]
    \param[in] order:   The order of spline interpolation (0-5)
    """
    # Calculate the new coordinates then sample with interpolation
    x = np.arange(grid_res[0])
    y = np.arange(grid_res[1])
    z = np.arange(grid_res[2])
    xv, yv, zv = np.meshgrid(x, y, z)

    # Warp (translation only)
    xv = xv.astype("float") + t[0]
    yv = yv.astype("float") + t[1]
    zv = zv.astype("float") + t[2]
    new_coords = [xv, yv, zv]

    out = []
    for volume in volumes:
        new_vol = np.zeros_like(volume)

        # Either do it for each feature in the vector if 4D or just do it once if 3D
        if len(volume.shape) > len(new_coords):
            for feat in range(volume.shape[-1]):
                new_vol[:, :, :, feat] = np.swapaxes(map_coordinates(volume[:, :, :, feat], new_coords, order=order), 0, 1)
        else:
            new_vol = np.swapaxes(map_coordinates(volume, new_coords, order=order), 0, 1)
        out.append(new_vol)

    return out

def bilinear_interpolate_torch(im, x, y):
    """
    Bilinear interpolation implementation for an image.
    Source: https://gist.github.com/peteflorence/a1da2c759ca1ac2b74af9a83f69ce20e

    \param[in] im: Image to sample
    \param[in] x:  X location
    \param[in] y:  Y location
    """
    import torch
    if torch.get_device(im) == -1:
        dtype      = torch.float
        dtype_long = torch.long
    else:
        dtype      = torch.cuda.FloatTensor
        dtype_long = torch.cuda.LongTensor

    x0 = torch.floor(x).type(dtype_long)
    x1 = x0 + 1
    
    y0 = torch.floor(y).type(dtype_long)
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1]-1)
    x1 = torch.clamp(x1, 0, im.shape[1]-1)
    y0 = torch.clamp(y0, 0, im.shape[0]-1)
    y1 = torch.clamp(y1, 0, im.shape[0]-1)
    
    # Grab surrounding elements
    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]
    
    wa = (x1.type(dtype)-x) * (y1.type(dtype)-y)
    wb = (x1.type(dtype)-x) * (y-y0.type(dtype))
    wc = (x-x0.type(dtype)) * (y1.type(dtype)-y)
    wd = (x-x0.type(dtype)) * (y-y0.type(dtype))

    return torch.t((torch.t(Ia)*wa)) + torch.t(torch.t(Ib)*wb) + torch.t(torch.t(Ic)*wc) + torch.t(torch.t(Id)*wd)