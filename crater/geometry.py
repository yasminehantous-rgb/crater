"""Spatial math: pinhole camera, projection matrix, ground homography, covariance propagation.

Camera frame: x right, y down, z forward (right-handed).  World frame: X East, Y North, Z up.
"""
import numpy as np
from .config import Camera


def intrinsics(cam: Camera) -> np.ndarray:
    """K = [[f,0,cx],[0,f,cy],[0,0,1]]"""
    return np.array([[cam.f, 0, cam.width / 2],
                     [0, cam.f, cam.height / 2],
                     [0, 0, 1.0]])


def rotation_world_to_cam(pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Rows of R are the camera axes (right, down, forward) written in world coordinates."""
    th, ps = np.radians(pitch_deg), np.radians(yaw_deg)
    fwd = np.array([np.sin(ps) * np.cos(th), np.cos(ps) * np.cos(th), -np.sin(th)])
    right = np.array([np.cos(ps), -np.sin(ps), 0.0])
    down = np.cross(fwd, right)
    return np.vstack([right, down, fwd])


def projection_matrix(cam: Camera) -> np.ndarray:
    """3x4 matrix P = K [R | t], with t = -R C and C the camera centre in world coordinates."""
    R = rotation_world_to_cam(cam.pitch_deg, cam.yaw_deg)
    C = np.array([cam.x, cam.y, cam.z])
    t = -R @ C
    return intrinsics(cam) @ np.hstack([R, t[:, None]])


def ground_homography(cam: Camera, z_plane=None) -> np.ndarray:
    """Points on the horizontal plane Z = z_plane: (X,Y,Z,1) = (X,Y,z_plane,1), so the Z column of P folds
    into the translation column:   H(z) = [p1  p2  (z*p3 + p4)],   s*(u,v,1)^T = H (X,Y,1)^T.
    Default plane = ground level under the camera (the classic 'flat-ground' assumption)."""
    P = projection_matrix(cam)
    z = cam.base_z if z_plane is None else z_plane
    return np.column_stack([P[:, 0], P[:, 1], z * P[:, 2] + P[:, 3]])


def project(cam: Camera, xyz) -> np.ndarray:
    """Forward model for 3-D world points: (X,Y,Z) -> pixel (u,v) using the full 3x4 matrix P."""
    xyz = np.atleast_2d(xyz).astype(float)
    p = (projection_matrix(cam) @ np.c_[xyz, np.ones(len(xyz))].T).T
    return p[:, :2] / p[:, 2:3]


def project_ground(cam: Camera, xy, z_plane=None) -> np.ndarray:
    """Forward model for points on the plane Z = z_plane (default: ground under the camera)."""
    xy = np.atleast_2d(xy).astype(float)
    z = cam.base_z if z_plane is None else z_plane
    return project(cam, np.c_[xy, np.full(len(xy), z)])


def pixel_to_ground(cam: Camera, uv, z_plane=None):
    """Inverse model (the key step): (X,Y,1)^T ~ H^-1 (u,v,1)^T, then divide by the 3rd component.
    Returns ground coords (N,2); NaN for pixels at/above the horizon (ray never hits the ground)."""
    uv = np.atleast_2d(uv).astype(float)
    M = np.linalg.inv(ground_homography(cam, z_plane))
    w = (M @ np.c_[uv, np.ones(len(uv))].T).T
    out = np.full((len(uv), 2), np.nan)
    ok = w[:, 2] > 1e-12          # w3 = 1/depth, so w3 <= 0 means "behind / above horizon"
    out[ok] = w[ok, :2] / w[ok, 2:3]
    return out


def ground_covariance(cam: Camera, uv, sigma_px: float, z_plane=None) -> np.ndarray:
    """First-order (delta-method) propagation of isotropic pixel noise to the ground:
         g(u,v) = (w1/w3, w2/w3),  w = M (u,v,1)^T,  M = H^-1
         J = dg/d(u,v) = (M[:2,:2] - g * M[2,:2]) / w3
         Sigma_g = sigma_px^2 * J J^T
    The result is strongly anisotropic: range error >> cross-range error far from the camera."""
    M = np.linalg.inv(ground_homography(cam, z_plane))
    w = M @ np.array([uv[0], uv[1], 1.0])
    g = w[:2] / w[2]
    J = (M[:2, :2] - np.outer(g, M[2, :2])) / w[2]
    return sigma_px ** 2 * (J @ J.T)
