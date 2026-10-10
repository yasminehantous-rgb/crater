"""Terrain-aware localisation: intersect the pixel's viewing RAY with a heightfield z = h(X, Y).

  ray     p(lam) = C + lam * d,     d = R^T K^-1 (u,v,1)^T          (lam = depth along the optical axis)
  solve   F(lam) = C_z + lam*d_z - h(C_x + lam*d_x, C_y + lam*d_y) = 0     (first root = visible surface)
  method  coarse marching until F changes sign (handles occlusion), then bisection.
  (A flat terrain h = const reduces this exactly to the homography H(z) = [p1 p2 z*p3+p4].)

  Uncertainty by implicit differentiation:   F_lam = d_z - grad(h).d_xy
      d(lam)/dq = -lam*(dd_z/dq - grad(h).dd_xy/dq) / F_lam ,     dg/dq = d_xy*d(lam)/dq + lam*dd_xy/dq
      Sigma = sigma_px^2 J J^T + (sigma_dem / F_lam)^2 d_xy d_xy^T
"""
import numpy as np
import cv2
from .config import Camera
from .geometry import rotation_world_to_cam


class Terrain:
    """Regular-grid DEM. Row 0 is the northern edge (y = y1); res in metres per cell."""

    def __init__(self, z, x0, y1, res, flat_level=None):
        self.flat_level = flat_level
        self.x0, self.y1, self.res = x0, y1, res
        self.z = None if z is None else z.astype(np.float32)
        if self.z is not None:
            gy, gx = np.gradient(self.z)
            self.dzdx, self.dzdy = (gx / res).astype(np.float32), (-gy / res).astype(np.float32)

    @classmethod
    def flat(cls, level: float = 0.0) -> "Terrain":
        return cls(None, 0, 0, 1, flat_level=level)

    def _sample(self, grid, x, y):
        x = np.asarray(x, np.float32); y = np.asarray(y, np.float32)
        shape, W = x.shape, 512
        mx, my = ((x - self.x0) / self.res).ravel(), ((self.y1 - y) / self.res).ravel()
        n, pad = mx.size, (-mx.size) % W
        mx = np.pad(mx, (0, pad)).reshape(-1, W).astype(np.float32)
        my = np.pad(my, (0, pad)).reshape(-1, W).astype(np.float32)
        out = cv2.remap(grid, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return out.ravel()[:n].reshape(shape)

    def height(self, x, y):
        if self.flat_level is not None:
            return np.full(np.shape(x), self.flat_level, np.float32)
        return self._sample(self.z, x, y)

    def gradient(self, x, y):
        if self.flat_level is not None:
            return np.zeros(np.shape(x), np.float32), np.zeros(np.shape(x), np.float32)
        return self._sample(self.dzdx, x, y), self._sample(self.dzdy, x, y)

    def slope_deg(self):
        return np.degrees(np.arctan(np.hypot(self.dzdx, self.dzdy)))


def ray_directions(cam: Camera, uv) -> np.ndarray:
    """(N,3) world-frame ray directions, scaled so the camera-z component is 1 (so lam = depth)."""
    uv = np.atleast_2d(uv).astype(float)
    xc = np.c_[(uv[:, 0] - cam.width / 2) / cam.f, (uv[:, 1] - cam.height / 2) / cam.f, np.ones(len(uv))]
    return xc @ rotation_world_to_cam(cam.pitch_deg, cam.yaw_deg)       # = (R^T xc)^T


def pixel_to_terrain(cam: Camera, uv, terrain=None, lam_min=0.3, lam_max=45.0, step=0.5, iters=14):
    """Pixels -> 3-D terrain points (N,3), NaN where the ray misses within lam_max.
    terrain=None means the flat-ground assumption (plane at the camera's ground level)."""
    terrain = terrain or Terrain.flat(cam.base_z)
    d = ray_directions(cam, uv)
    C = np.array([cam.x, cam.y, cam.z])
    N = len(d)

    def F(idx, lam):
        p = C + lam[:, None] * d[idx]
        return p[:, 2] - terrain.height(p[:, 0], p[:, 1])

    all_idx = np.arange(N)
    lam_prev = np.full(N, lam_min)
    f_prev = F(all_idx, lam_prev)
    active = all_idx[f_prev > 0]                       # rays that start above the terrain
    lo, hi = np.full(N, np.nan), np.full(N, np.nan)
    for lam_k in np.arange(lam_min + step, lam_max, step):
        if active.size == 0:
            break
        fk = F(active, np.full(active.size, lam_k))
        hit = fk <= 0
        hit_idx = active[hit]
        lo[hit_idx], hi[hit_idx] = lam_k - step, lam_k
        active = active[~hit]
    ok = ~np.isnan(lo)
    idx, a, b = all_idx[ok], lo[ok], hi[ok]
    for _ in range(iters):                              # bisection refinement
        m = 0.5 * (a + b)
        above = F(idx, m) > 0
        a, b = np.where(above, m, a), np.where(above, b, m)
    out = np.full((N, 3), np.nan)
    out[idx] = C + (0.5 * (a + b))[:, None] * d[idx]
    return out


def terrain_covariance(cam: Camera, uv, terrain, sigma_px: float, sigma_dem: float = 0.0) -> np.ndarray:
    """2x2 ground covariance of the terrain hit point (see module docstring)."""
    terrain = terrain or Terrain.flat(cam.base_z)
    uv = np.asarray(uv, float)
    P = pixel_to_terrain(cam, uv)[0] if terrain.flat_level is not None else pixel_to_terrain(cam, uv, terrain)[0]
    R = rotation_world_to_cam(cam.pitch_deg, cam.yaw_deg)
    d = ray_directions(cam, uv)[0]
    lam = (P[2] - cam.z) / d[2]
    gx, gy = (float(a) for a in terrain.gradient(P[0], P[1]))
    grad = np.array([gx, gy])
    F_lam = d[2] - grad @ d[:2]
    cols = []
    for k in range(2):                                   # q = u, then q = v
        dd = R[k] / cam.f                                # d(direction)/dq
        dlam = -lam * (dd[2] - grad @ dd[:2]) / F_lam
        cols.append(d[:2] * dlam + lam * dd[:2])
    J = np.column_stack(cols)
    return sigma_px ** 2 * (J @ J.T) + (sigma_dem / F_lam) ** 2 * np.outer(d[:2], d[:2])
