"""Uncertainty model: confidence + distance + data age.

(1) Trust score  T = c * exp(-d/d0) * exp(-dt/tau)      in [0,1], dimensionless
(2) Position covariance
        sigma_px_eff = sigma_px / c                        (low confidence -> noisier pixel position)
        Sigma_geom   = sigma_px_eff^2 * J J^T + DEM term   (distance AND terrain slope enter via the ray-cast Jacobian J)
        Sigma(dt)    = Sigma_geom + (kappa*dt)^2 * I        (age enters as isotropic drift)
"""
import numpy as np
from .config import Camera, UncertaintyParams
from .terrain import terrain_covariance


def trust(conf: float, dist_m: float, age_s: float, p: UncertaintyParams = UncertaintyParams()) -> float:
    return float(conf * np.exp(-dist_m / p.d0_m) * np.exp(-age_s / p.tau_s))


def geometric_covariance(cam: Camera, uv, conf: float, p: UncertaintyParams = UncertaintyParams(), terrain=None):
    """Pixel noise (inflated by low confidence) + DEM error, propagated through the terrain ray-cast.
    terrain=None -> flat-ground assumption (no DEM, no slope term)."""
    return terrain_covariance(cam, uv, terrain, p.sigma_px / max(conf, 0.05), p.dem_sigma_m if terrain else 0.0)


def inflate_for_age(cov: np.ndarray, age_s: float, p: UncertaintyParams = UncertaintyParams()):
    return cov + (p.drift_mps * max(age_s, 0.0)) ** 2 * np.eye(2)


def ellipse_params(cov: np.ndarray):
    """Return (sigma_major, sigma_minor, angle_rad of major axis from +X, in [0, pi))."""
    vals, vecs = np.linalg.eigh(cov)
    smin, smaj = np.sqrt(np.maximum(vals, 0))
    vx, vy = vecs[:, 1]
    return float(smaj), float(smin), float(np.arctan2(vy, vx) % np.pi)


def cov_from_ellipse(smaj: float, smin: float, ang: float) -> np.ndarray:
    c, s = np.cos(ang), np.sin(ang)
    R = np.array([[c, -s], [s, c]])
    return R @ np.diag([smaj ** 2, smin ** 2]) @ R.T
