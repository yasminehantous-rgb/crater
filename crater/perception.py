"""Lightweight, fully local hazard detector (classical CV, ~ms per frame, no model weights).

Idea: hazards are local brightness anomalies (shadow + sunlit face, bowl + rim).
  1. band-pass: fine blur minus very coarse blur -> removes illumination gradient
  2. robust threshold (MAD) -> mask,  morphological close + fill -> blobs
  3. each blob: pixel centroid, confidence c in [0.15,1] from contrast/solidity/size
  4. METRIC size via the terrain ray-cast (back-project bbox) -> classify rock vs crater with perspective-aware size
"""
from dataclasses import dataclass
import numpy as np
import cv2
from .config import Camera
from .geometry import project
from .terrain import pixel_to_terrain, Terrain

CRATER_MIN_DIAM_M = 2.2
MIN_DIAM_M = 0.35
THR_FLOOR, THR_MAD = 18.0, 5.0        # detection threshold = max(floor, k * MAD of the band-passed image)


@dataclass
class Detection:
    u: float
    v: float
    conf: float
    radius_m: float
    cls: int
    bbox: tuple


def detect(img: np.ndarray, cam: Camera, terrain=None, max_range_m: float = 18.0, min_area_px: int = 30):
    h, w = img.shape
    ahead = [cam.x + max_range_m * np.sin(np.radians(cam.yaw_deg)), cam.y + max_range_m * np.cos(np.radians(cam.yaw_deg))]
    z_ahead = float(terrain.height(*ahead)) if terrain is not None else cam.base_z
    v0 = int(np.clip(project(cam, [[ahead[0], ahead[1], z_ahead]])[0, 1], 0, h - 1))   # ignore pixels farther than max_range
    f = img.astype(np.float32)
    f[:v0] = np.median(img[v0:])
    d = cv2.GaussianBlur(f, (0, 0), 1.2) - cv2.GaussianBlur(f, (0, 0), 18)
    roi = d[v0:]
    mad = 1.4826 * np.median(np.abs(roi - np.median(roi)))
    thr = max(THR_FLOOR, THR_MAD * mad)
    mask = (np.abs(d) > thr).astype(np.uint8)
    mask[:v0] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area_px:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, [c], -1, 1, -1)
        m = cv2.moments(filled, binaryImage=True)
        u, v = m["m10"] / m["m00"], m["m01"] / m["m00"]
        solidity = area / max(cv2.contourArea(cv2.convexHull(c)), 1.0)
        p95 = np.percentile(np.abs(d[filled > 0]), 95)
        conf = 0.15 + 0.85 * (0.5 * min(1, p95 / 60) + 0.3 * solidity + 0.2 * min(1, area / 80))
        if x <= 1 or x + bw >= w - 1 or y + bh >= h - 1:      # clipped by the image border -> less trust
            conf *= 0.6
        # metric size from the homography (ground width x ground depth of the bbox)
        g = pixel_to_terrain(cam, [[x, v], [x + bw, v], [u, y], [u, y + bh]], terrain)[:, :2]
        if np.isnan(g).any():
            continue
        diam = float(np.sqrt(np.linalg.norm(g[1] - g[0]) * np.linalg.norm(g[3] - g[2])))
        if diam < MIN_DIAM_M:                                   # too small to matter physically
            continue
        out.append(Detection(u, v, float(np.clip(conf, 0.05, 0.99)), diam / 2, int(diam > CRATER_MIN_DIAM_M), (x, y, bw, bh)))
    return _suppress_duplicates(out, cam, terrain)


def _suppress_duplicates(dets, cam, terrain):
    """Metric-space NMS: a rock's shadow or a crater rim can fragment into several blobs. Keep the largest
    blob and drop any smaller one whose ground position lies within 1.5*R + 0.5 m of it (done on the ground
    plane, so the test is perspective-correct)."""
    dets = sorted(dets, key=lambda d: -d.radius_m)
    kept, gk = [], []
    for d in dets:
        g = pixel_to_terrain(cam, [d.u, d.v], terrain)[0, :2]
        if any(np.hypot(*(g - gj)) < 1.5 * k.radius_m + 0.5 for k, gj in zip(kept, gk)):
            continue
        kept.append(d); gk.append(g)
    return kept
