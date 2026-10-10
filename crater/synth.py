"""Synthetic lunar scene generator: terrain (DEM) + hazards + ray-cast rendering, with exact ground truth.

1. DEM: smooth undulations + a broad hill; every crater is carved into it (bowl + raised rim).
2. Texture map (top-down): regolith grain + hillshade from the DEM + painted rocks/craters with low-sun shadows.
3. Rendering: every pixel's ray is intersected with the DEM (same solver the pipeline uses), the hit (X,Y)
   samples the texture.  (Simplification: rocks are flat discs lying on the terrain.)
"""
from dataclasses import dataclass
import numpy as np
import cv2
from .config import Camera
from .terrain import Terrain, pixel_to_terrain

S = 0.05                                  # texture resolution (m / px)
TR = 0.25                                 # DEM resolution (m / cell)
X0, X1, Y0, Y1 = -30.0, 30.0, -5.0, 60.0  # map extent (m)
SHADOW_DIR = np.array([0.8, -0.6])        # direction shadows are cast (low sun from the NW)


@dataclass
class GTHazard:
    x: float
    y: float
    r: float
    cls: int      # 0 = rock, 1 = crater
    z: float = 0.0


@dataclass
class Scene:
    ground: np.ndarray
    hazards: list
    terrain: Terrain


def _noise(rng, shape, sigma):
    n = cv2.GaussianBlur(rng.normal(size=shape).astype(np.float32), (0, 0), sigma)
    return n / (n.std() + 1e-9)


def make_terrain(rng, hazards, relief=1.0) -> Terrain:
    h, w = int((Y1 - Y0) / TR) + 1, int((X1 - X0) / TR)
    X = X0 + np.arange(w)[None, :] * TR + np.zeros((h, 1))
    Y = Y1 - np.arange(h)[:, None] * TR + np.zeros((1, w))
    z = 0.40 * _noise(rng, (h, w), 20) + 0.12 * _noise(rng, (h, w), 8)
    z += 0.8 * np.exp(-((X - 4) ** 2 + (Y - 26) ** 2) / (2 * 6.0 ** 2))          # a broad hill
    z = relief * z                                                                # relief scales the background only
    for hz in hazards:
        if hz.cls == 1:                                                           # craters: bowl + rim
            rho = np.hypot(X - hz.x, Y - hz.y)
            z += -0.25 * hz.r * np.exp(-(rho / (0.6 * hz.r)) ** 2) + 0.03 * hz.r * np.exp(-((rho - hz.r) / (0.25 * hz.r)) ** 2)
    z -= z[int((Y1 - 0) / TR), int((0 - X0) / TR)]                                # landing site = datum
    return Terrain(z, X0, Y1, TR)


def make_scene(seed=7, n_rocks=18, n_craters=6, relief=1.0) -> Scene:
    rng = np.random.default_rng(seed)
    hazards = []
    for cls in [1] * n_craters + [0] * n_rocks:
        for _ in range(500):
            r = rng.uniform(1.2, 2.4) if cls else rng.uniform(0.25, 0.65)
            x, y = rng.uniform(-8, 8), rng.uniform(4, 38)
            if all(np.hypot(x - h.x, y - h.y) > r + h.r + 3.0 for h in hazards):
                hazards.append(GTHazard(x, y, r, cls))
                break
    terrain = make_terrain(rng, hazards, relief)
    for h in hazards:
        h.z = float(terrain.height(h.x, h.y))
    H, W = int((Y1 - Y0) / S), int((X1 - X0) / S)
    ground = 118 + 9 * _noise(rng, (H, W), 25) + 5 * _noise(rng, (H, W), 3)
    # hillshade: brightness ~ grad(h) . shadow_dir  (slopes rising away from the sun are lit)
    shade = cv2.resize(terrain.dzdx * SHADOW_DIR[0] + terrain.dzdy * SHADOW_DIR[1], (W, H), interpolation=cv2.INTER_LINEAR)
    ground = ground + 120 * shade
    layer = np.zeros((H, W), np.float32)
    px = lambda x, y: (int((x - X0) / S), int((Y1 - y) / S))
    for h in hazards:
        c, rp = px(h.x, h.y), h.r / S
        sx, sy = SHADOW_DIR[0], -SHADOW_DIR[1]            # image-space shadow direction (y flipped)
        if h.cls == 0:   # rock: dark shadow, then bright body on top
            sc = (int(c[0] + sx * 1.1 * rp), int(c[1] + sy * 1.1 * rp))
            cv2.ellipse(layer, sc, (int(1.5 * rp), int(0.9 * rp)), np.degrees(np.arctan2(sy, sx)), 0, 360, -60, -1)
            cv2.circle(layer, c, int(rp), 50, -1)
        else:            # crater: dark bowl, darker wall in shadow, bright rim
            cv2.circle(layer, c, int(rp), -28, -1)
            cv2.circle(layer, (int(c[0] + sx * 0.25 * rp), int(c[1] + sy * 0.25 * rp)), int(0.8 * rp), -58, -1)
            cv2.circle(layer, c, int(rp), 30, max(3, int(0.12 * rp)))
    layer = cv2.GaussianBlur(layer, (0, 0), 2.0)
    return Scene((ground + layer).astype(np.float32), hazards, terrain)


def render(scene: Scene, cam: Camera, rng: np.random.Generator, scale: int = 2) -> np.ndarray:
    """Ray-cast the DEM (at 1/scale resolution, then upsample the hit map) and sample the texture."""
    w, h = cam.width // scale, cam.height // scale
    u = np.arange(w) * scale + (scale - 1) / 2
    v = np.arange(h) * scale + (scale - 1) / 2
    uu, vv = np.meshgrid(u, v)
    P = pixel_to_terrain(cam, np.c_[uu.ravel(), vv.ravel()], scene.terrain, lam_max=60.0).reshape(h, w, 3)
    X = np.where(np.isnan(P[..., 0]), 0.0, P[..., 0]).astype(np.float32)
    Y = np.where(np.isnan(P[..., 1]), Y1, P[..., 1]).astype(np.float32)
    X, Y = cv2.resize(X, (cam.width, cam.height)), cv2.resize(Y, (cam.width, cam.height))
    img = cv2.remap(scene.ground, (X - X0) / S, (Y1 - Y) / S, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    img = img * rng.uniform(0.92, 1.08) + rng.normal(0, 3.0, img.shape)
    img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 0.8)
    return np.clip(img, 0, 255).astype(np.uint8)
