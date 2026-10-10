"""End-to-end pipeline: Perception -> Spatial math -> Compact telemetry -> Lossy transport -> Ground station."""
import numpy as np
from .config import Camera, UncertaintyParams, LinkParams
from .synth import make_scene, render
from .perception import detect
from .geometry import project
from .terrain import pixel_to_terrain
from .uncertainty import geometric_covariance, ellipse_params, trust
from .telemetry import HazardMsg, encode_frame
from .channel import LossyLink
from .groundstation import GroundStation


def process_frame(img, cam, frame_id, t_s, terrain=None, up=UncertaintyParams()):
    """EDGE NODE: image -> telemetry packets.  terrain=None -> flat-ground assumption; else DEM ray-casting."""
    hz, info = [], []
    for d in detect(img, cam, terrain):
        P = pixel_to_terrain(cam, [d.u, d.v], terrain)[0]
        if np.isnan(P).any():
            continue
        dist = float(np.hypot(P[0] - cam.x, P[1] - cam.y))
        smaj, smin, ang = ellipse_params(geometric_covariance(cam, [d.u, d.v], d.conf, up, terrain))
        hz.append((HazardMsg(P[0], P[1], d.radius_m, smaj, smin, ang, d.cls, trust(d.conf, dist, 0.0, up), P[2]), dist))
        info.append(dict(det=d, ground=P, dist=dist))
    return encode_frame(frame_id, t_s, hz), info


class Mission:
    """A rover drives past a hazard field; every frame is processed on the 'edge' and sent over the bad link."""

    def __init__(self, n_frames=12, seed=7, link=LinkParams(), dt_s=2.0, repeat=1, base_cam=Camera(),
                 terrain_aware=True, relief=1.0):
        self.n, self.dt, self.repeat, self.terrain_aware = n_frames, dt_s, repeat, terrain_aware
        self.scene = make_scene(seed, relief=relief)
        self.terrain = self.scene.terrain                       # truth, used by the renderer
        self.est_terrain = self.terrain if terrain_aware else None   # what the edge node believes
        self.rng = np.random.default_rng(seed + 100)
        self.link, self.station = LossyLink(link, seed + 1), GroundStation()
        self.poses = []
        for i in range(n_frames):
            x, y = 1.5 * np.sin(0.4 * i), 2.2 * i
            self.poses.append(base_cam.at(x, y, 6 * np.sin(0.4 * i + 1), float(self.terrain.height(x, y))))

    def step(self, i):
        cam, t = self.poses[i], i * self.dt
        img = render(self.scene, cam, self.rng)
        packets, info = process_frame(img, cam, i, t, self.est_terrain)
        for pk in packets:
            for _ in range(self.repeat):                  # optional redundancy: k copies of each packet
                self.link.send(pk, t + 0.2)
        for t_rx, raw in self.link.poll(t + self.dt):
            self.station.ingest(raw, t_rx)
        return dict(i=i, t=t, cam=cam, img=img, info=info, packets=packets, now=t + self.dt)

    def run(self):
        for i in range(self.n):
            self.step(i)
        for t_rx, raw in self.link.poll(1e9):             # flush packets still in flight
            self.station.ingest(raw, t_rx)
        return self.station.snapshot(self.n * self.dt)

    def observable(self, margin=20, rmin=3.0, rmax=15.0):
        """Ground-truth hazards that were VISIBLE (in image, 3-15 m, not hidden behind a rise) in >= 1 frame.
        Visibility uses the true terrain ray-cast: the pixel's ray must land within 1 m of the hazard."""
        seen = []
        for h in self.scene.hazards:
            for cam in self.poses:
                u, v = project(cam, [[h.x, h.y, h.z]])[0]
                d = np.hypot(h.x - cam.x, h.y - cam.y)
                if margin < u < cam.width - margin and margin < v < cam.height - margin and rmin < d < rmax:
                    hit = pixel_to_terrain(cam, [u, v], self.terrain)[0]
                    if not np.isnan(hit).any() and np.hypot(hit[0] - h.x, hit[1] - h.y) < 1.0:
                        seen.append(h); break
        return seen


def evaluate(snapshot, mission, match_m=1.5):
    gt = mission.observable()
    pos = np.array([[s["x"], s["y"]] for s in snapshot]) if snapshot else np.zeros((0, 2))
    errs, zerrs, hit = [], [], 0
    for h in gt:
        if len(pos):
            d = np.hypot(pos[:, 0] - h.x, pos[:, 1] - h.y); k = int(d.argmin())
            if d[k] < match_m:
                hit += 1; errs.append(d[k]); zerrs.append(snapshot[k]["z"] - h.z)
    prec = (sum(1 for p in pos if min(np.hypot(p[0] - h.x, p[1] - h.y) for h in mission.scene.hazards) < match_m) / len(pos)) if len(pos) else 0.0
    rms = lambda e: float(np.sqrt(np.mean(np.square(e)))) if e else float("nan")
    return dict(n_gt=len(gt), n_tracks=len(pos), recall=hit / max(len(gt), 1), precision=prec,
                rmse_m=rms(errs), rmse_z_m=rms(zerrs))
