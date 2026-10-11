"""Replay mode: play a PREPARED sequence of rover camera images from disk through the normal pipeline.

Frames are rendered once by the simulator's own camera model (same poses, same scene) and cached as PNGs in
data/rover_sequence/seed<S>_relief<R>/frame_NNN.png.  At demo time every frame is READ FROM DISK, then goes through
the unchanged chain:  detect -> localise -> compact telemetry -> lossy link -> ground station.
(You can open the PNGs in any image viewer, or overwrite them with your own images of the same size.)
"""
import pathlib
import numpy as np
import cv2
from .pipeline import Mission, process_frame
from .synth import render

SEQ_ROOT = pathlib.Path(__file__).resolve().parents[1] / "data" / "rover_sequence"
MAX_FRAMES = 20


def sequence_dir(seed, relief):
    return SEQ_ROOT / f"seed{int(seed)}_relief{relief:.1f}"


def ensure_sequence(m: Mission, seed, relief):
    """Create the PNG sequence once (skipped if it already exists). Returns the folder."""
    d = sequence_dir(seed, relief)
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(seed) + 100)          # same noise stream as the live simulator
    for i in range(MAX_FRAMES):
        path = d / f"frame_{i:03d}.png"
        img = render(m.scene, m.poses[i], rng) if i < len(m.poses) else None
        if img is not None and not path.exists():
            cv2.imwrite(str(path), img)
    return d


class ReplayMission(Mission):
    """Same as Mission, but the camera image of each frame is loaded from the prepared sequence."""

    def __init__(self, n_frames=12, seed=7, relief=1.0, **kw):
        super().__init__(n_frames=n_frames, seed=seed, relief=relief, **kw)
        # poses for all MAX_FRAMES so the cache is identical whatever the slider says
        full = Mission(n_frames=MAX_FRAMES, seed=seed, relief=relief)
        self.seq_dir = ensure_sequence(full, seed, relief)

    def load_frame(self, i):
        img = cv2.imread(str(self.seq_dir / f"frame_{i:03d}.png"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"missing prepared frame {i} in {self.seq_dir}")
        return img

    def step(self, i):
        cam, t = self.poses[i], i * self.dt
        img = self.load_frame(i)                                   # <- the only change vs Mission.step
        packets, info = process_frame(img, cam, i, t, self.est_terrain)
        for pk in packets:
            for _ in range(self.repeat):
                self.link.send(pk, t + 0.2)
        for t_rx, raw in self.link.poll(t + self.dt):
            self.station.ingest(raw, t_rx)
        return dict(i=i, t=t, cam=cam, img=img, info=info, packets=packets, now=t + self.dt)
