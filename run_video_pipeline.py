"""End-to-end: VIDEO FILE -> edge perception -> localisation -> 60-byte packets -> lossy link -> ground map.

    python run_video_pipeline.py                     # records outputs/mission.avi, then ingests it
    python run_video_pipeline.py --link harsh        # worse radio link
    python run_video_pipeline.py --video my.avi --poses my_poses.json --seed 7   # ingest an existing video

Step 1 (record) plays the role of the rover camera: it renders the simulated drive into a video file plus a
small JSON with the camera pose for each frame (a real rover would have this from odometry/IMU).
Step 2 (ingest) knows nothing about the simulator: it reads the video frame by frame, like a live feed.
The only simulator-derived input is the onboard terrain map (the README's stated 'known DEM' assumption)
and the ground truth, which is used ONLY to score the result.
"""
import argparse, json, pathlib, time
import cv2, numpy as np
from dataclasses import asdict
from crater.config import LinkParams
from crater.pipeline import Mission, process_frame, evaluate
from crater.synth import render
from crater.geometry import Camera
from dashboard.plots import map_figure

LINKS = {"clean": LinkParams(p_gb=0, loss_good=0, ber=0), "default": LinkParams(),
         "harsh": LinkParams(p_gb=0.3, p_bg=0.2, loss_bad=0.9)}


def record(video, poses_json, seed, n_frames):
    m = Mission(n_frames=n_frames, seed=seed)
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 2.0, (640, 480))
    poses = []
    for i in range(n_frames):
        img = render(m.scene, m.poses[i], m.rng)
        w.write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        poses.append({k: float(v) for k, v in asdict(m.poses[i]).items()})
    w.release()
    pathlib.Path(poses_json).write_text(json.dumps(poses))
    print(f"[record] wrote {video} ({n_frames} frames) and {poses_json}")


def ingest(video, poses_json, seed, link_name, out):
    poses = json.loads(pathlib.Path(poses_json).read_text())
    m = Mission(n_frames=len(poses), seed=seed, link=LINKS[link_name])   # transport + terrain prior + scoring only
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video {video}")
    ms, n_det, i, last_now = [], 0, 0, 0.0
    while True:
        ok, bgr = cap.read()
        if not ok or i >= len(poses):
            break
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        cam, t = Camera(**poses[i]), i * m.dt
        t0 = time.perf_counter()
        packets, info = process_frame(gray, cam, i, t, m.est_terrain)
        ms.append((time.perf_counter() - t0) * 1000)
        n_det += len(info)
        for pk in packets:
            m.link.send(pk, t + 0.2)
        for t_rx, raw in m.link.poll(t + m.dt):
            m.station.ingest(raw, t_rx)
        last_now, i = t + m.dt, i + 1
    cap.release()
    for t_rx, raw in m.link.poll(1e9):
        m.station.ingest(raw, t_rx)
    snap = m.station.snapshot(len(poses) * m.dt, 0.15)
    out.mkdir(exist_ok=True)
    map_figure(snap, m.poses, m.scene.hazards, last_now, terrain=m.terrain).savefig(out / f"video_map_{link_name}.png", dpi=110)
    ev = evaluate(snap, m)
    print(f"[ingest] {i} frames read | {n_det} detections | edge time {np.mean(ms):.1f} ms/frame ({1000/np.mean(ms):.1f} FPS)")
    print(f"[link]   {link_name}: sent {m.link.stats['sent']} pkts ({m.link.stats['bytes_sent']} B), received {m.station.stats['rx_packets']}")
    print(f"[map]    recall {ev['recall']:.2f} precision {ev['precision']:.2f} xy-RMSE {ev['rmse_m']:.2f} m z-RMSE {ev['rmse_z_m']:.2f} m")
    print(f"[out]    map saved to {out / f'video_map_{link_name}.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="outputs/mission.avi")
    ap.add_argument("--poses", default="outputs/mission_poses.json")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--link", choices=list(LINKS), default="default")
    a = ap.parse_args()
    out = pathlib.Path("outputs"); out.mkdir(exist_ok=True)
    if not pathlib.Path(a.video).exists():
        record(a.video, a.poses, a.seed, a.frames)
    ingest(a.video, a.poses, a.seed, a.link, out)
