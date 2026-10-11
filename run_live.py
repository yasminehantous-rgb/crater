"""LIVE system: images arrive from the 'rover', travel over a real UDP socket through the lossy radio link
(channel.py) to a separate ground-station socket, and the ground station publishes JSON for the dashboard.

      rover (camera frames) --UDP--> radio channel (drop/corrupt/delay) --UDP--> ground station --> live/*.json

  python run_live.py                       # rover watches the folder rover_inbox/ : drop images in, they are processed
  python run_live.py --video outputs/mission.avi --poses outputs/mission_poses.json --seed 7 --interval 1
  python run_live.py --link harsh          # clean | default | harsh

Output for the dashboard (rewritten ~2x per second):
  live/snapshot.json      hazards on the map (position, ellipse, age, trust)
  live/status.json        link + ground-station counters, recent packet fates, FPS, source
  live/latest_frame.png   the rover's latest image with detections drawn on it
"""
import argparse, json, os, pathlib, socket, threading, time
import cv2, numpy as np
from crater.config import Camera, LinkParams
from crater.channel import LossyLink, expected_loss
from crater.groundstation import GroundStation
from crater.pipeline import Mission, process_frame, evaluate
from dashboard.plots import annotate_frame

LINKS = {"clean": LinkParams(p_gb=0, loss_good=0, ber=0), "default": LinkParams(),
         "harsh": LinkParams(p_gb=0.3, p_bg=0.2, loss_bad=0.9)}
EXT = {".png", ".jpg", ".jpeg", ".bmp"}
LIVE = pathlib.Path("live")
T0 = time.time()
now = lambda: time.time() - T0
state = dict(source="", frames=0, last_ms=0.0, fps=0.0, last_detections=0, recent_packets=[], recent_frames=[])
lock = threading.Lock()


def write_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, default=float))
    for _ in range(5):                                   # Windows can briefly lock a file a reader has open
        try:
            os.replace(tmp, path); return
        except PermissionError:
            time.sleep(0.05)


def channel_thread(link, rx_sock, ground_addr, stop):
    """The radio: receives datagrams from the rover, applies loss / corruption / delay, forwards survivors."""
    out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    n = 0
    while not stop.is_set():
        try:
            raw, _ = rx_sock.recvfrom(2048)
        except socket.timeout:
            raw = None
        if raw is not None:
            n += 1; t = now(); s0 = dict(link.stats)
            link.send(raw, t)
            if link.stats["dropped"] > s0["dropped"]:
                fate, delay = "DROPPED", None
            else:
                arrive = [q[0] for q in link._q if q[1] == link._n][0]
                fate, delay = ("CORRUPTED" if link.stats["corrupted"] > s0["corrupted"] else "delivered"), round(arrive - t, 2)
            with lock:
                state["recent_packets"] = (state["recent_packets"] + [dict(n=n, t=round(t, 1), bytes=len(raw), radio_state="bad" if link.bad else "good", fate=fate, delay_s=delay)])[-40:]
        for _, pkt in link.poll(now()):
            out.sendto(pkt, ground_addr)


def ground_thread(station, sock, link, link_name, stop):
    """Mission control receiver: decode (CRC), de-duplicate, fuse, publish JSON."""
    last = 0.0
    while not stop.is_set():
        try:
            raw, _ = sock.recvfrom(2048)
            station.ingest(raw, now())
        except socket.timeout:
            pass
        if time.time() - last > 0.4:
            last = time.time(); t = now()
            write_json(LIVE / "snapshot.json", dict(time_s=round(t, 2), hazards=station.snapshot(t, 0.0)))
            with lock:
                st = dict(state)
            write_json(LIVE / "status.json", dict(time_s=round(t, 2), link=link_name, expected_loss=round(expected_loss(link.p), 3),
                                                  link_stats=dict(link.stats), ground_stats=dict(station.stats), **st))


def folder_source(folder, stop):
    seen = set()
    while not stop.is_set():
        for p in sorted(x for x in folder.iterdir() if x.suffix.lower() in EXT and x not in seen and time.time() - x.stat().st_mtime > 0.3):
            seen.add(p)
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape != (480, 640):
                img = cv2.resize(img, (640, 480))
            side = p.with_suffix(".json")                # optional camera pose next to the image: same name, .json
            cam = Camera(**json.loads(side.read_text())) if side.exists() else Camera()
            yield p.name, img, cam, None
        time.sleep(0.4)


def video_source(video, poses_json, seed, interval, stop):
    poses = json.loads(pathlib.Path(poses_json).read_text())
    terrain = Mission(n_frames=len(poses), seed=seed).est_terrain      # onboard terrain-map prior
    cap = cv2.VideoCapture(video)
    for i, pose in enumerate(poses):
        ok, bgr = cap.read()
        if not ok or stop.is_set():
            break
        yield f"frame {i}", cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), Camera(**pose), terrain
        time.sleep(interval)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--link", choices=list(LINKS), default="default")
    ap.add_argument("--folder", default="rover_inbox")
    ap.add_argument("--video"); ap.add_argument("--poses"); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between video frames")
    ap.add_argument("--port", type=int, default=9101, help="radio channel port; ground station uses port+1")
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()
    LIVE.mkdir(exist_ok=True); pathlib.Path(a.folder).mkdir(exist_ok=True)

    link, station, stop = LossyLink(LINKS[a.link], seed=1), GroundStation(), threading.Event()
    ch_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); ch_sock.bind(("127.0.0.1", a.port)); ch_sock.settimeout(0.05)
    gr_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); gr_sock.bind(("127.0.0.1", a.port + 1)); gr_sock.settimeout(0.1)
    threading.Thread(target=channel_thread, args=(link, ch_sock, ("127.0.0.1", a.port + 1), stop), daemon=True).start()
    threading.Thread(target=ground_thread, args=(station, gr_sock, link, a.link, stop), daemon=True).start()

    video_mode = bool(a.video)
    src = video_source(a.video, a.poses, a.seed, a.interval, stop) if video_mode else folder_source(pathlib.Path(a.folder), stop)
    state["source"] = f"video {a.video}" if video_mode else f"folder {a.folder}/"
    print(f"LIVE system running | link '{a.link}' | rover source: {state['source']}\n"
          f"Rover -> UDP :{a.port} (radio channel) -> UDP :{a.port + 1} (ground station) -> live/*.json\nPress Stop / Ctrl+C to quit.")
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    fid, t_start = 0, time.time()
    try:
        for name, img, cam, terrain in src:
            t0 = time.perf_counter()
            packets, info = process_frame(img, cam, fid, now(), terrain)
            ms = (time.perf_counter() - t0) * 1000
            for pk in packets:
                tx.sendto(pk, ("127.0.0.1", a.port))
            cv2.imwrite(str(LIVE / "latest_frame.png"), cv2.cvtColor(annotate_frame(img, info), cv2.COLOR_RGB2BGR))
            fid += 1
            with lock:
                state.update(frames=fid, last_ms=round(ms, 1), fps=round(fid / max(time.time() - t_start, 1e-6), 2), last_detections=len(info))
                state["recent_frames"] = (state["recent_frames"] + [dict(frame=fid - 1, image=name, detections=len(info), packets=len(packets),
                                                                     bytes=sum(map(len, packets)), edge_ms=round(ms, 1))])[-20:]
            print(f"[rover] {name}: {len(info)} hazards -> {len(packets)} packets, {sum(map(len, packets))} B, {ms:.0f} ms")
            if a.max_frames and fid >= a.max_frames:
                break
    except KeyboardInterrupt:
        pass
    time.sleep(link.p.latency_s + link.p.jitter_s + 1.5)          # let the last packets arrive
    stop.set(); time.sleep(0.3)
    st = link.stats
    print(f"\nLink: sent {st['sent']} pkts ({st['bytes_sent']} B), dropped {st['dropped']}, corrupted {st['corrupted']}; "
          f"ground accepted {station.stats['rx_packets']}, CRC rejects {station.stats['crc_fail']}")
    if video_mode:
        m = Mission(n_frames=fid, seed=a.seed)
        ev = evaluate(station.snapshot(now(), 0.15), m)
        print(f"Map vs simulator truth: recall {ev['recall']:.2f}, precision {ev['precision']:.2f}, xy-RMSE {ev['rmse_m']:.2f} m")
