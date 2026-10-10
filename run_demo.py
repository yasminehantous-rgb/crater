"""Headless end-to-end demo:  python run_demo.py   (writes outputs/*.png and prints metrics)"""
import pathlib, cv2, numpy as np
from crater.config import LinkParams
from crater.pipeline import Mission, evaluate
from crater.channel import expected_loss
from dashboard.plots import map_figure, annotate_frame

out = pathlib.Path("outputs"); out.mkdir(exist_ok=True)
LINKS = [("clean", LinkParams(p_gb=0, loss_good=0, ber=0)), ("default", LinkParams()),
         ("harsh", LinkParams(p_gb=0.3, p_bg=0.2, loss_bad=0.9))]

print("=== Link robustness (terrain-aware, seed 7) ===")
for name, lp in LINKS:
    m = Mission(link=lp); last = None
    for i in range(m.n):
        last = m.step(i)
        if i == 4 and name == "default":
            cv2.imwrite(str(out / "frame4_detections.png"), cv2.cvtColor(annotate_frame(last["img"], last["info"]), cv2.COLOR_RGB2BGR))
    for t_rx, raw in m.link.poll(1e9):
        m.station.ingest(raw, t_rx)
    snap = m.station.snapshot(last["now"], 0.15)
    map_figure(snap, m.poses, m.scene.hazards, last["now"], terrain=m.terrain).savefig(out / f"map_{name}.png", dpi=110)
    ev = evaluate(snap, m)
    print(f"{name:8s} expected loss {expected_loss(lp):4.0%} | {m.link.stats['sent']} pkts ({m.link.stats['bytes_sent']} B), "
          f"rx {m.station.stats['rx_packets']} | recall {ev['recall']:.2f} precision {ev['precision']:.2f} "
          f"xy-RMSE {ev['rmse_m']:.2f} m z-RMSE {ev['rmse_z_m']:.2f} m")

print("\n=== Ablation: flat-ground assumption vs terrain-aware (clean link, 5 seeds) ===")
for aware in (False, True):
    R = []
    for seed in (7, 11, 21, 33, 42):
        m = Mission(link=LINKS[0][1], seed=seed, terrain_aware=aware); m.run()
        R.append(evaluate(m.station.snapshot(m.n * m.dt, 0.15), m))
    f = lambda k: np.nanmean([r[k] for r in R])
    print(f"{'terrain-aware' if aware else 'flat ground  '} recall {f('recall'):.2f} precision {f('precision'):.2f} "
          f"xy-RMSE {f('rmse_m'):.2f} m z-RMSE {f('rmse_z_m'):.2f} m")
