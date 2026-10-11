"""Edge-perception benchmark:  python benchmark.py   (writes outputs/benchmark_results.txt)
Measures the real per-frame cost of process_frame (detect + localise + pack) on this machine."""
import time, tracemalloc, pathlib, platform, os
import numpy as np
from crater.pipeline import Mission, process_frame
from crater.synth import render

N_FRAMES, WARMUP = 60, 5
m = Mission(n_frames=12, seed=7)
frames = [(render(m.scene, m.poses[i % 12], m.rng), m.poses[i % 12]) for i in range(N_FRAMES)]

for img, cam in frames[:WARMUP]:
    process_frame(img, cam, 0, 0.0, m.est_terrain)

times, pk_bytes = [], []
c0 = time.process_time(); w0 = time.perf_counter()
for i, (img, cam) in enumerate(frames):
    t0 = time.perf_counter()
    pk, _ = process_frame(img, cam, i, i * 2.0, m.est_terrain)
    times.append((time.perf_counter() - t0) * 1000)
    pk_bytes.append(sum(len(p) for p in pk))
wall, cpu = time.perf_counter() - w0, time.process_time() - c0

tracemalloc.start()                      # separate pass: tracemalloc slows code down, so it is not used for timing
for i, (img, cam) in enumerate(frames[:10]):
    process_frame(img, cam, i, i * 2.0, m.est_terrain)
peak = tracemalloc.get_traced_memory()[1] / 1e6
tracemalloc.stop()

t = np.array(times)
lines = [
    f"Machine: {platform.processor() or platform.machine()} | {os.cpu_count()} logical cores | Python {platform.python_version()}",
    f"Frames: {N_FRAMES} (640x480 grayscale), single thread, no GPU",
    f"Mean   {t.mean():6.1f} ms/frame  -> {1000 / t.mean():5.1f} FPS",
    f"Median {np.median(t):6.1f} ms | p95 {np.percentile(t, 95):6.1f} ms | worst {t.max():6.1f} ms",
    f"CPU time / wall time: {cpu / wall:4.2f} (about {cpu / wall:.0%} of one core)",
    f"Peak Python allocations (10-frame pass): {peak:5.1f} MB",
    f"Telemetry per frame: mean {np.mean(pk_bytes):.0f} B vs raw frame {640 * 480} B (~{640 * 480 / max(np.mean(pk_bytes), 1):,.0f}x smaller)",
]
print("\n".join(lines))
out = pathlib.Path("outputs"); out.mkdir(exist_ok=True)
(out / "benchmark_results.txt").write_text("\n".join(lines) + "\n")
