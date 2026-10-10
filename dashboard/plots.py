"""Plot helpers for the Mission Control dashboard (kept separate so they can be tested headless)."""
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from crater.config import UncertaintyParams

BG, FG = "#0d1117", "#c9d1d9"


def map_figure(snapshot, poses, truth=None, t_now=0.0, p=UncertaintyParams(), terrain=None):
    """Top-down hazard map. Colour = trust, fade = data age, ellipse = 2-sigma position uncertainty."""
    fig, ax = plt.subplots(figsize=(6.4, 6.4), facecolor=BG)
    ax.set_facecolor(BG)
    if terrain is not None and terrain.z is not None:       # orbital-DEM prior: hillshade + elevation contours
        ext = (terrain.x0, terrain.x0 + terrain.z.shape[1] * terrain.res, terrain.y1 - terrain.z.shape[0] * terrain.res, terrain.y1)
        shade = np.clip(0.5 + 2.0 * (terrain.dzdx * 0.8 - terrain.dzdy * 0.6), 0, 1)
        ax.imshow(shade, extent=ext, cmap="gray", alpha=0.35, origin="upper", vmin=0, vmax=1, zorder=0)
        X = terrain.x0 + np.arange(terrain.z.shape[1]) * terrain.res
        Y = terrain.y1 - np.arange(terrain.z.shape[0]) * terrain.res
        cs = ax.contour(X, Y, terrain.z, levels=np.arange(-3, 3, 0.25), colors="#58a6ff", linewidths=0.4, alpha=0.5, zorder=1)
        ax.clabel(cs, fmt="%.2g m", fontsize=5, colors="#58a6ff")
    if truth is not None:                                    # sim-only ground truth (hollow grey marks)
        for h in truth:
            ax.add_patch(Circle((h.x, h.y), h.r, fill=False, ec="#555", lw=0.8, ls=":"))
    ax.plot([c.x for c in poses], [c.y for c in poses], "-", c="#58a6ff", lw=1.5, label="rover path")
    ax.plot(0, 0, "w*", ms=14, label="landing site")
    cmap = plt.get_cmap("RdYlGn")
    for s in snapshot:
        a = 0.3 + 0.7 * np.exp(-s["age"] / p.tau_s)         # old data fades
        col = cmap(np.clip(s["trust"], 0, 1))
        ax.add_patch(Ellipse((s["x"], s["y"]), 4 * s["sig_major"], 4 * s["sig_minor"],
                             angle=np.degrees(s["angle"]), fc=col, ec=col, alpha=0.25 * a + 0.1))
        ax.add_patch(Circle((s["x"], s["y"]), max(s["r"], 0.25), fill=s["cls"] == "rock", fc=col, ec=col, lw=2, alpha=a,
                            ls="-" if s["cls"] == "rock" else "--"))
        ax.annotate(f"#{s['id']}  T={s['trust']:.2f}  z={s['z']:+.1f}m\n{s['age']:.0f}s old", (s["x"], s["y"]), xytext=(6, 6),
                    textcoords="offset points", color=FG, fontsize=6.5, alpha=a)
    ax.set_aspect("equal"); ax.set_xlim(-14, 14); ax.set_ylim(-3, 46)
    ax.set_xlabel("East X (m)", color=FG); ax.set_ylabel("North Y (m)", color=FG)
    ax.set_title(f"Hazard map relative to landing site  |  t = {t_now:.0f}s", color=FG)
    ax.tick_params(colors=FG); ax.grid(alpha=0.15)
    for sp in ax.spines.values():
        sp.set_color("#30363d")
    ax.legend(loc="lower right", facecolor=BG, labelcolor=FG, fontsize=7)
    fig.tight_layout()
    return fig


def annotate_frame(img, info):
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for k in info:
        d = k["det"]; x, y, w, h = d.bbox
        col = (255, 120, 60) if d.cls else (80, 220, 120)
        cv2.rectangle(vis, (x, y), (x + w, y + h), col, 2)
        cv2.putText(vis, f"{'crater' if d.cls else 'rock'} c={d.conf:.2f} {k['dist']:.0f}m", (x, max(12, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    return vis
