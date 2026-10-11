"""Packet-by-packet demo of the lossy radio link (channel.py):   python network_demo.py [--link clean|default|harsh]

Shows, for every packet the rover sends: was it dropped, bit-corrupted (CRC will reject it), or delivered,
how long it took, and whether it arrived out of order. Saves outputs/network_timeline_<link>.png."""
import argparse, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from crater.config import LinkParams
from crater.pipeline import Mission, evaluate
from crater.channel import expected_loss

LINKS = {"clean": LinkParams(p_gb=0, loss_good=0, ber=0), "default": LinkParams(),
         "harsh": LinkParams(p_gb=0.3, p_bg=0.2, loss_bad=0.9)}
ap = argparse.ArgumentParser(); ap.add_argument("--link", choices=list(LINKS), default="default")
name = ap.parse_args().link; lp = LINKS[name]

m = Mission(link=lp)
log, orig = [], m.link.send
def logged_send(raw, t):
    s0 = dict(m.link.stats); state_bad = None
    orig(raw, t)
    state_bad = m.link.bad
    if m.link.stats["dropped"] > s0["dropped"]:
        fate, arrive = "DROPPED", None
    else:
        arrive = m.link._q[-1][0] if m.link._q else None
        arrive = [q[0] for q in m.link._q if q[1] == m.link._n][0]
        fate = "CORRUPTED" if m.link.stats["corrupted"] > s0["corrupted"] else "delivered"
    log.append(dict(n=len(log) + 1, t=t, size=len(raw), fate=fate, arrive=arrive, bad=state_bad))
m.link.send = logged_send
m.run()

arr = [(p["arrive"], p["n"]) for p in log if p["arrive"] is not None]
order = [n for _, n in sorted(arr)]
ooo = sum(1 for a, b in zip(order, order[1:]) if b < a)
print(f"Link '{name}': expected loss {expected_loss(lp):.0%} (burst model: good<->bad radio state)\n")
print(" #  sent(s)  bytes  radio-state  outcome     delay(s)")
for p in log:
    d = f"{p['arrive'] - p['t']:.2f}" if p["arrive"] is not None else "  - "
    print(f"{p['n']:2d}  {p['t']:6.1f}  {p['size']:5d}  {'BAD ' if p['bad'] else 'good'}         {p['fate']:10s}  {d}")
st, rx = m.link.stats, m.station.stats
ok = rx["rx_packets"]
print(f"\nSent {st['sent']} packets ({st['bytes_sent']} B): dropped {st['dropped']}, bit-corrupted {st['corrupted']} "
      f"(CRC rejected {rx['crc_fail']}), accepted by ground station {ok}, arrived out of order {ooo}")
ev = evaluate(m.station.snapshot(m.n * m.dt, 0.15), m)
print(f"Resulting map: recall {ev['recall']:.2f}, precision {ev['precision']:.2f}")

fig, ax = plt.subplots(figsize=(9, 0.28 * len(log) + 1.6))
col = {"delivered": "tab:green", "CORRUPTED": "tab:orange", "DROPPED": "tab:red"}
for p in log:
    y = -p["n"]
    if p["arrive"] is None:
        ax.plot(p["t"], y, "x", color=col["DROPPED"], ms=8)
    else:
        ax.plot([p["t"], p["arrive"]], [y, y], "-o", color=col[p["fate"]], ms=4)
    ax.text(-0.3, y, "BAD" if p["bad"] else "", color="grey", fontsize=7, ha="right", va="center")
ax.set_yticks([-p["n"] for p in log]); ax.set_yticklabels([f"pkt {p['n']}" for p in log], fontsize=7)
ax.set_xlabel("time (s)  -  line = packet in flight from rover to ground; red x = lost on the way")
ax.set_title(f"Lossy link '{name}': green delivered, orange corrupted (rejected by CRC), red dropped")
fig.tight_layout()
out = pathlib.Path("outputs"); out.mkdir(exist_ok=True)
fig.savefig(out / f"network_timeline_{name}.png", dpi=110)
print(f"Saved outputs/network_timeline_{name}.png")
