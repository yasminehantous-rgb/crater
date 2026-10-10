"""Mission-control side: decode, de-duplicate, associate and FUSE hazard tracks, report age-aware uncertainty.

Fusion = information filter (inverse-covariance weighting) of two Gaussian estimates of the same static hazard:
    S = (S1^-1 + S2^-1)^-1,   x = S (S1^-1 x1 + S2^-1 x2)
Older evidence is first inflated by drift (kappa*dt)^2 I.  Trust of repeated sightings combines by noisy-OR.
"""
from dataclasses import dataclass
import numpy as np
from .config import UncertaintyParams
from .telemetry import decode_packet
from .uncertainty import cov_from_ellipse, inflate_for_age, ellipse_params, trust


@dataclass
class Track:
    id: int; cls: int; pos: np.ndarray; cov: np.ndarray; r: float
    trust0: float; t_cap: float; n_obs: int = 1; z: float = 0.0


class GroundStation:
    def __init__(self, params: UncertaintyParams = UncertaintyParams()):
        self.p, self.tracks, self._seen, self._next = params, [], set(), 1
        self.frames = {}                                  # frame -> (set of idx received, total)
        self.stats = dict(rx_packets=0, crc_fail=0, duplicates=0)

    def ingest(self, raw: bytes, t_rx: float):
        pk = decode_packet(raw)
        if pk is None:
            self.stats["crc_fail"] += 1
            return
        if (pk.frame, pk.idx) in self._seen:
            self.stats["duplicates"] += 1
            return
        self._seen.add((pk.frame, pk.idx))
        self.stats["rx_packets"] += 1
        got, _ = self.frames.setdefault(pk.frame, (set(), pk.total))
        got.add(pk.idx)
        for h in pk.hazards:
            self._associate(h, pk.t_s)

    def _associate(self, h, t_cap):
        pos = np.array([h.x, h.y])
        if h.cls == 0 and any(t.cls == 1 and np.linalg.norm(t.pos - pos) < 1.1 * t.r + 0.3 for t in self.tracks):
            return                                        # containment rule: a 'rock' inside a crater is a rim fragment
        cov = cov_from_ellipse(h.sig_maj, h.sig_min, h.ang) + 1e-4 * np.eye(2)
        best, bd = None, 1e9
        for t in self.tracks:
            d = np.linalg.norm(t.pos - pos)
            gate = 1.0 + h.r + 2 * (h.sig_maj + np.sqrt(np.linalg.eigvalsh(t.cov)[-1]))
            if t.cls == h.cls and d < gate and d < bd:
                best, bd = t, d
        if best is None:
            self.tracks.append(Track(self._next, h.cls, pos, cov, h.r, h.trust0, t_cap, 1, h.z))
            self._next += 1
            self._absorb_fragments(self.tracks[-1])
            return
        t = best
        dt = abs(t_cap - t.t_cap)                         # inflate whichever evidence is older
        c_old, c_new = (inflate_for_age(t.cov, dt, self.p), cov) if t_cap >= t.t_cap else (t.cov, inflate_for_age(cov, dt, self.p))
        I1, I2 = np.linalg.inv(c_old), np.linalg.inv(c_new)
        t.cov = np.linalg.inv(I1 + I2)
        t.pos = t.cov @ (I1 @ t.pos + I2 @ pos)
        t.r = 0.5 * (t.r + h.r)
        t.trust0 = 1 - (1 - t.trust0) * (1 - h.trust0)    # noisy-OR
        t.t_cap = max(t.t_cap, t_cap)
        t.z = (t.z * t.n_obs + h.z) / (t.n_obs + 1)       # running mean elevation
        t.n_obs += 1
        self._absorb_fragments(t)

    def _absorb_fragments(self, t):
        if t.cls == 1:                                    # drop earlier rock tracks that lie inside this crater
            self.tracks = [o for o in self.tracks if o is t or o.cls == 1 or np.linalg.norm(o.pos - t.pos) >= 1.1 * t.r + 0.3]

    def snapshot(self, t_now: float, min_trust: float = 0.0):
        """What the dashboard shows at mission time t_now: position, 1-sigma ellipse, data age, trust."""
        out = []
        for t in self.tracks:
            age = max(t_now - t.t_cap, 0.0)
            smaj, smin, ang = ellipse_params(inflate_for_age(t.cov, age, self.p))
            out.append(dict(id=t.id, cls="crater" if t.cls else "rock", x=t.pos[0], y=t.pos[1], z=t.z, r=t.r,
                            sig_major=smaj, sig_minor=smin, angle=ang, age=age, n_obs=t.n_obs,
                            trust=trust(t.trust0, 0.0, age, self.p)))
        return [o for o in out if o["trust"] >= min_trust]
