"""Lossy radio link: Gilbert-Elliott two-state burst-loss model + latency/jitter (=> reordering) + bit errors.

  State G (good) --p_gb--> B (bad) --p_bg--> G.   Loss prob: loss_good in G, loss_bad in B.
  Stationary:  pi_B = p_gb / (p_gb + p_bg),   mean loss = (1-pi_B)*loss_good + pi_B*loss_bad
"""
import heapq
import numpy as np
from .config import LinkParams


def expected_loss(p: LinkParams) -> float:
    pi_b = p.p_gb / (p.p_gb + p.p_bg)
    return (1 - pi_b) * p.loss_good + pi_b * p.loss_bad


class LossyLink:
    def __init__(self, params: LinkParams = LinkParams(), seed: int = 0):
        self.p, self.rng, self.bad = params, np.random.default_rng(seed), False
        self._q, self._n = [], 0
        self.stats = dict(sent=0, dropped=0, corrupted=0, bytes_sent=0)

    def send(self, raw: bytes, t_s: float):
        p = self.p
        self.stats["sent"] += 1
        self.stats["bytes_sent"] += len(raw)
        self.bad = self.rng.random() < ((1 - p.p_bg) if self.bad else p.p_gb)     # Markov step
        if self.rng.random() < (p.loss_bad if self.bad else p.loss_good):
            self.stats["dropped"] += 1
            return
        n_err = self.rng.binomial(len(raw) * 8, p.ber)
        if n_err:
            b = bytearray(raw)
            for pos in self.rng.integers(0, len(raw) * 8, n_err):
                b[pos // 8] ^= 1 << int(pos % 8)
            raw = bytes(b)
            self.stats["corrupted"] += 1
        arrive = t_s + p.latency_s + self.rng.uniform(0, p.jitter_s)
        self._n += 1
        heapq.heappush(self._q, (arrive, self._n, raw))

    def poll(self, t_s: float):
        """All packets that have arrived by time t_s, in arrival order: [(t_arrive, bytes)]."""
        out = []
        while self._q and self._q[0][0] <= t_s:
            a, _, raw = heapq.heappop(self._q)
            out.append((a, raw))
        return out
