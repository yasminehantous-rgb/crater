"""Compact binary telemetry. Every packet is self-contained (own header + CRC-16), so any packet that
survives the lossy link is useful on its own.

Packet  = header(8) + n * hazard(10) + crc16(2)     -> max 5 hazards => 60 bytes
Header  : <H frame_id> <I t_capture_ms> <B n_hazards> <B (idx<<4)|total_packets>
Hazard  : <h x_dm> <h y_dm> <b z_quarter_m> <B radius_dm> <B sig_major(0.05m)> <B sig_minor(0.05m)> <B angle(pi/256)>
          <B (class<<5)|trust(0..31)>
"""
import struct
import binascii
from dataclasses import dataclass, field
import numpy as np

HDR = struct.Struct("<HIBB")
HAZ = struct.Struct("<hhbBBBBB")
MAX_PER_PACKET = 5


@dataclass
class HazardMsg:
    x: float; y: float; r: float
    sig_maj: float; sig_min: float; ang: float
    cls: int; trust0: float
    z: float = 0.0


@dataclass
class Packet:
    frame: int; idx: int; total: int; t_s: float
    hazards: list = field(default_factory=list)


def _q(v, scale, lo, hi):
    return int(np.clip(round(v / scale), lo, hi))


def encode_frame(frame_id: int, t_s: float, hazards, max_per_packet: int = MAX_PER_PACKET):
    """Closest hazards first (they matter most to a lander). An empty frame still sends a 10-byte heartbeat."""
    hazards = sorted(hazards, key=lambda h: h[1])           # (HazardMsg, dist) tuples
    chunks = [hazards[i:i + max_per_packet] for i in range(0, len(hazards), max_per_packet)] or [[]]
    total = len(chunks)
    pkts = []
    for idx, ch in enumerate(chunks):
        body = HDR.pack(frame_id & 0xFFFF, int(t_s * 1000) & 0xFFFFFFFF, len(ch), (idx << 4) | total)
        for h, _ in ch:
            body += HAZ.pack(_q(h.x, 0.1, -32768, 32767), _q(h.y, 0.1, -32768, 32767), _q(h.z, 0.25, -128, 127), _q(h.r, 0.1, 0, 255),
                             _q(h.sig_maj, 0.05, 0, 255), _q(h.sig_min, 0.05, 0, 255),
                             _q(h.ang, np.pi / 256, 0, 255), (h.cls << 5) | _q(h.trust0, 1 / 31, 0, 31))
        pkts.append(body + struct.pack("<H", binascii.crc_hqx(body, 0xFFFF)))
    return pkts


def decode_packet(raw: bytes):
    """Return Packet, or None if length/CRC is wrong (corrupted packets are dropped, never trusted)."""
    if len(raw) < HDR.size + 2:
        return None
    body, crc = raw[:-2], struct.unpack("<H", raw[-2:])[0]
    if binascii.crc_hqx(body, 0xFFFF) != crc:
        return None
    frame, t_ms, n, it = HDR.unpack_from(body)
    if len(body) != HDR.size + n * HAZ.size:
        return None
    pk = Packet(frame, it >> 4, it & 0xF, t_ms / 1000)
    for i in range(n):
        x, y, z, r, a, b, ang, fl = HAZ.unpack_from(body, HDR.size + i * HAZ.size)
        pk.hazards.append(HazardMsg(x / 10, y / 10, r / 10, a * 0.05, b * 0.05, ang * np.pi / 256, fl >> 5, (fl & 31) / 31, z * 0.25))
    return pk
