"""Automatic tests:  pytest -q"""
import numpy as np
from crater.config import Camera, LinkParams
from crater.geometry import (project_ground, pixel_to_ground, ground_covariance, projection_matrix, ground_homography)
from crater.telemetry import HazardMsg, encode_frame, decode_packet
from crater.channel import LossyLink, expected_loss
from crater.uncertainty import trust, ellipse_params, cov_from_ellipse
from crater.terrain import Terrain, pixel_to_terrain, terrain_covariance
from crater.geometry import project
from crater.pipeline import Mission, evaluate


def test_principal_ray_hits_h_over_tan_pitch():
    cam = Camera()
    g = pixel_to_ground(cam, [[cam.width / 2, cam.height / 2]])[0]
    assert abs(g[0]) < 1e-9 and abs(g[1] - cam.height_m / np.tan(np.radians(cam.pitch_deg))) < 1e-9


def test_round_trip_with_pose():
    cam = Camera().at(3.0, 7.0, 25.0)
    pts = np.array([[2.0, 12.0], [5.0, 15.0], [-1.0, 20.0]])
    assert np.allclose(pixel_to_ground(cam, project_ground(cam, pts)), pts, atol=1e-8)


def test_homography_is_P_without_Z_column():
    cam = Camera().at(1, 2, 10)
    X = np.array([4.0, 9.0, 0.0, 1.0])
    a = projection_matrix(cam) @ X
    b = ground_homography(cam) @ np.array([4.0, 9.0, 1.0])
    assert np.allclose(a, b)


def test_covariance_matches_finite_differences_and_grows_with_range():
    cam = Camera()
    uv = np.array([300.0, 150.0]); eps = 1e-3
    J = np.column_stack([(pixel_to_ground(cam, uv + [eps, 0])[0] - pixel_to_ground(cam, uv - [eps, 0])[0]) / (2 * eps),
                         (pixel_to_ground(cam, uv + [0, eps])[0] - pixel_to_ground(cam, uv - [0, eps])[0]) / (2 * eps)])
    assert np.allclose(ground_covariance(cam, uv, 1.5), 1.5 ** 2 * J @ J.T, rtol=1e-4)
    near, far = (np.trace(ground_covariance(cam, [320, v], 1.5)) for v in (400, 60))
    assert far > 10 * near


def test_trust_monotonic():
    assert trust(0.9, 5, 0) > trust(0.5, 5, 0) > trust(0.5, 15, 0) > trust(0.5, 15, 60)


def test_ellipse_roundtrip():
    c = cov_from_ellipse(2.0, 0.5, 1.0)
    a, b, ang = ellipse_params(c)
    assert np.allclose([a, b, ang], [2.0, 0.5, 1.0], atol=1e-9)


def test_telemetry_roundtrip_size_and_crc():
    hz = [(HazardMsg(12.34, -5.6, 0.8, 0.4, 0.1, 1.2, i % 2, 0.7, -1.3), 5 + i) for i in range(9)]
    pkts = encode_frame(42, 12.5, hz)
    assert len(pkts) == 2 and all(len(p) <= 64 for p in pkts)    # 5 + 4 hazards
    d = decode_packet(pkts[0])
    assert d.frame == 42 and d.total == 2 and abs(d.t_s - 12.5) < 1e-3
    assert abs(d.hazards[0].x - 12.3) < 0.06 and abs(d.hazards[0].y + 5.6) < 0.06
    assert abs(d.hazards[0].z + 1.25) < 0.13
    bad = bytearray(pkts[0]); bad[10] ^= 0x04
    assert decode_packet(bytes(bad)) is None
    assert decode_packet(encode_frame(1, 0, [])[0]).hazards == []          # heartbeat


def test_channel_loss_matches_theory():
    p = LinkParams(ber=0)
    link = LossyLink(p, seed=3)
    for i in range(20000):
        link.send(b"x" * 64, i * 0.1)
    assert abs(link.stats["dropped"] / 20000 - expected_loss(p)) < 0.02


def _hill(slope_y=0.1, wave=0.3):
    res, x0, y1 = 0.25, -30.0, 60.0
    ny, nx = int(65 / res), int(60 / res)
    Y = y1 - np.arange(ny)[:, None] * res + np.zeros((1, nx))
    X = x0 + np.arange(nx)[None, :] * res + np.zeros((ny, 1))
    return Terrain(slope_y * Y + wave * np.sin(X / 4), x0, y1, res)


def test_flat_terrain_reduces_to_homography():
    cam = Camera().at(3, 2, 20, base_z=0.5)
    uv = np.array([[300., 200.], [100., 300.], [500., 120.]])
    assert np.allclose(pixel_to_terrain(cam, uv, None)[:, :2], pixel_to_ground(cam, uv), atol=1e-3)
    assert np.allclose(terrain_covariance(cam, uv[0], None, 1.5), ground_covariance(cam, uv[0], 1.5), atol=1e-6)


def test_terrain_raycast_reprojects_and_lies_on_surface():
    cam, T = Camera().at(3, 2, 20, base_z=0.5), _hill()
    uv = np.array([[300., 200.], [100., 300.], [500., 120.]])
    P = pixel_to_terrain(cam, uv, T)
    assert np.allclose(project(cam, P), uv, atol=1e-6)                                     # lies on the viewing ray
    assert np.allclose(P[:, 2], T.height(P[:, 0], P[:, 1]), atol=1e-3)                    # lies on the terrain


def test_terrain_covariance_matches_finite_differences():
    cam, T = Camera().at(3, 2, 20, base_z=0.5), _hill()
    uv, eps = np.array([300., 200.]), 1e-2
    f = lambda q: pixel_to_terrain(cam, q, T)[0, :2]
    J = np.column_stack([(f(uv + [eps, 0]) - f(uv - [eps, 0])) / (2 * eps), (f(uv + [0, eps]) - f(uv - [0, eps])) / (2 * eps)])
    assert np.allclose(terrain_covariance(cam, uv, T, 1.5), 1.5 ** 2 * J @ J.T, rtol=0.2)   # DEM is bilinear -> approx


def test_slope_changes_range_uncertainty():
    """Ground rising toward the camera shortens the ray (smaller range error); falling away lengthens it."""
    cam, uv = Camera(), [320., 150.]
    var = lambda T: terrain_covariance(cam, uv, T, 1.5)[1, 1]
    assert var(_hill(0.15, 0)) < var(_hill(0.0, 0)) < var(_hill(-0.15, 0))


def test_flat_assumption_is_biased_on_sloped_ground():
    cam, T = Camera(), _hill(0.1, 0)
    true = pixel_to_terrain(cam, [[320., 150.]], T)[0]
    flat = pixel_to_terrain(cam, [[320., 150.]], None)[0]
    assert np.hypot(*(true[:2] - flat[:2])) > 0.5 and abs(true[2] - flat[2]) > 0.3


def test_end_to_end_clean_and_lossy():
    m = Mission(link=LinkParams(p_gb=0, loss_good=0, ber=0)); snap = m.run()
    ev = evaluate(snap, m)
    assert ev["recall"] >= 0.8 and ev["rmse_m"] < 1.0 and ev["rmse_z_m"] < 0.35
    m2 = Mission(link=LinkParams()); ev2 = evaluate(m2.run(), m2)
    assert m2.link.stats["dropped"] > 0 and ev2["recall"] >= 0.6


def test_terrain_aware_beats_flat():
    ev = {}
    for aware in (False, True):
        R = []
        for seed in (7, 11, 21):
            m = Mission(link=LinkParams(p_gb=0, loss_good=0, ber=0), seed=seed, terrain_aware=aware); m.run()
            R.append(evaluate(m.station.snapshot(m.n * m.dt, 0.15), m))
        ev[aware] = {k: np.nanmean([r[k] for r in R]) for k in ("recall", "rmse_z_m")}
    assert ev[True]["recall"] > ev[False]["recall"] and ev[True]["rmse_z_m"] < 0.6 * ev[False]["rmse_z_m"]
