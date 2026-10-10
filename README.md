# C.R.A.T.E.R. - edge hazard pipeline (fully local, pure software, terrain-aware)

Perception -> Spatial Math -> Compact Telemetry -> Lossy Transport -> Dashboard

    pip install -r requirements.txt
    python -m pytest -q                 # 15 automatic tests (~25 s)
    python run_demo.py                  # headless end-to-end, writes outputs/*.png + metrics
    streamlit run dashboard/app.py      # Mission Control UI

| Module | Role |
|---|---|
| `crater/config.py` | camera, uncertainty and link parameters |
| `crater/synth.py` | synthetic lunar scenes with exact ground truth (warped through the same homography) |
| `crater/perception.py` | lightweight classical-CV detector (no weights, ~65 ms/frame on CPU) |
| `crater/geometry.py` | K, R, t, P = K[R\|t], plane homography H(z) = [p1 p2 z*p3+p4], 3-D projection |
| `crater/terrain.py` | DEM class, ray-vs-heightfield intersection (march + bisect), implicit-differentiation covariance incl. slope and DEM error |
| `crater/uncertainty.py` | trust score T and covariance (confidence + distance + age) |
| `crater/telemetry.py` | 60-byte packets (5 hazards: x,y int16 dm, z int8 0.25 m, ellipse, trust), CRC-16, self-contained |
| `crater/channel.py` | Gilbert-Elliott burst loss + jitter/reordering + bit errors |
| `crater/groundstation.py` | decode, dedupe, associate, information-filter fusion, age-aware snapshot |
| `crater/pipeline.py` | `Mission` (rover drives, edge -> link -> station) and `evaluate` |
| `dashboard/` | Streamlit UI (`app.py`) and plotting helpers |

## Using real or other images
`process_frame(img_gray, cam, frame_id, t_s)` takes any grayscale frame. Set `Camera(...)` to your real
intrinsics/height/pitch. Good open datasets: NASA/Chang'e/LROC crops, "Artificial Lunar Landscape Dataset" (Kaggle),
and Blender/Unreal renders (you then export your own K, R, t and ground truth).

## Terrain-aware localisation (replaces the flat-ground assumption)
Ray p(lam) = C + lam*d, d = R^T K^-1 (u,v,1)^T.  Solve F(lam) = C_z + lam*d_z - h(C_xy + lam*d_xy) = 0 (first root).
Flat terrain reduces exactly to the homography (tested).  Covariance by implicit differentiation:
F_lam = d_z - grad(h).d_xy,  dlam/dq = -lam*(dd_z/dq - grad(h).dd_xy/dq)/F_lam,
Sigma = sigma_px^2 J J^T + (sigma_dem/F_lam)^2 d_xy d_xy^T.  `python run_demo.py` prints the flat-vs-terrain ablation.

## Known limitations (be upfront with the judges)
* Terrain comes from a DEM the edge node is assumed to have (e.g. LOLA orbital data); the DEM here is the true
  simulated terrain plus a modelled error sigma_dem in the covariance (not a perturbed copy).
* Rocks are flat discs on the terrain in the simulator; real rocks have height (use their base point).
* Camera attitude (pitch/yaw) is assumed known from an IMU; attitude/odometry error is not yet in the covariance.
* Detector is classical CV tuned to the synthetic look; swap in a tiny YOLO/UNet by replacing `detect()`.
