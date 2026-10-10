"""Mission Control dashboard.   Run:  streamlit run dashboard/app.py"""
import sys, time, pathlib
import cv2
import numpy as np
from crater.perception import detect
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
import streamlit as st
from crater.config import Camera, LinkParams
from crater.channel import expected_loss
from crater.pipeline import Mission, evaluate
from dashboard.plots import map_figure, annotate_frame

st.set_page_config(page_title="C.R.A.T.E.R. Mission Control", layout="wide")
st.title("C.R.A.T.E.R. - Mission Control")

st.divider()
st.header("Image Lab — Test a real image")

uploaded = st.file_uploader(
    "Upload a lunar image",
    type=["jpg", "jpeg", "png"],
    help="Upload a local image to test CRATER's experimental candidate detector."
)

if uploaded is not None:
    file_bytes = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image_bgr is None:
        st.error("Could not read this image. Try another JPG or PNG.")
    else:
        # Limit image size to keep processing responsive.
        h, w = image_bgr.shape[:2]
        scale = min(1.0, 1600 / max(h, w))

        if scale < 1.0:
            image_bgr = cv2.resize(
                image_bgr,
                (int(w * scale), int(h * scale))
            )

        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # Provisional camera assumptions, not real-image calibration.
        camera = Camera(width=w, height=h)

        try:
            detections = detect(gray, camera, terrain=None)

            annotated = image_bgr.copy()

            for i, d in enumerate(detections, start=1):
                x, y, bw, bh = d.bbox
                cv2.rectangle(
                    annotated, (x, y), (x + bw, y + bh),
                    (0, 255, 0), 2
                )
                cv2.putText(
                    annotated,
                    f"Candidate {i}: {d.conf:.2f}",
                    (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA
                )

            col_original, col_result = st.columns(2)

            with col_original:
                st.subheader("Uploaded image")
                st.image(
                    cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
                    use_container_width=True
                )

            with col_result:
                st.subheader("Candidate detections")
                st.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    use_container_width=True
                )

            st.metric("Candidate regions", len(detections))

            if detections:
                st.dataframe(
                    [
                        {
                            "Candidate": i,
                            "Pixel X": round(d.u, 1),
                            "Pixel Y": round(d.v, 1),
                            "Confidence": round(d.conf, 3),
                            "Provisional class": (
                                "Crater candidate" if d.cls == 1
                                else "Rock candidate"
                            ),
                            "Bounding box": str(d.bbox),
                        }
                        for i, d in enumerate(detections, start=1)
                    ],
                    hide_index=True
                )
            else:
                st.warning(
                    "No candidate regions detected. This does not prove "
                    "the image contains no hazards."
                )

            st.caption(
                "Experimental classical computer vision. The confidence "
                "score is not a calibrated probability. Pixel detections "
                "are shown; metric size and real-world position are not "
                "validated for this image."
            )

        except Exception as exc:
            st.error(f"Image analysis failed: {exc}")

st.divider()

st.caption("Edge perception -> homography -> 64-byte telemetry -> lossy link -> uncertainty-aware map")

with st.sidebar:
    st.header("Simulation")
    seed = st.number_input("Scene seed", 0, 999, 7)
    n_frames = st.slider("Frames", 4, 20, 12)
    st.header("Radio link (Gilbert-Elliott)")
    p_gb = st.slider("P(Good->Bad)", 0.0, 0.6, 0.10, 0.01)
    p_bg = st.slider("P(Bad->Good)", 0.05, 1.0, 0.30, 0.05)
    loss_bad = st.slider("Loss in Bad state", 0.0, 1.0, 0.80, 0.05)
    repeat = st.slider("Redundant copies per packet", 1, 3, 1)
    link = LinkParams(p_gb=p_gb, p_bg=p_bg, loss_bad=loss_bad)
    st.info(f"Expected packet loss: **{expected_loss(link):.0%}**")
    st.header("Localisation")
    terrain_aware = st.checkbox("Terrain-aware (DEM ray-casting)", True,
                                help="Off = classic flat-ground homography. Compare the position error in the scoring line.")
    relief = st.slider("Terrain relief x", 0.0, 2.0, 1.0, 0.1)
    st.header("Display")
    show_dem = st.checkbox("Show DEM hillshade + contours", True)
    min_trust = st.slider("Hide hazards with trust below", 0.0, 0.5, 0.10, 0.01)
    show_truth = st.checkbox("Show simulator ground truth (grey)", True)
    delay = st.slider("Seconds per frame (animation)", 0.0, 1.0, 0.25, 0.05)
    run = st.button("Run mission", type="primary")

top = st.empty()
col_map, col_cam = st.columns([1.1, 1])
map_slot, cam_slot, table_slot = col_map.empty(), col_cam.empty(), col_cam.empty()


def draw(m, res, snap, final=False):
    ls, ss = m.link.stats, m.station.stats
    c = top.columns(6)
    c[0].metric("Packets sent", ls["sent"], f"{ls['bytes_sent']} B")
    c[1].metric("Received OK", ss["rx_packets"])
    c[2].metric("Lost in air", ls["dropped"])
    c[3].metric("CRC rejects", ss["crc_fail"])
    c[4].metric("Hazards shown", len(snap))
    c[5].metric("Oldest data", f"{max([s['age'] for s in snap], default=0):.0f} s")
    map_slot.pyplot(map_figure(snap, m.poses[:res["i"] + 1], m.scene.hazards if show_truth else None, res["now"], terrain=m.terrain if show_dem else None), clear_figure=True)
    cam_slot.image(annotate_frame(res["img"], res["info"]), caption=f"Edge camera, frame {res['i']} (detections BEFORE the lossy link)")
    if snap:
        df = pd.DataFrame(snap)[["id", "cls", "x", "y", "z", "r", "sig_major", "sig_minor", "age", "trust", "n_obs"]].round(2)
        table_slot.dataframe(df.sort_values("trust", ascending=False), hide_index=True)


if run:
    m = Mission(n_frames=n_frames, seed=int(seed), link=link, repeat=repeat, terrain_aware=terrain_aware, relief=relief)
    for i in range(n_frames):
        res = m.step(i)
        draw(m, res, m.station.snapshot(res["now"], min_trust))
        time.sleep(delay)
    for t_rx, raw in m.link.poll(1e9):
        m.station.ingest(raw, t_rx)
    snap = m.station.snapshot(res["now"], min_trust)
    draw(m, res, snap, True)
    ev = evaluate(snap, m)
    st.success(f"Simulator scoring (uses hidden ground truth): recall {ev['recall']:.0%}, precision {ev['precision']:.0%}, "
               f"position RMSE {ev['rmse_m']:.2f} m, elevation RMSE {ev['rmse_z_m']:.2f} m over {ev['n_gt']} observable hazards.")
else:
    st.write("Set the link parameters on the left and press **Run mission**.")
