"""LIVE mode for the Mission Control dashboard: reads the files written by run_live.py and refreshes every second.

Hook it into dashboard/app.py with 3 small edits (see the comments at the bottom of this file)."""
import json, pathlib, time
import pandas as pd
import streamlit as st
from dashboard.plots import map_figure

LIVE = pathlib.Path("live")
STALE_S = 4.0          # no new snapshot for this long -> show ROVER OFFLINE


def _read(name):
    try:
        return json.loads((LIVE / name).read_text())
    except Exception:          # missing, or being rewritten at this instant -> try again on the next refresh
        return None


def _pill(slot, text, ok=True):
    if slot is None:
        return
    col, bg, line = ("#8ce9f4", "#102638", "#31516a") if ok else ("#ffb4a8", "#33181a", "#6a3131")
    slot.markdown(f'<div class="status-wrap"><div class="status-pill" style="color:{col};background:{bg};border-color:{line}">'
                  f'<span class="status-dot" style="background:{col}"></span> {text}</div></div>', unsafe_allow_html=True)


def _rerun():
    (getattr(st, "rerun", None) or st.experimental_rerun)()


def render_live(min_trust=0.10, status_slot=None):
    snap, status = _read("snapshot.json"), _read("status.json")
    if not snap or not status:
        _pill(status_slot, "WAITING FOR ROVER", ok=False)
        st.info("Waiting for the rover... start it in another window with:  python run_live.py")
        time.sleep(1.0); _rerun(); return

    try:
        since = time.time() - (LIVE / "snapshot.json").stat().st_mtime
    except OSError:
        since = 0.0
    online = since < STALE_S
    _pill(status_slot, f"LIVE · UPDATED {since:.0f} s AGO" if online else f"ROVER OFFLINE · LAST UPDATE {since:.0f} s AGO", ok=online)

    hazards = [h for h in snap["hazards"] if h["trust"] >= min_trust]
    ls, gs = status["link_stats"], status["ground_stats"]
    metrics = [("Packets sent", ls["sent"], f'{ls["bytes_sent"]} bytes'), ("Received OK", gs["rx_packets"], None),
               ("Lost in air", ls["dropped"], None), ("CRC rejects", gs["crc_fail"], None),
               ("Hazards tracked", len(hazards), None),
               ("Data age", f'{max((h["age"] for h in hazards), default=0):.1f} s', None)]
    for col, (label, value, help_text) in zip(st.columns(6), metrics):
        col.metric(label, value, help=help_text)

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.markdown('<div class="section-label">01 / Ground situation (live)</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-note">Colour = trust, fade = data age, ellipse = position uncertainty.</div>', unsafe_allow_html=True)
        st.pyplot(map_figure(hazards, [], None, snap["time_s"]), clear_figure=True, use_container_width=True)
    with right:
        st.markdown('<div class="section-label">02 / Rover camera (latest image)</div>', unsafe_allow_html=True)
        frames = status.get("recent_frames") or []
        last = frames[-1] if frames else None
        try:
            st.image(str(LIVE / "latest_frame.png"), use_container_width=True,
                     caption=(f"{last['image']} · {last['detections']} hazards detected on-board · {last['edge_ms']:.0f} ms" if last else None))
        except Exception:
            st.info("No image from the rover yet. Copy a .png/.jpg into the rover_inbox folder.")

    st.markdown('<div class="section-label">Radio link: recent packets</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-note">Link "{status["link"]}" · expected loss {status["expected_loss"]:.0%}. '
                f'DROPPED = lost in the air, CORRUPTED = rejected by the checksum.</div>', unsafe_allow_html=True)
    pk = status.get("recent_packets") or []
    if pk:
        st.dataframe(pd.DataFrame(pk).iloc[::-1], hide_index=True, use_container_width=True, height=240)

    st.markdown('<div class="section-label">Detected hazard register</div>', unsafe_allow_html=True)
    if hazards:
        cols = ["id", "cls", "x", "y", "z", "r", "sig_major", "sig_minor", "age", "trust", "n_obs"]
        st.dataframe(pd.DataFrame(hazards)[cols].round(2).sort_values("trust", ascending=False), hide_index=True, use_container_width=True)
    else:
        st.info("No hazards meet the selected trust threshold yet.")

    time.sleep(1.0)
    _rerun()


# ---- How to hook it into dashboard/app.py (3 small edits) -------------------------------------------------------
# 1) In the sidebar, directly after the first  st.divider() :
#        source = st.radio("Data source", ["Simulation", "Live rover (run_live.py)"])
# 2) Directly after the header's  st.divider()  (the one just before "Reusable display areas"):
#        if source.startswith("Live"):
#            from dashboard.live_view import render_live
#            render_live(min_trust, status_slot)
#            st.stop()
# 3) Nothing else. Start the rover in another window first:  python run_live.py
