"""Minimal LIVE dashboard example (reads the files written by run_live.py).
Run order:  1) python run_live.py      2) streamlit run dashboard/live_example.py
Copy whatever you like into your own dashboard. (Not tested inside Streamlit by the author - adjust as needed.)"""
import sys, json, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import streamlit as st
import pandas as pd
from dashboard.plots import map_figure

LIVE = pathlib.Path("live")
st.set_page_config(page_title="C.R.A.T.E.R. LIVE", layout="wide")
st.title("C.R.A.T.E.R. - Mission Control (LIVE)")


def read(name):
    try:
        return json.loads((LIVE / name).read_text())
    except Exception:          # file missing or being rewritten at this instant -> just try again next refresh
        return None


snap, status = read("snapshot.json"), read("status.json")
if not snap or not status:
    st.info("Waiting for the rover... start it with:  python run_live.py")
else:
    ls, gs = status["link_stats"], status["ground_stats"]
    c = st.columns(5)
    c[0].metric("Frames from rover", status["frames"])
    c[1].metric("Packets sent / lost", f"{ls['sent']} / {ls['dropped']}")
    c[2].metric("CRC rejects", gs["crc_fail"])
    c[3].metric("Edge time (ms/frame)", status["last_ms"])
    c[4].metric("Hazards on map", len(snap["hazards"]))
    left, right = st.columns(2)
    with left:
        st.subheader("Rover camera (latest image)")
        if (LIVE / "latest_frame.png").exists():
            st.image(str(LIVE / "latest_frame.png"))
        st.subheader("Recent packets")
        st.dataframe(pd.DataFrame(status["recent_packets"]).iloc[::-1], height=250)
    with right:
        st.subheader("Hazard map (colour = trust, fade = data age, ellipse = uncertainty)")
        st.pyplot(map_figure(snap["hazards"], [], None, snap["time_s"]))
    st.subheader("Hazard table")
    if snap["hazards"]:
        st.dataframe(pd.DataFrame(snap["hazards"])[["id", "cls", "x", "y", "r", "age", "trust", "n_obs"]])
    time.sleep(1.0)
    st.rerun()
