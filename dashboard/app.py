"""C.R.A.T.E.R. Mission Control
Run from the project root:
    streamlit run dashboard/app.py
"""

import sys
import time
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from crater.config import LinkParams
from crater.channel import expected_loss
from crater.pipeline import Mission, evaluate
from crater.replay import ReplayMission
from dashboard.plots import map_figure, annotate_frame


# ───────────────────── Page configuration ─────────────────────

st.set_page_config(
    page_title="C.R.A.T.E.R. | Mission Control",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --bg: #080d17;
        --panel: #101a2a;
        --panel-light: #152238;
        --line: #24344b;
        --cyan: #57d7ed;
        --text: #e8f0fa;
        --muted: #91a3bb;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 50% -20%, #172943 0%, #080d17 60%);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: #0b1220;
        border-right: 1px solid #1d2a3d;
    }

    [data-testid="stSidebar"] > div {
        padding-top: 1.2rem;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1800px;
    }

    h1, h2, h3 {
        color: var(--text);
        letter-spacing: -0.035em;
    }

    .hero {
        display: flex;
        flex-direction: column;
        gap: 5px;
        padding: 8px 0 18px 0;
    }

    .eyebrow {
        color: var(--cyan);
        font-size: 0.69rem;
        font-weight: 800;
        letter-spacing: 0.2em;
        text-transform: uppercase;
    }

    .hero-title {
        color: #f2f7ff;
        font-size: clamp(1.8rem, 3vw, 2.65rem);
        line-height: 1.1;
        font-weight: 800;
        letter-spacing: -0.055em;
        margin: 0;
    }

    .hero-subtitle {
        color: var(--muted);
        font-size: 0.88rem;
        margin-top: 3px;
    }

    .status-wrap {
        display: flex;
        justify-content: flex-end;
        padding-top: 8px;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        border: 1px solid #31516a;
        border-radius: 999px;
        background: #102638;
        color: #8ce9f4;
        padding: 7px 12px;
        font-size: 0.69rem;
        font-weight: 800;
        letter-spacing: 0.09em;
        white-space: nowrap;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #57d7ed;
        display: inline-block;
    }

    .section-label {
        color: #b7c7db;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin: 12px 0 8px 0;
    }

    .section-note {
        color: var(--muted);
        font-size: 0.78rem;
        margin: -3px 0 10px 0;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #111d30, #0d1625);
        border: 1px solid #22344b;
        border-radius: 12px;
        padding: 14px 15px;
        min-height: 105px;
    }

    [data-testid="stMetricLabel"] {
        color: #9aadc4;
        font-size: 0.75rem;
    }

    [data-testid="stMetricValue"] {
        color: #edf6ff;
        font-size: clamp(1.25rem, 1.8vw, 1.8rem);
        font-weight: 750;
    }

    [data-testid="stMetricDelta"] {
        font-size: 0.73rem;
    }

    [data-testid="stProgressBar"] > div {
        background: #17263a;
        border-radius: 20px;
    }

    [data-testid="stProgressBar"] > div > div {
        background: linear-gradient(90deg, #2c9fb9, #67e4f0);
        border-radius: 20px;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #22344b;
        border-radius: 10px;
        overflow: hidden;
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(100deg, #168ba8, #27b8cb);
        border: none;
        color: #04131c;
        font-weight: 850;
        letter-spacing: 0.05em;
        min-height: 44px;
        border-radius: 9px;
        transition: filter 0.2s ease;
    }

    div.stButton > button[kind="primary"]:hover {
        filter: brightness(1.13);
        border: none;
        color: #04131c;
    }

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    hr {
        border-color: #203047;
        margin: 0.8rem 0 1rem 0;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────── Sidebar ───────────────────────────

with st.sidebar:
    st.markdown(
        """
        <div class="eyebrow">C.R.A.T.E.R. / CONTROL</div>
        <h2 style="margin:5px 0 0 0;">Mission setup</h2>
        <p style="color:#91a3bb;font-size:0.8rem;">
            Configure the simulation environment.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("*01 · SCENARIO*")
    seed = st.number_input("Scene seed", 0, 999, 7)
    n_frames = st.slider("Mission frames", 4, 20, 12)
    use_replay = st.checkbox(
        "Rover camera: prepared image sequence",
        value=True,
        help="Plays saved rover camera frames from disk through the pipeline. Off = images are rendered live.",
    )

    st.markdown("*02 · COMMUNICATION LINK*")
    p_gb = st.slider("Good → Bad transition", 0.0, 0.6, 0.10, 0.01)
    p_bg = st.slider("Bad → Good transition", 0.05, 1.0, 0.30, 0.05)
    loss_bad = st.slider("Packet loss in Bad state", 0.0, 1.0, 0.80, 0.05)
    repeat = st.slider("Redundant packet copies", 1, 3, 1)

    link = LinkParams(
        p_gb=p_gb,
        p_bg=p_bg,
        loss_bad=loss_bad,
    )

    st.info(f"Estimated packet loss: *{expected_loss(link):.0%}*")

    st.markdown("*03 · LOCALISATION*")
    terrain_aware = st.checkbox(
        "Terrain-aware ray-casting",
        value=True,
        help="Off uses classic flat-ground homography.",
    )
    relief = st.slider("Terrain relief scale", 0.0, 2.0, 1.0, 0.1)

    st.markdown("*04 · DISPLAY*")
    show_dem = st.checkbox("Terrain hillshade and contours", True)
    min_trust = st.slider("Minimum hazard trust", 0.0, 0.5, 0.10, 0.01)
    show_truth = st.checkbox("Show ground truth", True)
    delay = st.slider("Frame interval (seconds)", 0.0, 1.0, 0.25, 0.05)

    st.markdown("")
    run = st.button(
        "▶️  RUN MISSION",
        type="primary",
        use_container_width=True,
    )

    st.caption("SIMULATION ENVIRONMENT · LOCAL EXECUTION")


AUTOPLAY = True   # True: the mission starts by itself when the page opens (no click needed in front of the judges)
if AUTOPLAY and "autoplayed" not in st.session_state:
    st.session_state["autoplayed"] = True
    run = True

frame_history = []   # recent annotated camera frames for the filmstrip


# ───────────────────────── Main header ─────────────────────────

header_left, header_right = st.columns([4, 1])

with header_left:
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Autonomous perception · Telemetry · Mapping</div>
            <div class="hero-title">Mission Control</div>
            <div class="hero-subtitle">
                Edge detection → telemetry link → uncertainty-aware ground map
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with header_right:
    status_slot = st.empty()
    status_slot.markdown(
        """
        <div class="status-wrap">
            <div class="status-pill">
                <span class="status-dot"></span> STANDBY
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()


# ───────────────────── Reusable display areas ──────────────────

st.markdown('<div class="section-label">Live telemetry</div>', unsafe_allow_html=True)

metric_columns = st.columns(6)
metric_slots = [column.empty() for column in metric_columns]

progress_slot = st.empty()

camera_column, map_column = st.columns([1.15, 1], gap="large")

with map_column:
    st.markdown('<div class="section-label">02 / Ground situation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Hazard positions, uncertainty and trajectory.</div>',
        unsafe_allow_html=True,
    )
    map_slot = st.empty()

with camera_column:
    st.markdown('<div class="section-label">01 / Rover camera · live feed</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Frames from the rover camera with on-board hazard detections.</div>',
        unsafe_allow_html=True,
    )
    cam_slot = st.empty()
    strip_slot = st.empty()

st.markdown('<div class="section-label">Detected hazard register</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-note">Latest received observations, ordered by trust.</div>',
    unsafe_allow_html=True,
)
table_slot = st.empty()

st.markdown('<div class="section-label">Mission assessment</div>', unsafe_allow_html=True)
score_slot = st.empty()


# ───────────────────────── Render functions ────────────────────

def render_metrics(m, snap):
    """Update the existing telemetry cells rather than adding new ones."""
    link_stats = m.link.stats
    station_stats = m.station.stats

    metrics = [
        ("Packets sent", link_stats["sent"], f'{link_stats["bytes_sent"]} bytes'),
        ("Received OK", station_stats["rx_packets"], None),
        ("Lost in air", link_stats["dropped"], None),
        ("CRC rejects", station_stats["crc_fail"], None),
        ("Hazards tracked", len(snap), None),
        (
            "Data age",
            f'{max((item["age"] for item in snap), default=0):.1f} s',
            None,
        ),
    ]

    for slot, (label, value, help_text) in zip(metric_slots, metrics):
        slot.metric(label, value, help=help_text)


def render_frame(m, result, snap, frame_index):
    """Refresh the dashboard in place for the current mission frame."""
    render_metrics(m, snap)

    progress_slot.progress(
        (frame_index + 1) / n_frames,
        text=f"Mission progress · Frame {frame_index + 1} of {n_frames}",
    )

    map_slot.pyplot(
        map_figure(
            snap,
            m.poses[: result["i"] + 1],
            m.scene.hazards if show_truth else None,
            result["now"],
            terrain=m.terrain if show_dem else None,
        ),
        clear_figure=True,
        use_container_width=True,
    )

    cam_slot.image(
        annotate_frame(result["img"], result["info"]),
        caption=f"Rover camera · frame {result['i'] + 1} · "
                f"{'loaded from prepared sequence' if use_replay else 'rendered live'} · "
                f"{len(result['info'])} hazards detected on-board",
        use_container_width=True,
    )

    recent = frame_history[-4:]
    if recent:
        with strip_slot.container():
            strip_cols = st.columns(4)
            for col, (im, k) in zip(strip_cols, recent):
                col.image(im, caption=f"Frame {k + 1}", use_container_width=True)

    if snap:
        columns = [
            "id", "cls", "x", "y", "z", "r",
            "sig_major", "sig_minor", "age", "trust", "n_obs",
        ]
        df = pd.DataFrame(snap)[columns].round(2)
        df = df.sort_values("trust", ascending=False)

        table_slot.dataframe(
            df,
            hide_index=True,
            use_container_width=True,
        )
    else:
        table_slot.info(
            "No hazards currently meet the selected trust threshold."
        )


def render_assessment(result, mission):
    """Show final simulator metrics after the mission completes."""
    score_slot.empty()

    with score_slot.container():
        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Recall", f"{result['recall']:.0%}")
        c2.metric("Precision", f"{result['precision']:.0%}")
        c3.metric("Position RMSE", f"{result['rmse_m']:.2f} m")
        c4.metric("Elevation RMSE", f"{result['rmse_z_m']:.2f} m")

        st.caption(
            f"Scoring uses simulator ground truth · "
            f"{result['n_gt']} observable hazards · "
            f"Final assessment only"
        )


# ───────────────────────── Mission execution ───────────────────

if run:
    status_slot.markdown(
        """
        <div class="status-wrap">
            <div class="status-pill">
                <span class="status-dot"></span> MISSION ACTIVE
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    score_slot.info("Mission assessment will appear when the simulation finishes.")

    MissionClass = ReplayMission if use_replay else Mission
    mission = MissionClass(
        n_frames=n_frames,
        seed=int(seed),
        link=link,
        repeat=repeat,
        terrain_aware=terrain_aware,
        relief=relief,
    )

    result = None

    frame_history.clear()
    for i in range(n_frames):
        result = mission.step(i)
        frame_history.append((annotate_frame(result["img"], result["info"]), i))
        snapshot = mission.station.snapshot(result["now"], min_trust)

        render_frame(mission, result, snapshot, i)

        if delay > 0:
            time.sleep(delay)

    # Flush remaining packets, matching the existing simulator workflow.
    for receive_time, raw_packet in mission.link.poll(1e9):
        mission.station.ingest(raw_packet, receive_time)

    snapshot = mission.station.snapshot(result["now"], min_trust)

    render_frame(mission, result, snapshot, n_frames - 1)

    assessment = evaluate(snapshot, mission)
    render_assessment(assessment, mission)

    status_slot.markdown(
        """
        <div class="status-wrap">
            <div class="status-pill">
                <span class="status-dot"></span> MISSION COMPLETE
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

else:
    for slot, label, value in zip(
        metric_slots,
        [
            "Packets sent",
            "Received OK",
            "Lost in air",
            "CRC rejects",
            "Hazards tracked",
            "Data age",
        ],
        ["—", "—", "—", "—", "—", "—"],
    ):
        slot.metric(label, value)

    progress_slot.progress(0, text="Awaiting mission launch")

    map_slot.info("The ground situation map will appear after launch.")
    cam_slot.info("The onboard camera feed will appear after launch.")
    table_slot.info("Hazard observations will appear here.")
    score_slot.caption("Final performance metrics will appear after the mission.")
