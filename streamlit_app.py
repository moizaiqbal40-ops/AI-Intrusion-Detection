"""
streamlit_app.py
----------------
Web demo for the AI Intrusion Detection project 

Cloud servers have no camera / no native GUI window, so this app works
on an UPLOADED VIDEO instead of a live webcam feed. Upload any short clip,
draw a restricted zone using the sliders, and it runs YOLOv8 detection +
tracking on it, flagging intrusions frame by frame.

"""

import os
import sys
import tempfile
import cv2
import pandas as pd
import streamlit as st
from collections import defaultdict
from ultralytics import YOLO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from zone_utils import point_in_polygon, bbox_bottom_center, draw_zone, draw_detection  # noqa: E402

st.set_page_config(page_title="AI Intrusion Detection", page_icon="🛡️", layout="wide")

# ------------------------------------------------------------------ #
# Styling
# ------------------------------------------------------------------ #
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #0b1220 0%, #111827 100%); color: #e5e7eb; }

    .hero {
        background: linear-gradient(120deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        padding: 28px 36px;
        border-radius: 18px;
        margin-bottom: 24px;
    }
    .hero h1 { margin: 0 0 6px 0; font-size: 28px; color: #f8fafc; }
    .hero p { margin: 0; color: #94a3b8; font-size: 14.5px; }

    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 16px 18px;
        text-align: center;
    }
    .metric-card .value { font-size: 26px; font-weight: 700; color: #f8fafc; }
    .metric-card .label { font-size: 12.5px; color: #94a3b8; margin-top: 2px; letter-spacing: .02em; }

    .status-pill {
        display: inline-block;
        padding: 5px 16px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 13.5px;
    }
    .pill-safe { background: #052e1c; color: #4ade80; border: 1px solid #166534; }
    .pill-alert { background: #3f0d0d; color: #f87171; border: 1px solid #7f1d1d; }

    section[data-testid="stSidebar"] { background: #0f172a; border-right: 1px solid #1e293b; }
    div[data-testid="stFileUploader"] {
        background: #1e293b; border: 1px dashed #475569; border-radius: 14px; padding: 6px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🛡️ AI-Powered Intrusion Detection</h1>
        <p>YOLOv8 + OpenCV — detects and tracks people & vehicles in real time, and flags
        anyone who enters a restricted zone. Upload a clip below to see it in action.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

COCO_NAME_TO_ID = {"person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "bus": 5, "truck": 7}


@st.cache_resource
def load_model(model_name="yolov8n.pt"):
    return YOLO(model_name)


with st.sidebar:
    st.markdown("### ⚙️ Detection settings")
    model_choice = st.selectbox("YOLO model", ["yolov8n.pt", "yolov8s.pt"], index=0)
    conf_thresh = st.slider("Confidence threshold", 0.1, 0.9, 0.45, 0.05)
    frame_threshold = st.slider("Frames before confirming intrusion", 1, 20, 5)
    watched_labels = st.multiselect(
        "Watch for these classes",
        ["person", "car", "motorcycle", "truck", "bicycle", "bus"],
        default=["person", "car"],
    )
    process_every_n = st.slider("Process every Nth frame", 1, 5, 2, help="Higher = faster, less smooth")

    st.markdown("---")
    st.markdown("### 🟥 Restricted zone")
    st.caption("Rectangle, as % of frame size")
    z1, z2 = st.columns(2)
    x1_pct = z1.slider("Left %", 0, 100, 25)
    y1_pct = z2.slider("Top %", 0, 100, 25)
    z3, z4 = st.columns(2)
    x2_pct = z3.slider("Right %", 0, 100, 75)
    y2_pct = z4.slider("Bottom %", 0, 100, 75)

watched_ids = [COCO_NAME_TO_ID[label] for label in watched_labels] or [0]

uploaded_file = st.file_uploader("📹 Upload a video (mp4 / avi / mov)", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    zone = [
        [int(width * x1_pct / 100), int(height * y1_pct / 100)],
        [int(width * x2_pct / 100), int(height * y1_pct / 100)],
        [int(width * x2_pct / 100), int(height * y2_pct / 100)],
        [int(width * x1_pct / 100), int(height * y2_pct / 100)],
    ]

    model = load_model(model_choice)

    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps / process_every_n, (width, height))

    inside_streak = defaultdict(int)
    alert_log = []
    alerted_ids = set()
    total_unique_objects = set()

    status_placeholder = st.empty()
    progress_bar = st.progress(0, text="Processing video...")
    live_frame = st.empty()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % process_every_n != 0:
            continue

        results = model.track(frame, persist=True, conf=conf_thresh, classes=watched_ids, verbose=False)[0]
        breached = False

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            ids = results.boxes.id.cpu().numpy().astype(int)
            clss = results.boxes.cls.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()

            for box, track_id, cls_id, conf in zip(boxes, ids, clss, confs):
                total_unique_objects.add(int(track_id))
                class_name = model.names[cls_id]
                anchor = bbox_bottom_center(box)
                inside = point_in_polygon(anchor, zone)
                label = f"{class_name} #{track_id} {conf:.2f}"
                draw_detection(frame, box, label, color=(60, 60, 240) if inside else (255, 190, 60))

                if inside:
                    breached = True
                    inside_streak[track_id] += 1
                    if inside_streak[track_id] >= frame_threshold and track_id not in alerted_ids:
                        alerted_ids.add(track_id)
                        alert_log.append({
                            "frame": frame_idx,
                            "time_sec": round(frame_idx / fps, 2),
                            "object": class_name,
                            "track_id": int(track_id),
                        })
                else:
                    inside_streak[track_id] = 0

        draw_zone(frame, zone, breached=breached)
        if breached:
            cv2.putText(frame, "INTRUSION DETECTED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        writer.write(frame)

        if frame_idx % (process_every_n * 4) == 0:
            live_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            pill = '<span class="status-pill pill-alert">🔴 INTRUSION</span>' if breached else '<span class="status-pill pill-safe">🟢 CLEAR</span>'
            status_placeholder.markdown(pill, unsafe_allow_html=True)

        progress_bar.progress(min(frame_idx / max(total_frames, 1), 1.0))

    cap.release()
    writer.release()
    progress_bar.empty()
    live_frame.empty()
    status_placeholder.empty()

    st.success("✅ Processing complete")

    m1, m2, m3, m4 = st.columns(4)
    for col, label, value in [
        (m1, "FRAMES PROCESSED", frame_idx // process_every_n),
        (m2, "OBJECTS TRACKED", len(total_unique_objects)),
        (m3, "INTRUSIONS CONFIRMED", len(alert_log)),
        (m4, "CLASSES WATCHED", ", ".join(watched_labels) or "—"),
    ]:
        col.markdown(
            f'<div class="metric-card"><div class="value">{value}</div><div class="label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    video_col, alert_col = st.columns([3, 2])

    with video_col:
        st.markdown("#### 🎬 Annotated output")
        st.video(out_path)

    with alert_col:
        st.markdown("#### 🚨 Intrusion log")
        if alert_log:
            df = pd.DataFrame(alert_log)
            st.dataframe(df, use_container_width=True, height=260)
            st.download_button(
                "⬇️ Download log (CSV)",
                df.to_csv(index=False).encode("utf-8"),
                file_name="intrusion_log.csv",
                mime="text/csv",
            )
        else:
            st.info("No intrusions detected with the current zone/settings.")

    os.unlink(video_path)

else:
    st.info("👆 Upload a video to run detection — try a clip of someone walking across a room or a car passing by.")
    with st.expander("🎥 Need a real live-camera / CCTV setup instead?"):
        st.markdown(
            """
            A live webcam or RTSP CCTV stream can't run on Streamlit Cloud (no camera
            access on the server). Run it locally instead:
            ```bash
            python src/zone_selector.py --source 0     # draw your zone interactively
            python src/detector.py --config config.yaml
            ```
            Works with a webcam (`0`), a video file, or an RTSP URL from a real camera.
            See `README.md` for full setup.
            """
        )
