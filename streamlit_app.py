"""
streamlit_app.py
----------------
Web demo for the AI Intrusion Detection project — deployable on
Streamlit Community Cloud.

IMPORTANT: Cloud servers have no camera / no native GUI window, so this
app works on an UPLOADED VIDEO instead of a live webcam feed. Upload any
short clip (a walking person, a car passing through a doorway, etc.),
draw a restricted zone using the sliders, and the app will run YOLOv8
detection + tracking on it and flag intrusions frame by frame.

For a REAL live-camera / CCTV deployment, run `src/detector.py` locally
or on the edge device connected to the camera — see README.md.
"""

import os
import sys
import time
import tempfile
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from collections import defaultdict
from ultralytics import YOLO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from zone_utils import point_in_polygon, bbox_bottom_center, draw_zone, draw_detection  # noqa: E402

st.set_page_config(page_title="AI Intrusion Detection", page_icon="🛡️", layout="wide")

st.title("🛡️ AI-Powered Intrusion Detection — Web Demo")
st.caption(
    "Upload a video clip below. YOLOv8 detects & tracks people/vehicles, and flags "
    "anyone who enters the restricted zone you draw with the sliders. "
    "(This demo processes an uploaded video — a live camera feed needs a local run, see README.)"
)


@st.cache_resource
def load_model(model_name="yolov8n.pt"):
    return YOLO(model_name)


with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.selectbox("YOLO model", ["yolov8n.pt", "yolov8s.pt"], index=0)
    conf_thresh = st.slider("Confidence threshold", 0.1, 0.9, 0.45, 0.05)
    frame_threshold = st.slider("Frames before confirming intrusion", 1, 20, 5)
    watched_labels = st.multiselect(
        "Watch for these object classes",
        ["person", "car", "motorcycle", "truck", "bicycle", "bus"],
        default=["person", "car"],
    )
    process_every_n = st.slider(
        "Process every Nth frame (higher = faster, less smooth)", 1, 5, 2
    )
    st.markdown("---")
    st.markdown(
        "**Restricted zone** (as % of frame width/height — a rectangle for simplicity):"
    )
    x1_pct = st.slider("Left edge (%)", 0, 100, 25)
    y1_pct = st.slider("Top edge (%)", 0, 100, 25)
    x2_pct = st.slider("Right edge (%)", 0, 100, 75)
    y2_pct = st.slider("Bottom edge (%)", 0, 100, 75)

COCO_NAME_TO_ID = {
    "person": 0, "bicycle": 1, "car": 2, "motorcycle": 3,
    "bus": 5, "truck": 7,
}
watched_ids = [COCO_NAME_TO_ID[label] for label in watched_labels] or [0]

uploaded_file = st.file_uploader("Upload a video (mp4 / avi / mov)", type=["mp4", "avi", "mov"])

col1, col2 = st.columns([3, 2])

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

    progress_bar = st.progress(0, text="Processing video...")
    frame_idx = 0

    with col1:
        frame_placeholder = st.empty()

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
                class_name = model.names[cls_id]
                anchor = bbox_bottom_center(box)
                inside = point_in_polygon(anchor, zone)
                label = f"{class_name} #{track_id} {conf:.2f}"
                draw_detection(frame, box, label, color=(0, 0, 255) if inside else (255, 200, 0))

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
            cv2.putText(frame, "INTRUSION DETECTED", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        writer.write(frame)

        if frame_idx % (process_every_n * 5) == 0:
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        progress_bar.progress(min(frame_idx / max(total_frames, 1), 1.0))

    cap.release()
    writer.release()
    progress_bar.empty()

    st.success("Processing complete.")

    with col1:
        st.subheader("Annotated output")
        st.video(out_path)

    with col2:
        st.subheader("🚨 Intrusion alerts")
        if alert_log:
            df = pd.DataFrame(alert_log)
            st.dataframe(df, use_container_width=True)
            st.metric("Total intrusions confirmed", len(alert_log))
        else:
            st.info("No intrusions detected in this clip with the current zone/settings.")

    os.unlink(video_path)

else:
    st.info("👆 Upload a video to run detection. Try a clip of someone walking across a room or a car passing by.")
    st.markdown(
        """
        **For a live webcam / real CCTV RTSP stream**, this app can't be used on Streamlit
        Cloud (no camera access on the server). Instead, run it locally:
        ```bash
        python src/zone_selector.py --source 0     # draw your zone
        python src/detector.py --config config.yaml
        ```
        See `README.md` for full instructions.
        """
    )
