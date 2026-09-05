<div align="center">

# 🛡️ AI Intrusion Detection

### Detect restricted-zone intrusions from video using YOLOv8 and tracking.

</div>

## 💡 What It Is

A computer-vision system that detects and tracks selected objects, checks whether they enter a restricted zone, and confirms an intrusion only after it persists for a configurable number of frames.

I built it around a realistic CCTV workflow: **detect → track → validate → alert → log**.

## 🛠️ Tech Stack

- **Python**
- **YOLOv8 / Ultralytics**
- **OpenCV** — video processing
- **NumPy · Pandas**
- **Streamlit** — uploaded-video demo
- **PyYAML** — configuration

## ⚙️ How It Works

```text
Video / CCTV
    ↓
YOLO Detection + Tracking
    ↓
Restricted-Zone Check
    ↓
Consecutive-Frame Validation
    ↓
Confirmed Intrusion
    ↓
Snapshot + CSV Log
```

The local detector supports **webcam, video files, and RTSP streams**. The Streamlit version provides a browser-friendly uploaded-video demo.

## 🚀 Run Locally

```bash
git clone https://github.com/moizaiqbal40-ops/AI-Intrusion-Detection.git
cd AI-Intrusion-Detection
pip install -r requirements.txt
streamlit run streamlit_app.py
```

For the local detector:

```bash
python src/detector.py --config config.yaml
```

## ✨ What I Learned / Challenges

The interesting part was reducing false alerts — instead of reacting to one frame, the system combines tracking, spatial zone checks, and consecutive-frame validation before creating an intrusion event.
