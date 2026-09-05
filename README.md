<h1 align="center">🛡️ AI Intrusion Detection System</h1>

<p align="center">
  Real-time restricted-zone intrusion detection using <strong>YOLOv8, OpenCV, and object tracking</strong>.
</p>

<p align="center">
  <em>Detect · Track · Validate · Alert · Log</em>
</p>

---

## 📌 Overview

This project is a computer-vision-based intrusion detection system designed around a practical CCTV / physical-security workflow.

It detects and tracks selected objects in a video stream, checks whether they enter a user-defined restricted zone, and confirms an intrusion only after the object remains inside the zone for a configurable number of consecutive frames.

When an intrusion is confirmed, the system can save visual evidence and record an audit entry. The project also includes a **Streamlit web demo** for processing uploaded video clips in environments where webcam or RTSP access is unavailable.

## ✨ Key Features

- **YOLOv8 object detection** for people and vehicles
- **Persistent object tracking** with stable track IDs across frames
- **Polygon-based restricted zones** instead of fixed rectangles
- **Interactive zone selector** for defining a custom camera region
- **Consecutive-frame confirmation** to reduce false alarms
- **Per-object alert cooldown** to prevent repeated alerts
- **Snapshot evidence** saved for confirmed intrusions
- **CSV audit logging** with timestamp, class, track ID, and snapshot path
- **Webcam, video-file, and RTSP CCTV support** in the local detector
- **Headless mode** for server-style execution
- **Streamlit demo** with uploaded-video processing
- Configurable model, confidence, watched classes, zone, thresholds, and output paths

## 🧠 Detection Pipeline

```text
Camera / Video / RTSP
        ↓
    Video Frame
        ↓
 YOLOv8 Detection + Tracking
        ↓
  Filter Watched Classes
        ↓
Bottom-Center Anchor Point
        ↓
 Restricted-Zone Check
        ↓
Consecutive-Frame Validation
        ↓
 Confirmed Intrusion
     ↙          ↘
Snapshot      CSV Log
```

The detector uses the **bottom-center point of each bounding box** as the object's anchor point. This approximates where a person or vehicle is positioned relative to the restricted zone.

## 🖥️ Demo & Runtime Modes

### Streamlit Web Demo

The Streamlit interface is designed for uploaded video clips. It provides:

- YOLO model selection
- Confidence threshold control
- Watched-object selection
- Intrusion confirmation threshold
- Configurable rectangular restricted zone
- Processing progress
- Live preview during processing
- Annotated output video
- Intrusion summary metrics
- Downloadable CSV intrusion log

This mode is suitable for cloud-hosted demos because it does not require the server to access a user's physical webcam or CCTV camera.

### Local Real-Time Detector

The Python detector supports:

- Webcam (`0`)
- Local video files
- RTSP CCTV streams
- Headless/server execution with `--no-display`

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Detection | YOLOv8 / Ultralytics |
| Computer Vision | OpenCV |
| Object Tracking | YOLOv8 tracking with persistent IDs |
| Application | Python |
| Web Demo | Streamlit |
| Data Processing | NumPy, Pandas |
| Configuration | YAML / PyYAML |
| Logging | CSV |

## 📁 Project Structure

```text
AI-Intrusion-Detection/
├── config.yaml                 # Detection and zone configuration
├── requirements.txt            # Python dependencies
├── packages.txt                # System packages for deployment
├── streamlit_app.py            # Uploaded-video web demo
├── src/
│   ├── detector.py             # Main detection + intrusion pipeline
│   ├── zone_selector.py        # Interactive polygon zone selection
│   ├── zone_utils.py           # Polygon calculations + overlays
│   └── alert_utils.py          # Snapshot, CSV, and optional sound alerts
├── alerts/                     # Generated intrusion snapshots
├── logs/                       # Generated CSV logs
├── assets/                     # Optional alert assets
├── .gitignore
└── LICENSE
```

## ⚙️ How Intrusion Detection Works

1. A frame is read from the configured camera or video source.
2. YOLOv8 detects the configured object classes and assigns tracking IDs.
3. The bottom-center point of each bounding box is calculated.
4. The point is tested against the configured restricted-zone polygon.
5. An object must remain inside the zone for `intrusion_frame_threshold` consecutive frames.
6. Once confirmed, the alert manager can save a snapshot and append an entry to the CSV log.
7. `alert_cooldown_seconds` prevents the same tracked object from generating repeated alerts too frequently.

## 🚀 Setup

### 1. Clone the repository

```bash
git clone https://github.com/moizaiqbal40-ops/AI-Intrusion-Detection.git
cd AI-Intrusion-Detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

The YOLO model weights are downloaded automatically by Ultralytics when required.

## 🎥 Local Camera / CCTV Usage

### Step 1 — Select the restricted zone

For a webcam:

```bash
python src/zone_selector.py --source 0
```

Click the points that define the restricted polygon, then save the zone to `config.yaml`.

### Step 2 — Start detection

```bash
python src/detector.py --config config.yaml
```

For a video file:

```bash
python src/detector.py --source path/to/video.mp4
```

For an RTSP camera:

```bash
python src/detector.py --source "rtsp://username:password@camera-ip:554/stream1"
```

For headless execution:

```bash
python src/detector.py --config config.yaml --no-display
```

Press `q` to stop a displayed local detection session.

## 🌐 Streamlit Demo

Run locally with:

```bash
streamlit run streamlit_app.py
```

Then upload an `.mp4`, `.avi`, or `.mov` clip and configure the detection settings from the sidebar.

## 🔧 Configuration

The main configuration lives in `config.yaml`.

| Setting | Purpose |
|---|---|
| `source` | Webcam index, video path, or RTSP URL |
| `model` | YOLO weights such as `yolov8n.pt` |
| `confidence_threshold` | Minimum detection confidence |
| `watched_classes` | COCO class IDs to monitor |
| `restricted_zone` | Polygon coordinates for the protected area |
| `intrusion_frame_threshold` | Frames required before confirmation |
| `alert_cooldown_seconds` | Delay between repeat alerts for one track |
| `snapshot_dir` | Intrusion evidence directory |
| `log_file` | CSV audit-log path |
| `enable_sound_alert` | Optional local sound alert |

The current default configuration monitors **person, car, motorcycle, and truck** with a confidence threshold of `0.45`. fileciteturn12file0

## 📊 Example Output

A confirmed intrusion produces an evidence trail consisting of:

```text
alerts/
└── timestamped_snapshot.jpg

logs/
└── intrusion_log.csv
```

The CSV records information such as the timestamp, detected object class, tracking ID, and snapshot path.

## 📸 Screenshots

<!--
Add screenshots here when the final UI/demo captures are ready.

Recommended gallery:

![Detection](screenshots/detection.png)
![Restricted Zone](screenshots/restricted-zone.png)
![Streamlit Demo](screenshots/streamlit-demo.png)
![Intrusion Log](screenshots/intrusion-log.png)
-->

## 🔐 Engineering Considerations

- Detection thresholds and watched classes are configurable rather than hard-coded.
- Persistent tracking IDs allow intrusion state to be associated with individual objects.
- Consecutive-frame validation helps avoid triggering an alert from a single noisy frame.
- Alert cooldown reduces duplicate events for the same tracked object.
- Local RTSP credentials should be supplied securely and **must not be committed to the repository**.
- The project uses stock COCO classes by default; domain-specific detection would require a suitable custom dataset and model fine-tuning.

## ⚠️ Current Limitations

- The default YOLO model is trained on COCO rather than a security-specific dataset.
- The Streamlit demo processes uploaded videos rather than directly accessing a user's live camera or RTSP stream.
- The Streamlit interface currently represents the restricted zone as a rectangle, while the local detector supports polygon zones.
- Alert delivery is primarily local evidence/log generation; external notification integrations are not included yet.
- Production CCTV deployments would require additional concerns such as authentication, secure secrets management, monitoring, and resource management.

## 🚧 Future Improvements

- Multi-camera / multi-RTSP processing
- Custom YOLO training for domain-specific objects
- Web dashboard for remote alert review
- Email, Telegram, or other notification integrations
- Database-backed event storage
- Authentication and role-based access
- Better zone editing directly inside the web interface
- Automated tests and CI checks
- GPU-aware deployment and performance benchmarking

## 🎯 What This Project Demonstrates

This project demonstrates practical software-engineering and computer-vision concepts including:

- Computer vision pipeline design
- Object detection and tracking
- Spatial reasoning with polygons
- Temporal event validation
- Config-driven application design
- CLI-based tooling
- Web-based ML demos
- Evidence generation and audit logging
- Handling multiple video-source types
- Separation of detection, geometry, and alert responsibilities

## 👩‍💻 Portfolio Note

Built as a hands-on computer-vision project to explore how an AI detection pipeline can be structured around a realistic physical-security workflow — from raw video input to object tracking, zone validation, confirmed events, and auditable evidence.
