# 🛡️ AI-Powered Real-Time Intrusion Detection (YOLOv8 + OpenCV)

A real-time object detection and **restricted-zone intrusion detection**
system built with YOLOv8 and OpenCV. Point it at a webcam, a video file, or
a live RTSP CCTV stream, draw a restricted zone on the frame, and it will
detect and log any person/vehicle that enters it — complete with snapshot
evidence and a CSV audit trail.

This was built to directly mirror a real physical-security / DSS (Digital
Surveillance Systems) use case: turning raw CCTV footage into automated
suspicious-activity alerts instead of relying on someone watching a monitor.

## Features

- **Real-time detection + tracking** using YOLOv8 (Ultralytics), so each
  object keeps a stable ID across frames instead of being re-detected from
  scratch every time.
- **Custom restricted zone** — draw any polygon (not just a rectangle) as
  the "do not enter" area.
- **Interactive zone selector** (`src/zone_selector.py`) — click points
  directly on your own camera feed instead of guessing pixel coordinates.
- **False-alarm reduction** — an object must stay inside the zone for N
  consecutive frames before it counts as a confirmed intrusion, and each
  tracked object has a cooldown so one intrusion doesn't spam dozens of
  alerts.
- **Evidence trail** — every confirmed intrusion saves a timestamped
  snapshot image to `alerts/` and appends a row to `logs/intrusion_log.csv`
  (timestamp, object class, track ID, snapshot path).
- **Works with any video source**: webcam (`0`), a video file, or a live
  RTSP CCTV stream URL.
- Fully configurable via `config.yaml` — no code changes needed to point it
  at a new camera or change which object classes to watch.

## Project structure

```
ai-intrusion-detection/
├── config.yaml              # source, model, zone, thresholds — all settings live here
├── requirements.txt
├── src/
│   ├── detector.py          # main detection + intrusion logic
│   ├── zone_selector.py     # click-to-draw restricted zone on your own feed
│   ├── zone_utils.py        # polygon math + drawing overlays
│   └── alert_utils.py       # snapshot saving + CSV logging + optional sound alert
├── alerts/                  # saved intrusion snapshots (generated)
└── logs/                    # intrusion_log.csv (generated)
```

## Setup

```bash
git clone <your-repo-url>
cd ai-intrusion-detection
pip install -r requirements.txt
```

The first run will auto-download the YOLOv8-nano weights (`yolov8n.pt`, ~6MB)
via Ultralytics — no manual download needed.

## Step 1 — Draw your restricted zone

Run this once against your actual camera/video to set the zone accurately:

```bash
python src/zone_selector.py --source 0
```

Left-click to add polygon points, press `s` to save straight into
`config.yaml`, or `q` to cancel.

## Step 2 — Run the detector

```bash
python src/detector.py --config config.yaml
```

Or override the source directly from the CLI:

```bash
python src/detector.py --source path/to/cctv_clip.mp4
python src/detector.py --source "rtsp://username:password@192.168.1.64:554/stream1"
```

Press `q` to quit the live window. Use `--no-display` to run headless on a
server (e.g. a Raspberry Pi or a cloud box with no monitor attached).

## How the intrusion logic works

1. YOLOv8 detects and tracks every object of interest (person, car,
   motorcycle, truck by default — configurable in `config.yaml`).
2. For each tracked object, we check whether its **bottom-center point**
   (approximates where it's "standing") falls inside the restricted zone
   polygon.
3. If it stays inside for `intrusion_frame_threshold` consecutive frames
   (default: 5), the intrusion is confirmed.
4. On confirmation (and if the per-object cooldown has passed), a snapshot
   is saved and a log row is written.

## Configuration reference (`config.yaml`)

| Key | Description |
|---|---|
| `source` | `0` for webcam, a file path, or an RTSP URL |
| `model` | YOLOv8 weights file (nano/small/medium — trade speed vs accuracy) |
| `confidence_threshold` | Minimum detection confidence to consider |
| `watched_classes` | COCO class IDs to track (0=person, 2=car, etc.) |
| `restricted_zone` | Polygon points `[[x,y], ...]` |
| `intrusion_frame_threshold` | Consecutive frames before confirming an intrusion |
| `alert_cooldown_seconds` | Minimum time between repeat alerts for the same object |
| `snapshot_dir` / `log_file` | Where evidence is saved |

## Why this matters (DSS / physical security angle)

This project is designed to plug directly into existing CCTV infrastructure:
any camera that can output an RTSP stream can feed this pipeline with no
hardware changes. It turns passive footage into active alerting — the same
core capability commercial DSS (Digital Surveillance Systems) products
sell, built here from first principles to show the underlying computer
vision pipeline end-to-end.

## Future work

- Add multi-camera support (run one detector instance per RTSP stream).
- Send alerts to Telegram/WhatsApp/email instead of (or in addition to) a
  local CSV log.
- Fine-tune YOLO on a custom dataset for domain-specific objects (e.g.
  weapons, unattended bags) rather than relying only on stock COCO classes.
- Add a lightweight web dashboard to review snapshots and logs remotely.
