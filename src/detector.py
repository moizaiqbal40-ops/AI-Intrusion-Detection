"""
detector.py
-----------
Real-time object detection + restricted-zone intrusion detection using
YOLOv8 (Ultralytics) + OpenCV.

Pipeline:
  1. Read frames from a webcam / video file / RTSP CCTV stream.
  2. Run YOLOv8 detection + tracking on each frame.
  3. Check whether any watched object (person, car, etc.) is inside the
     restricted zone polygon.
  4. If an object stays inside the zone for N consecutive frames, raise
     an alert: save a snapshot + log entry (and optionally play a sound).

Usage:
    python src/detector.py --config config.yaml
    python src/detector.py --source path/to/cctv_clip.mp4
    python src/detector.py --source "rtsp://user:pass@ip:554/stream1"
"""

import os
import sys
import time
import argparse
import yaml
import cv2
from collections import defaultdict
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(__file__))
from zone_utils import point_in_polygon, bbox_bottom_center, draw_zone, draw_detection
from alert_utils import AlertManager


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Real-Time Object/Intrusion Detection")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--source", type=str, default=None, help="Override video source from config")
    parser.add_argument("--no-display", action="store_true", help="Run headless (no cv2.imshow window)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    source = args.source if args.source is not None else cfg["source"]
    # allow "0" string from CLI to mean webcam index 0
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    model = YOLO(cfg["model"])
    coco_names = model.names

    alert_mgr = AlertManager(
        snapshot_dir=cfg["snapshot_dir"],
        log_file=cfg["log_file"],
        enable_sound=cfg.get("enable_sound_alert", False),
        sound_path=cfg.get("alert_sound_path"),
    )

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source}")
        sys.exit(1)

    zone = cfg["restricted_zone"]
    watched_classes = set(cfg.get("watched_classes", [0]))
    conf_thresh = cfg.get("confidence_threshold", 0.45)
    frame_threshold = cfg.get("intrusion_frame_threshold", 5)
    cooldown = cfg.get("alert_cooldown_seconds", 10)

    # Tracks how many consecutive frames each tracked object has spent
    # inside the zone, and when we last alerted for that track id.
    inside_streak = defaultdict(int)
    last_alert_time = defaultdict(lambda: 0.0)

    print("[INFO] Starting detection loop. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Video stream ended.")
            break

        results = model.track(
            frame, persist=True, conf=conf_thresh, classes=list(watched_classes), verbose=False
        )[0]

        breached = False

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            ids = results.boxes.id.cpu().numpy().astype(int)
            clss = results.boxes.cls.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()

            for box, track_id, cls_id, conf in zip(boxes, ids, clss, confs):
                class_name = coco_names[cls_id]
                anchor_point = bbox_bottom_center(box)
                inside = point_in_polygon(anchor_point, zone)

                label = f"{class_name} #{track_id} {conf:.2f}"
                draw_detection(frame, box, label, color=(0, 0, 255) if inside else (255, 200, 0))

                if inside:
                    breached = True
                    inside_streak[track_id] += 1

                    confirmed = inside_streak[track_id] >= frame_threshold
                    cooled_down = (time.time() - last_alert_time[track_id]) > cooldown

                    if confirmed and cooled_down:
                        alert_mgr.raise_alert(frame, class_name, track_id)
                        last_alert_time[track_id] = time.time()
                else:
                    inside_streak[track_id] = 0

        draw_zone(frame, zone, breached=breached)

        if breached:
            cv2.putText(frame, "INTRUSION DETECTED", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        if not args.no_display:
            cv2.imshow("AI Intrusion Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
