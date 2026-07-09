"""
alert_utils.py
--------------
Handles what happens when an intrusion is confirmed: saving a snapshot
image, appending a row to the CSV log, and (optionally) playing a sound.
"""

import os
import csv
import cv2
from datetime import datetime


class AlertManager:
    def __init__(self, snapshot_dir="alerts", log_file="logs/intrusion_log.csv",
                 enable_sound=False, sound_path=None):
        self.snapshot_dir = snapshot_dir
        self.log_file = log_file
        self.enable_sound = enable_sound
        self.sound_path = sound_path

        os.makedirs(self.snapshot_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "object_class", "track_id", "snapshot_path"])

    def raise_alert(self, frame, object_class: str, track_id):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{timestamp}_{object_class}_{track_id}.jpg"
        snapshot_path = os.path.join(self.snapshot_dir, filename)
        cv2.imwrite(snapshot_path, frame)

        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, object_class, track_id, snapshot_path])

        print(f"[ALERT] {object_class} (id={track_id}) entered restricted zone -> {snapshot_path}")

        if self.enable_sound and self.sound_path and os.path.exists(self.sound_path):
            try:
                from playsound import playsound
                playsound(self.sound_path, block=False)
            except Exception as e:
                print(f"[WARN] Could not play alert sound: {e}")
