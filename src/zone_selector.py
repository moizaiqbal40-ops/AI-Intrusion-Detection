"""
zone_selector.py
----------------
Interactive tool to draw the restricted zone polygon directly on a frame
from your camera / CCTV feed, instead of guessing pixel coordinates.

Usage:
    python src/zone_selector.py --source 0
    python src/zone_selector.py --source path/to/cctv_clip.mp4
    python src/zone_selector.py --source "rtsp://user:pass@ip:554/stream1"

Controls:
    Left click   -> add a point to the polygon
    'u'          -> undo last point
    'r'          -> reset all points
    's'          -> save polygon to config.yaml and quit
    'q'          -> quit without saving
"""

import argparse
import yaml
import cv2
import numpy as np

points = []


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])


def main():
    parser = argparse.ArgumentParser(description="Interactive restricted-zone selector")
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open source: {source}")
        return

    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Could not read a frame from the source.")
        return
    cap.release()

    clone = frame.copy()
    cv2.namedWindow("Draw Restricted Zone")
    cv2.setMouseCallback("Draw Restricted Zone", mouse_callback)

    print("Left-click to add points. 'u'=undo  'r'=reset  's'=save  'q'=quit")

    while True:
        display = clone.copy()
        for p in points:
            cv2.circle(display, tuple(p), 4, (0, 0, 255), -1)
        if len(points) > 1:
            pts_array = np.array(points)
            cv2.polylines(display, [pts_array], isClosed=len(points) > 2, color=(0, 255, 0), thickness=2)

        cv2.imshow("Draw Restricted Zone", display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("u") and points:
            points.pop()
        elif key == ord("r"):
            points.clear()
        elif key == ord("s"):
            if len(points) < 3:
                print("[WARN] Need at least 3 points to form a polygon.")
                continue
            with open(args.config, "r") as f:
                cfg = yaml.safe_load(f)
            cfg["restricted_zone"] = [[int(x), int(y)] for x, y in points]
            with open(args.config, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            print(f"[INFO] Saved {len(points)}-point zone to {args.config}")
            break
        elif key == ord("q"):
            print("[INFO] Quit without saving.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
