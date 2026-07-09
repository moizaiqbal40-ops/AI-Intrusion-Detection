"""
zone_utils.py
-------------
Helpers for working with a restricted-zone polygon: point-in-polygon
testing and drawing overlays on a video frame.
"""

import cv2
import numpy as np


def point_in_polygon(point, polygon):
    """Return True if `point` (x, y) lies inside `polygon` (list of [x,y])."""
    contour = np.array(polygon, dtype=np.int32)
    result = cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False)
    return result >= 0


def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def bbox_bottom_center(bbox):
    """Bottom-center of a box approximates where a person 'stands' — more
    reliable than the box centroid for zone-crossing decisions."""
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int(y2))


def draw_zone(frame, polygon, breached=False):
    """Draw the restricted zone polygon on the frame. Turns red when breached."""
    overlay = frame.copy()
    color = (0, 0, 255) if breached else (0, 200, 0)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
    return frame


def draw_detection(frame, bbox, label, color=(255, 200, 0)):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
