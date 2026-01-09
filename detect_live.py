"""
Smart Live Detection - With Object Dimensions & Distance Estimation
====================================================================
Measures dimensions of ALL objects and estimates distance from camera
(Industrial calibrated)
"""

import cv2
import numpy as np
import csv
import json
from datetime import datetime
import os
import sys


print("="*60)
print("AI DEFECT DETECTION - ENHANCED VERSION")
print("="*60)

# ============================================================
# INDUSTRIAL CAMERA CALIBRATION (ONLY ADDITION)
# ============================================================
FX = 1430.0   # focal length X (pixels)  <-- REPLACE
FY = 1432.0   # focal length Y (pixels)  <-- REPLACE
CX = 640.0
CY = 360.0

# Real object widths (mm) for distance estimation
OBJECT_WIDTHS_MM = {
    "bolt": 12.0,
    "nut": 10.0,
    "screw": 6.0,
    "metal_part": 50.0,
    "defect": 30.0,
    "pencil": 7.0,
    "pen": 8.0,
    "bottle": 65.0,
    "phone": 70.0
}
DEFAULT_WIDTH_MM = 50.0

# ============================================================
# CHECK IF MODEL EXISTS (UNCHANGED)
# ============================================================
MODEL_PATHS = [
    "runs/detect/train_fixed/weights/best.pt",
    "runs/detect/train2/weights/best.pt",
    "best.pt",
]

MODEL_PATH = None
for path in MODEL_PATHS:
    if os.path.exists(path):
        MODEL_PATH = path
        print(f"✓ Found model at: {path}")
        break

if MODEL_PATH is None:
    print("\n❌ ERROR: No trained model found!")
    input("\nPress Enter to exit...")
    exit(1)

# ============================================================
# LOAD YOLO (UNCHANGED)
# ============================================================
from ultralytics import YOLO
model = YOLO(MODEL_PATH)
general_model = YOLO("yolov8n.pt")
USE_GENERAL_MODEL = True

# ============================================================
# CONFIG (UNCHANGED)
# ============================================================
DETECTION_CONFIDENCE = 0.25
IOU_THRESHOLD = 0.45

CRITICAL_THRESHOLD = 20
MODERATE_THRESHOLD = 10

NON_DEFECT_CLASSES = [
    'person','human','people','man','woman','child',
    'pencil','pen','book','bottle','phone','keyboard',
    'mouse','laptop','chair','table'
]

WINDOW_NAME = "AI Defect Detection - Enhanced"
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ============================================================
# OUTPUT FILES (UNCHANGED)
# ============================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_file = f"inspection_{timestamp}.csv"
json_file = f"inspection_{timestamp}.json"
records = []

with open(csv_file, "w", newline="") as f:
    csv.writer(f).writerow([
        "timestamp","frame","label","is_defect",
        "width_mm","height_mm","area_mm2",
        "distance_mm","confidence","severity"
    ])

# ============================================================
# INDUSTRIAL MEASUREMENT FUNCTIONS (EDITED)
# ============================================================
def estimate_distance(width_px, label):
    """Industrial pinhole camera distance"""
    real_width = OBJECT_WIDTHS_MM.get(label.lower(), DEFAULT_WIDTH_MM)
    if width_px <= 0:
        return 0
    return (real_width * FX) / width_px


def calculate_dimensions(w_px, h_px, distance_mm):
    """Depth-aware real-world dimensions"""
    width_mm = (w_px * distance_mm) / FX
    height_mm = (h_px * distance_mm) / FY

    return {
        "width_mm": width_mm,
        "height_mm": height_mm,
        "area_mm2": width_mm * height_mm,
        "diagonal_mm": np.sqrt(width_mm**2 + height_mm**2),
        "perimeter_mm": 2 * (width_mm + height_mm)
    }


def calculate_severity(width_mm, height_mm):
    max_dim = max(width_mm, height_mm)
    if max_dim > CRITICAL_THRESHOLD:
        return "CRITICAL", (0, 0, 255)
    elif max_dim > MODERATE_THRESHOLD:
        return "MODERATE", (0, 165, 255)
    else:
        return "MINOR", (0, 255, 255)

# ============================================================
# CAMERA START (UNCHANGED)
# ============================================================
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

frame_count = 0
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

# ============================================================
# MAIN LOOP (UNCHANGED LOGIC)
# ============================================================
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        results_custom = model(frame, conf=DETECTION_CONFIDENCE, iou=IOU_THRESHOLD, verbose=False)
        results_general = general_model(frame, conf=DETECTION_CONFIDENCE, iou=IOU_THRESHOLD, verbose=False)

        detections = []

        for r in results_custom:
            for box in r.boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                label = model.names[int(box.cls[0])]
                detections.append((x1,y1,x2,y2,label,True,float(box.conf[0])))

        for r in results_general:
            for box in r.boxes:
                label = general_model.names[int(box.cls[0])]
                if label.lower() in NON_DEFECT_CLASSES:
                    x1,y1,x2,y2 = map(int, box.xyxy[0])
                    detections.append((x1,y1,x2,y2,label,False,float(box.conf[0])))

        for x1,y1,x2,y2,label,is_defect,conf in detections:
            w_px = x2 - x1
            h_px = y2 - y1

            distance_mm = estimate_distance(w_px, label)
            dims = calculate_dimensions(w_px, h_px, distance_mm)

            severity, color = calculate_severity(
                dims["width_mm"], dims["height_mm"]
            ) if is_defect else ("OK",(0,255,0))

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(
                frame,
                f"{label} | {distance_mm/10:.1f}cm | {dims['width_mm']:.1f}x{dims['height_mm']:.1f}mm",
                (x1,y1-10),
                FONT,0.5,color,2
            )

            record = {
                "timestamp": datetime.now().isoformat(),
                "frame": frame_count,
                "label": label,
                "is_defect": is_defect,
                "width_mm": round(dims["width_mm"],2),
                "height_mm": round(dims["height_mm"],2),
                "area_mm2": round(dims["area_mm2"],2),
                "distance_mm": round(distance_mm,2),
                "confidence": round(conf,3),
                "severity": severity
            }

            records.append(record)
            with open(csv_file,"a",newline="") as f:
                csv.writer(f).writerow(record.values())

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

    with open(json_file,"w") as f:
        json.dump({
            "calibration":{
                "FX":FX,"FY":FY,"CX":CX,"CY":CY,
                "object_widths_mm":OBJECT_WIDTHS_MM
            },
            "detections":records
        },f,indent=4)

    print("✔ Detection complete")
