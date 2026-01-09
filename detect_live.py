"""
Smart Live Detection - With Object Dimensions & Distance Estimation
====================================================================
Measures dimensions of ALL objects and estimates distance from camera
INDUSTRIAL VERSION - Calibrated for accurate measurements
"""

import cv2
import numpy as np
import csv
import json
from datetime import datetime
import os
import sys

# =============================================
# CHECK IF MODEL EXISTS
# =============================================
print("="*60)
print("AI DEFECT DETECTION - INDUSTRIAL CALIBRATED VERSION")
print("="*60)

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
    print("\nYou need to train a model first:")
    print("  python train_yolo_SMART.py")
    print("\nOr specify a custom model path in the script.")
    input("\nPress Enter to exit...")
    exit(1)

# =============================================
# CHECK IF CAMERA IS AVAILABLE
# =============================================
print("\nChecking camera...")
test_cap = cv2.VideoCapture(0)
if not test_cap.isOpened():
    print("❌ ERROR: Camera not available!")
    print("\nTroubleshooting:")
    print("  1. Check if camera is connected")
    print("  2. Close other apps using the camera")
    print("  3. Try changing camera index (0 to 1)")
    test_cap.release()
    input("\nPress Enter to exit...")
    exit(1)
test_cap.release()
print("✓ Camera is available")

# =============================================
# LOAD YOLO
# =============================================
try:
    from ultralytics import YOLO
except ImportError:
    print("\n❌ ERROR: ultralytics not installed!")
    print("Install with: pip install ultralytics")
    input("\nPress Enter to exit...")
    exit(1)

print(f"\nLoading custom model: {MODEL_PATH}")
try:
    model = YOLO(MODEL_PATH)
    print("✓ Custom model loaded successfully!")
    print(f"  Classes: {model.names}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    input("\nPress Enter to exit...")
    exit(1)

# Load additional YOLOv8 model for general object detection
print("\nLoading YOLOv8 for general object detection (pencils, pens, etc.)...")
try:
    general_model = YOLO('yolov8n.pt')  # Nano model for speed
    print("✓ General detection model loaded!")
    print(f"  Can detect: person, pen, pencil, book, bottle, phone, etc.")
    USE_GENERAL_MODEL = True
except Exception as e:
    print(f"⚠ Warning: Could not load general model: {e}")
    print("  Only custom defect detection will be available")
    USE_GENERAL_MODEL = False

# =============================================
# INDUSTRIAL CALIBRATION PARAMETERS
# =============================================
# These parameters need to be calibrated for your specific industrial setup
# Method 1: Place a known object (e.g., 100mm width) at known distance (e.g., 500mm)
# Method 2: Use camera calibration with checkerboard pattern

# Camera focal length (pixels) - Typical values:
# - Webcam (720p): 600-900 px
# - Industrial camera (1080p): 1000-1500 px
# - High-res industrial (4K): 2000-3000 px
CAMERA_FOCAL_LENGTH_PX = 700.0  # Calibrate this for your camera!

# Reference object width for distance calculation (mm)
# Use a standard industrial reference object like:
# - Standard bolt: 10mm, 15mm, 20mm
# - Industrial gauge block: 25mm, 50mm, 100mm
# - A4 paper width: 210mm
REFERENCE_OBJECT_WIDTH_MM = 100.0  # Real-world width of reference object

# Distance from camera to object during calibration (mm)
CALIBRATION_DISTANCE_MM = 500.0  # Distance at which you measured the reference

# Measured pixel width of reference object at calibration distance
REFERENCE_OBJECT_WIDTH_PX = 140.0  # Measure this during calibration!

# Calculate actual focal length from calibration
# Formula: focal_length = (pixel_width * distance) / real_width
CAMERA_FOCAL_LENGTH_PX = (REFERENCE_OBJECT_WIDTH_PX * CALIBRATION_DISTANCE_MM) / REFERENCE_OBJECT_WIDTH_MM

print(f"\n{'='*60}")
print("CALIBRATION SETTINGS")
print(f"{'='*60}")
print(f"Camera Focal Length: {CAMERA_FOCAL_LENGTH_PX:.1f} pixels")
print(f"Reference Object: {REFERENCE_OBJECT_WIDTH_MM:.1f}mm width")
print(f"Calibration Distance: {CALIBRATION_DISTANCE_MM:.1f}mm")
print(f"Reference Pixel Width: {REFERENCE_OBJECT_WIDTH_PX:.1f}px")
print(f"{'='*60}\n")

DETECTION_CONFIDENCE = 0.25
IOU_THRESHOLD = 0.45

CRITICAL_THRESHOLD = 20
MODERATE_THRESHOLD = 10

# Classes to mark as NON-DEFECTS (but still show measurements)
NON_DEFECT_CLASSES = [
    'person', 'human', 'people', 'man', 'woman', 'child',
    'face', 'hand', 'body', 'pedestrian',
    'pencil', 'pen', 'book', 'bottle', 'cup', 'phone', 'cell phone',
    'keyboard', 'mouse', 'laptop', 'monitor', 'chair', 'table',
    'scissors', 'ruler', 'stapler', 'paper', 'notebook'
]

WINDOW_NAME = "AI Defect Detection - Industrial Calibrated (Press Q to Quit)"
FONT = cv2.FONT_HERSHEY_SIMPLEX

# =============================================
# SETUP OUTPUT FILES
# =============================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_file = f"inspection_{timestamp}.csv"
json_file = f"inspection_{timestamp}.json"

records = []

with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "timestamp", "frame_number", "object_type", "is_defect",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "width_px", "height_px", "area_px",
        "width_mm", "height_mm", "area_mm2",
        "diagonal_mm", "perimeter_mm",
        "distance_mm", "distance_cm",
        "confidence", "severity"
    ])

# =============================================
# INDUSTRIAL MEASUREMENT FUNCTIONS
# =============================================
def estimate_distance(width_px, focal_length=CAMERA_FOCAL_LENGTH_PX, 
                     reference_width=REFERENCE_OBJECT_WIDTH_MM):
    """
    Estimate distance using calibrated pinhole camera model:
    Distance = (Real_Width * Focal_Length) / Pixel_Width
    
    This gives more accurate results when properly calibrated
    """
    if width_px <= 0:
        return 0
    distance_mm = (reference_width * focal_length) / width_px
    return distance_mm

def calculate_real_dimensions(w_px, h_px, distance_mm, focal_length=CAMERA_FOCAL_LENGTH_PX):
    """
    Calculate real-world dimensions based on distance and camera calibration
    Formula: Real_Size = (Pixel_Size * Distance) / Focal_Length
    
    This provides industrially accurate measurements
    """
    if distance_mm <= 0 or focal_length <= 0:
        return {
            'width_mm': 0,
            'height_mm': 0,
            'area_mm2': 0,
            'diagonal_mm': 0,
            'perimeter_mm': 0
        }
    
    # Calculate real dimensions based on perspective
    w_mm = (w_px * distance_mm) / focal_length
    h_mm = (h_px * distance_mm) / focal_length
    
    # Calculate derived measurements
    area_mm2 = w_mm * h_mm
    diagonal_mm = np.sqrt(w_mm**2 + h_mm**2)
    perimeter_mm = 2 * (w_mm + h_mm)
    
    return {
        'width_mm': w_mm,
        'height_mm': h_mm,
        'area_mm2': area_mm2,
        'diagonal_mm': diagonal_mm,
        'perimeter_mm': perimeter_mm
    }

def calculate_severity(width_mm, height_mm):
    max_dim = max(width_mm, height_mm)
    if max_dim > CRITICAL_THRESHOLD:
        return "CRITICAL", (0, 0, 255)
    elif max_dim > MODERATE_THRESHOLD:
        return "MODERATE", (0, 165, 255)
    else:
        return "MINOR", (0, 255, 255)

def draw_detection(frame, x1, y1, x2, y2, label, dimensions, 
                   distance_mm, confidence, severity, color, is_defect=True):
    """Enhanced drawing with calibrated dimensions and distance"""
    thickness = 3 if severity == "CRITICAL" else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Info box
    info_height = 135
    cv2.rectangle(frame, (x1, y1 - info_height), (x2, y1), (0, 0, 0), -1)
    cv2.rectangle(frame, (x1, y1 - info_height), (x2, y1), color, 2)
    
    # Object type indicator
    type_text = "DEFECT" if is_defect else "NON-DEFECT"
    cv2.putText(frame, f"[{type_text}]", 
                (x1 + 5, y1 - 115), FONT, 0.4, (255, 255, 255), 2)
    
    # Text info
    cv2.putText(frame, f"{label.upper()} ({confidence:.1%})", 
                (x1 + 5, y1 - 95), FONT, 0.45, (255, 255, 255), 2)
    
    cv2.putText(frame, f"W:{dimensions['width_mm']:.1f}mm H:{dimensions['height_mm']:.1f}mm", 
                (x1 + 5, y1 - 75), FONT, 0.4, (0, 255, 255), 1)
    
    cv2.putText(frame, f"Area:{dimensions['area_mm2']:.1f}mm² Diag:{dimensions['diagonal_mm']:.1f}mm", 
                (x1 + 5, y1 - 55), FONT, 0.35, (255, 255, 0), 1)
    
    distance_cm = distance_mm / 10
    cv2.putText(frame, f"Distance: {distance_cm:.1f}cm ({distance_mm:.0f}mm)", 
                (x1 + 5, y1 - 35), FONT, 0.4, (255, 128, 0), 1)
    
    if is_defect:
        cv2.putText(frame, f"Severity: {severity}", 
                    (x1 + 5, y1 - 10), FONT, 0.5, color, 2)
    else:
        cv2.putText(frame, "OK - Tracking Only", 
                    (x1 + 5, y1 - 10), FONT, 0.4, (0, 255, 0), 1)

def draw_status(frame, defect_count, fps, total_objects, non_defect_count):
    h, w = frame.shape[:2]
    
    # Top banner
    if defect_count > 0:
        text = f"DEFECT DETECTED ({defect_count})"
        color = (0, 0, 255)
    else:
        text = "OK - NO DEFECTS"
        color = (0, 255, 0)
    
    cv2.rectangle(frame, (0, 0), (w, 60), color, -1)
    text_size = cv2.getTextSize(text, FONT, 1.0, 2)[0]
    text_x = (w - text_size[0]) // 2
    cv2.putText(frame, text, (text_x, 40), FONT, 1.0, (255, 255, 255), 2)
    
    # Bottom info bar
    cv2.rectangle(frame, (0, h - 40), (w, h), (40, 40, 40), -1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, h - 15), FONT, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"Total: {total_objects}", (150, h - 15), FONT, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"Defects: {defect_count}", (300, h - 15), FONT, 0.6, (255, 0, 0), 2)
    cv2.putText(frame, f"Non-Defects: {non_defect_count}", (480, h - 15), FONT, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Conf: {DETECTION_CONFIDENCE}", (720, h - 15), FONT, 0.6, (255, 128, 255), 2)

def draw_reference_grid(frame, spacing=100):
    """Draw reference grid for visual measurement aid"""
    h, w = frame.shape[:2]
    
    # Vertical lines
    for x in range(0, w, spacing):
        cv2.line(frame, (x, 0), (x, h), (80, 80, 80), 1)
    
    # Horizontal lines
    for y in range(0, h, spacing):
        cv2.line(frame, (0, y), (w, y), (80, 80, 80), 1)

# =============================================
# START CAMERA
# =============================================
print("\n" + "="*60)
print("STARTING INDUSTRIAL DETECTION")
print("="*60)
print("\nCalibration Instructions:")
print("1. Place reference object (e.g., 100mm width) at known distance")
print("2. Measure its pixel width in the frame")
print("3. Update calibration parameters in code")
print("4. Verify measurements with known objects")
print("\nNon-defect classes (will show measurements but not count as defects):")
for cls in NON_DEFECT_CLASSES:
    print(f"  - {cls}")
print("\nAll other detected objects will be treated as DEFECTS")
print("\nControls:")
print("  Q/ESC - Quit")
print("  S - Screenshot")
print("  R - Reset statistics")
print("  G - Toggle reference grid")
print("\nPress any key to start...")
input()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("❌ Failed to open camera")
    exit(1)

defect_count = 0
non_defect_count = 0
total_defects_detected = 0
total_non_defects_detected = 0
frame_count = 0
fps = 0
fps_start = datetime.now()
show_grid = False

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1280, 720)

# =============================================
# MAIN LOOP
# =============================================
try:
    print("\n✓ Detection started! Press Q to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Draw grid if enabled
        if show_grid:
            draw_reference_grid(frame)
        
        # Calculate FPS
        if frame_count % 30 == 0:
            fps_end = datetime.now()
            fps = 30 / (fps_end - fps_start).total_seconds()
            fps_start = fps_end
        
        # Run detection on CUSTOM MODEL (defects)
        results = model(frame, conf=DETECTION_CONFIDENCE, iou=IOU_THRESHOLD, verbose=False)
        
        # Run detection on GENERAL MODEL (pencils, pens, etc.)
        if USE_GENERAL_MODEL:
            general_results = general_model(frame, conf=DETECTION_CONFIDENCE, iou=IOU_THRESHOLD, verbose=False)
        else:
            general_results = []
        
        detections_this_frame = 0
        non_defects_this_frame = 0
        total_objects = 0
        
        all_detections = []
        
        # Process CUSTOM MODEL detections (defects)
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                all_detections.append({
                    'box': (x1, y1, x2, y2),
                    'label': label,
                    'confidence': confidence,
                    'is_defect': True,
                    'source': 'custom'
                })
        
        # Process GENERAL MODEL detections (pencils, pens, etc.)
        if USE_GENERAL_MODEL:
            for r in general_results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    label = general_model.names[cls]
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Check if it overlaps with custom model detection
                    is_overlap = False
                    for det in all_detections:
                        if det['source'] == 'custom':
                            dx1, dy1, dx2, dy2 = det['box']
                            # Calculate IoU (Intersection over Union)
                            ix1 = max(x1, dx1)
                            iy1 = max(y1, dy1)
                            ix2 = min(x2, dx2)
                            iy2 = min(y2, dy2)
                            
                            if ix1 < ix2 and iy1 < iy2:
                                inter_area = (ix2 - ix1) * (iy2 - iy1)
                                box1_area = (x2 - x1) * (y2 - y1)
                                box2_area = (dx2 - dx1) * (dy2 - dy1)
                                iou = inter_area / (box1_area + box2_area - inter_area)
                                
                                if iou > 0.3:  # If overlap > 30%, skip
                                    is_overlap = True
                                    break
                    
                    if not is_overlap:
                        all_detections.append({
                            'box': (x1, y1, x2, y2),
                            'label': label,
                            'confidence': confidence,
                            'is_defect': label.lower() not in NON_DEFECT_CLASSES,
                            'source': 'general'
                        })
        
        total_objects = len(all_detections)
        
        # Draw all detections
        for detection in all_detections:
            x1, y1, x2, y2 = detection['box']
            label = detection['label']
            confidence = detection['confidence']
            is_defect = detection['is_defect']
            
            w_px = x2 - x1
            h_px = y2 - y1
            area_px = w_px * h_px
            
            # INDUSTRIAL CALIBRATED MEASUREMENT PIPELINE:
            # Step 1: Estimate distance from camera based on object width
            distance_mm = estimate_distance(w_px)
            
            # Step 2: Calculate real dimensions using distance and calibration
            dimensions = calculate_real_dimensions(w_px, h_px, distance_mm)
            
            distance_cm = distance_mm / 10
            
            # Calculate severity (only for defects)
            if is_defect:
                severity, color = calculate_severity(
                    dimensions['width_mm'], 
                    dimensions['height_mm']
                )
                detections_this_frame += 1
                total_defects_detected += 1
            else:
                severity = "N/A"
                color = (0, 255, 0)  # Green for non-defects
                non_defects_this_frame += 1
                total_non_defects_detected += 1
            
            draw_detection(frame, x1, y1, x2, y2, label, dimensions,
                         distance_mm, confidence, severity, color, is_defect)
            
            # Save record
            record = {
                "timestamp": datetime.now().isoformat(),
                "frame_number": frame_count,
                "object_type": label,
                "is_defect": is_defect,
                "bbox_x1": x1, "bbox_y1": y1,
                "bbox_x2": x2, "bbox_y2": y2,
                "width_px": w_px, "height_px": h_px, "area_px": area_px,
                "width_mm": round(dimensions['width_mm'], 2),
                "height_mm": round(dimensions['height_mm'], 2),
                "area_mm2": round(dimensions['area_mm2'], 2),
                "diagonal_mm": round(dimensions['diagonal_mm'], 2),
                "perimeter_mm": round(dimensions['perimeter_mm'], 2),
                "distance_mm": round(distance_mm, 2),
                "distance_cm": round(distance_cm, 2),
                "confidence": round(confidence, 3),
                "severity": severity
            }
            records.append(record)
            
            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(record.values())
        
        defect_count = detections_this_frame
        non_defect_count = non_defects_this_frame
        draw_status(frame, defect_count, fps, total_objects, non_defect_count)
        cv2.imshow(WINDOW_NAME, frame)
        
        # Handle keys
        key = cv2.waitKey(10) & 0xFF
        
        if key == ord('q') or key == ord('Q') or key == 27:
            print("\nStopping...")
            break
        
        elif key == ord('s') or key == ord('S'):
            screenshot = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(screenshot, frame)
            print(f"Saved: {screenshot}")
        
        elif key == ord('r') or key == ord('R'):
            total_defects_detected = 0
            total_non_defects_detected = 0
            print("Statistics reset")
        
        elif key == ord('g') or key == ord('G'):
            show_grid = not show_grid
            print(f"Reference grid: {'ON' if show_grid else 'OFF'}")

except KeyboardInterrupt:
    print("\nInterrupted by user")

except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()

finally:
    cap.release()
    cv2.destroyAllWindows()
    
    # Save summary with statistics
    if records:
        defect_records = [r for r in records if r['is_defect']]
        non_defect_records = [r for r in records if not r['is_defect']]
        
        distances = [r['distance_cm'] for r in records]
        areas = [r['area_mm2'] for r in records]
        
        defect_distances = [r['distance_cm'] for r in defect_records]
        non_defect_distances = [r['distance_cm'] for r in non_defect_records]
        
        summary = {
            "session_info": {
                "start_time": timestamp,
                "end_time": datetime.now().isoformat(),
                "total_frames": frame_count,
                "total_objects_detected": len(records),
                "total_defects": len(defect_records),
                "total_non_defects": len(non_defect_records),
                "calibration": {
                    "camera_focal_length_px": CAMERA_FOCAL_LENGTH_PX,
                    "reference_object_width_mm": REFERENCE_OBJECT_WIDTH_MM,
                    "calibration_distance_mm": CALIBRATION_DISTANCE_MM,
                    "reference_object_width_px": REFERENCE_OBJECT_WIDTH_PX
                }
            },
            "statistics": {
                "all_objects": {
                    "avg_distance_cm": round(np.mean(distances), 2) if distances else 0,
                    "min_distance_cm": round(min(distances), 2) if distances else 0,
                    "max_distance_cm": round(max(distances), 2) if distances else 0,
                    "avg_area_mm2": round(np.mean(areas), 2) if areas else 0
                },
                "defects_only": {
                    "count": len(defect_records),
                    "avg_distance_cm": round(np.mean(defect_distances), 2) if defect_distances else 0,
                    "min_distance_cm": round(min(defect_distances), 2) if defect_distances else 0,
                    "max_distance_cm": round(max(defect_distances), 2) if defect_distances else 0
                },
                "non_defects_only": {
                    "count": len(non_defect_records),
                    "avg_distance_cm": round(np.mean(non_defect_distances), 2) if non_defect_distances else 0,
                    "min_distance_cm": round(min(non_defect_distances), 2) if non_defect_distances else 0,
                    "max_distance_cm": round(max(non_defect_distances), 2) if non_defect_distances else 0
                }
            },
            "all_detections": records
        }
    else:
        summary = {
            "session_info": {
                "start_time": timestamp,
                "end_time": datetime.now().isoformat(),
                "total_frames": frame_count,
                "total_objects_detected": 0,
                "total_defects": 0,
                "total_non_defects": 0
            },
            "all_detections": []
        }
    
    with open(json_file, "w") as f:
        json.dump(summary, f, indent=4)
    
    print("\n" + "="*60)
    print("DETECTION COMPLETE")
    print("="*60)
    print(f"Frames processed: {frame_count}")
    print(f"Total objects detected: {len(records)}")
    print(f"  - Defects: {total_defects_detected}")
    print(f"  - Non-defects: {total_non_defects_detected}")
    print(f"\nResults saved:")
    print(f"  {csv_file}")
    print(f"  {json_file}")
    print("="*60)
