"""
Smart Live Detection - COMPLETE FIXED VERSION
==============================================
Full working defect detection system with camera stability
"""

import cv2
import numpy as np
import csv
import json
from datetime import datetime
import os
import sys
import traceback


class MultiReferenceCalibration:
    """Per-class reference dimension system for accurate real-world measurements"""
    
    def __init__(self):
        self.reference_dimensions = {
            'crack': (50, 30, 0.7), 'scratch': (80, 10, 0.6), 'dent': (60, 60, 0.7),
            'corrosion': (100, 100, 0.6), 'hole': (40, 40, 0.8), 'chip': (30, 30, 0.7),
            'deformation': (120, 80, 0.6), 'defect': (70, 50, 0.65),
            'pencil': (180, 8, 0.9), 'pen': (140, 10, 0.9), 'book': (210, 280, 0.8),
            'phone': (70, 150, 0.85), 'cell phone': (70, 150, 0.85), 'bottle': (70, 200, 0.8),
            'cup': (80, 100, 0.7), 'keyboard': (450, 150, 0.85), 'mouse': (110, 60, 0.8),
            'laptop': (350, 240, 0.85), 'monitor': (500, 300, 0.8), 'scissors': (180, 80, 0.8),
            'ruler': (300, 30, 0.9), 'stapler': (120, 50, 0.75), 'notebook': (210, 297, 0.8),
            'unknown': (100, 100, 0.5),
        }
        self.sanity_bounds = {
            'width_mm': (5, 2000),
            'height_mm': (5, 2000),
            'distance_mm': (100, 5000),
            'area_mm2': (25, 4000000)
        }
        print("✓ Multi-Reference Calibration System Initialized")
        print(f"  Reference objects: {len(self.reference_dimensions)}")
    
    def get_reference_for_class(self, class_name, confidence):
        class_name_lower = class_name.lower().strip()
        
        if class_name_lower in self.reference_dimensions:
            ref_width, ref_height, base_confidence = self.reference_dimensions[class_name_lower]
        else:
            matched = False
            for ref_class in self.reference_dimensions.keys():
                if ref_class in class_name_lower or class_name_lower in ref_class:
                    ref_width, ref_height, base_confidence = self.reference_dimensions[ref_class]
                    matched = True
                    break
            if not matched:
                ref_width, ref_height, base_confidence = self.reference_dimensions['unknown']
        
        adjusted_confidence = base_confidence * min(confidence, 1.0)
        return ref_width, ref_height, adjusted_confidence
    
    def calculate_adaptive_distance(self, pixel_width, pixel_height, class_name, confidence, focal_length):
        if pixel_width <= 0:
            return 0
        
        ref_width, ref_height, conf_factor = self.get_reference_for_class(class_name, confidence)
        distance_from_width = (ref_width * focal_length) / pixel_width
        
        if pixel_height > 0:
            distance_from_height = (ref_height * focal_length) / pixel_height
            aspect_ratio = pixel_width / pixel_height
            
            if 0.3 <= aspect_ratio <= 3.0:
                distance_mm = (distance_from_width * 0.6 + distance_from_height * 0.4) * conf_factor
            else:
                distance_mm = distance_from_width * conf_factor
        else:
            distance_mm = distance_from_width * conf_factor
        
        min_dist, max_dist = self.sanity_bounds['distance_mm']
        distance_mm = max(min_dist, min(distance_mm, max_dist))
        
        return distance_mm
    
    def calculate_adaptive_dimensions(self, pixel_width, pixel_height, distance_mm, focal_length, class_name, confidence):
        if distance_mm <= 0 or focal_length <= 0:
            return {
                'width_mm': 0, 'height_mm': 0, 'area_mm2': 0,
                'diagonal_mm': 0, 'perimeter_mm': 0,
                'measurement_quality': "INVALID"
            }
        
        width_mm = (pixel_width * distance_mm) / focal_length
        height_mm = (pixel_height * distance_mm) / focal_length
        
        width_mm = max(5, min(width_mm, 2000))
        height_mm = max(5, min(height_mm, 2000))
        
        area_mm2 = width_mm * height_mm
        diagonal_mm = np.sqrt(width_mm**2 + height_mm**2)
        perimeter_mm = 2 * (width_mm + height_mm)
        
        if confidence > 0.7:
            quality = "HIGH"
        elif confidence > 0.5:
            quality = "MEDIUM"
        else:
            quality = "LOW"
        
        return {
            'width_mm': width_mm,
            'height_mm': height_mm,
            'area_mm2': area_mm2,
            'diagonal_mm': diagonal_mm,
            'perimeter_mm': perimeter_mm,
            'measurement_quality': quality
        }


class BlueprintManager:
    """Manages reference blueprints for comparison"""
    
    def __init__(self):
        self.blueprints = {}
        os.makedirs("blueprints", exist_ok=True)
        os.makedirs("comparisons", exist_ok=True)
        print("✓ Blueprint Manager Initialized")
    
    def has_blueprint(self, class_name):
        return class_name in self.blueprints
    
    def list_blueprints(self):
        return list(self.blueprints.keys())
    
    def capture_blueprint(self, frame, bbox, class_name):
        x1, y1, x2, y2 = bbox
        padding = 10
        h, w = frame.shape[:2]
        
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        blueprint_img = frame[y1:y2, x1:x2].copy()
        
        if blueprint_img.size == 0:
            return False
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"blueprints/{class_name}_{timestamp}.jpg"
        cv2.imwrite(filename, blueprint_img)
        
        self.blueprints[class_name] = {
            'image': blueprint_img,
            'filename': filename,
            'timestamp': timestamp
        }
        
        print(f"\n✓ Blueprint captured for '{class_name}'")
        print(f"  Saved: {filename}")
        return True


def draw_detection_box(frame, x1, y1, x2, y2, label, dimensions, distance_mm, confidence, severity, color, is_defect):
    """Draw detection box with all information"""
    thickness = 3 if severity == "CRITICAL" else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Info box
    info_height = 140
    cv2.rectangle(frame, (x1, y1 - info_height), (x2, y1), (0, 0, 0), -1)
    cv2.rectangle(frame, (x1, y1 - info_height), (x2, y1), color, 2)
    
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    y_offset = y1 - info_height + 20
    
    type_text = "DEFECT" if is_defect else "NON-DEFECT"
    cv2.putText(frame, f"[{type_text}]", (x1 + 5, y_offset), FONT, 0.4, (255, 255, 255), 2)
    y_offset += 20
    
    cv2.putText(frame, f"{label.upper()} ({confidence:.1%})", (x1 + 5, y_offset), FONT, 0.45, (255, 255, 255), 2)
    y_offset += 20
    
    cv2.putText(frame, f"W:{dimensions['width_mm']:.1f}mm H:{dimensions['height_mm']:.1f}mm",
                (x1 + 5, y_offset), FONT, 0.4, (0, 255, 255), 1)
    y_offset += 20
    
    cv2.putText(frame, f"Area:{dimensions['area_mm2']:.1f}mm² Diag:{dimensions['diagonal_mm']:.1f}mm",
                (x1 + 5, y_offset), FONT, 0.35, (255, 255, 0), 1)
    y_offset += 20
    
    distance_cm = distance_mm / 10
    cv2.putText(frame, f"Distance: {distance_cm:.1f}cm ({distance_mm:.0f}mm)",
                (x1 + 5, y_offset), FONT, 0.4, (255, 128, 0), 1)
    y_offset += 20
    
    quality_color = (0, 255, 0) if dimensions['measurement_quality'] == "HIGH" else \
                   (0, 165, 255) if dimensions['measurement_quality'] == "MEDIUM" else (128, 128, 128)
    cv2.putText(frame, f"Quality: {dimensions['measurement_quality']}",
                (x1 + 5, y_offset), FONT, 0.35, quality_color, 1)
    y_offset += 20
    
    if is_defect:
        cv2.putText(frame, f"Severity: {severity}", (x1 + 5, y_offset), FONT, 0.5, color, 2)
    else:
        cv2.putText(frame, "OK - Tracking Only", (x1 + 5, y_offset), FONT, 0.4, (0, 255, 0), 1)


def draw_status_bar(frame, defect_count, fps, total_objects, non_defect_count):
    """Draw status bar at top and bottom"""
    h, w = frame.shape[:2]
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    
    # Top status
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
    cv2.rectangle(frame, (0, h - 70), (w, h), (40, 40, 40), -1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, h - 45), FONT, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"Total: {total_objects}", (150, h - 45), FONT, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"Defects: {defect_count}", (350, h - 45), FONT, 0.6, (255, 0, 0), 2)
    cv2.putText(frame, f"Non-Defects: {non_defect_count}", (550, h - 45), FONT, 0.6, (0, 255, 0), 2)


def draw_grid(frame, spacing=100):
    """Draw reference grid overlay"""
    h, w = frame.shape[:2]
    for x in range(0, w, spacing):
        cv2.line(frame, (x, 0), (x, h), (80, 80, 80), 1)
    for y in range(0, h, spacing):
        cv2.line(frame, (0, y), (w, y), (80, 80, 80), 1)


def calculate_severity(width_mm, height_mm, critical_threshold=20, moderate_threshold=10):
    """Determine severity level based on dimensions"""
    max_dim = max(width_mm, height_mm)
    if max_dim > critical_threshold:
        return "CRITICAL", (0, 0, 255)
    elif max_dim > moderate_threshold:
        return "MODERATE", (0, 165, 255)
    else:
        return "MINOR", (0, 255, 255)


def main():
    print("="*70)
    print("AI DEFECT DETECTION SYSTEM - COMPLETE WORKING VERSION")
    print("="*70)
    
    # === MODEL LOADING ===
    MODEL_PATHS = [
        "runs/detect/train_fixed/weights/best.pt",
        "runs/detect/train2/weights/best.pt",
        "best.pt",
    ]
    
    MODEL_PATH = None
    for path in MODEL_PATHS:
        if os.path.exists(path):
            MODEL_PATH = path
            print(f"\n✓ Found model at: {path}")
            break
    
    if MODEL_PATH is None:
        print("\n❌ ERROR: No trained model found!")
        print("\nSearched paths:")
        for path in MODEL_PATHS:
            print(f"  - {path}")
        print("\nYou need to train a model first or specify the correct path.")
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    # === LOAD YOLO ===
    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n❌ ERROR: ultralytics not installed!")
        print("Install with: pip install ultralytics")
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    try:
        model = YOLO(MODEL_PATH)
        print("✓ Custom model loaded successfully!")
        print(f"  Classes: {model.names}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        traceback.print_exc()
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    # === OPTIONAL GENERAL MODEL ===
    print("\nLoading YOLOv8 for general object detection...")
    try:
        general_model = YOLO('yolov8n.pt')
        print("✓ General detection model loaded!")
        USE_GENERAL_MODEL = True
    except Exception as e:
        print(f"⚠ Warning: Could not load general model: {e}")
        USE_GENERAL_MODEL = False
    
    # === CALIBRATION SETTINGS ===
    CAMERA_FOCAL_LENGTH_PX = 700.0
    DETECTION_CONFIDENCE = 0.25
    IOU_THRESHOLD = 0.45
    CRITICAL_THRESHOLD = 20
    MODERATE_THRESHOLD = 10
    
    NON_DEFECT_CLASSES = [
        'person', 'human', 'people', 'man', 'woman', 'child', 'face', 'hand', 'body',
        'pedestrian', 'pencil', 'pen', 'book', 'bottle', 'cup', 'phone', 'cell phone',
        'keyboard', 'mouse', 'laptop', 'monitor', 'chair', 'table', 'scissors', 'ruler',
        'stapler', 'paper', 'notebook'
    ]
    
    print(f"\n{'='*70}")
    print("CALIBRATION SETTINGS")
    print(f"{'='*70}")
    print(f"Camera Focal Length: {CAMERA_FOCAL_LENGTH_PX:.1f} pixels")
    print(f"Detection Confidence: {DETECTION_CONFIDENCE}")
    print(f"IOU Threshold: {IOU_THRESHOLD}")
    print(f"Critical Threshold: {CRITICAL_THRESHOLD}mm")
    print(f"Moderate Threshold: {MODERATE_THRESHOLD}mm")
    print(f"{'='*70}\n")
    
    # === INITIALIZE SYSTEMS ===
    calibrator = MultiReferenceCalibration()
    blueprint_manager = BlueprintManager()
    
    # === CAMERA INITIALIZATION ===
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ ERROR: Camera not available!")
        print("\nTroubleshooting:")
        print("  1. Check if camera is connected")
        print("  2. Close other apps using the camera")
        print("  3. Try changing camera index (0 to 1)")
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce lag
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✓ Camera initialized: {actual_width}x{actual_height}")
    
    # === SETUP LOGGING ===
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
            "width_mm", "height_mm", "area_mm2", "diagonal_mm", "perimeter_mm",
            "distance_mm", "distance_cm", "confidence", "severity", "measurement_quality"
        ])
    
    # === RUNTIME VARIABLES ===
    frame_count = 0
    fps = 0
    fps_start = datetime.now()
    show_grid = False
    last_detections = []
    
    WINDOW_NAME = "AI Defect Detection (Q=Quit, S=Screenshot, G=Grid, B=Blueprint)"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)
    
    print("\n" + "="*70)
    print("DETECTION STARTED")
    print("="*70)
    print("\nControls:")
    print("  Q/ESC - Quit")
    print("  S - Screenshot")
    print("  G - Toggle grid overlay")
    print("  B - Capture blueprints of detected objects")
    print("\nPress Q to stop...\n")
    
    # === MAIN LOOP ===
    try:
        while True:
            # Read frame with error handling
            ret, frame = cap.read()
            
            if not ret or frame is None:
                print("⚠ Frame read failed, attempting camera recovery...")
                cap.release()
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    print("❌ Camera reconnection failed!")
                    break
                continue
            
            frame_count += 1
            
            # Calculate FPS
            if frame_count % 30 == 0:
                fps_end = datetime.now()
                elapsed = (fps_end - fps_start).total_seconds()
                fps = 30 / max(elapsed, 0.001)
                fps_start = fps_end
            
            # Draw grid if enabled
            if show_grid:
                draw_grid(frame)
            
            # === RUN DETECTION ===
            try:
                results = model(frame, conf=DETECTION_CONFIDENCE, iou=IOU_THRESHOLD, verbose=False)
                
                if USE_GENERAL_MODEL:
                    general_results = general_model(frame, conf=DETECTION_CONFIDENCE, iou=IOU_THRESHOLD, verbose=False)
                else:
                    general_results = []
            
            except Exception as e:
                print(f"⚠ Detection error: {e}")
                continue
            
            # === PROCESS DETECTIONS ===
            all_detections = []
            
            # Custom model detections
            for r in results:
                for box in r.boxes:
                    try:
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
                    except Exception as e:
                        print(f"⚠ Box processing error: {e}")
                        continue
            
            # General model detections (if enabled)
            if USE_GENERAL_MODEL:
                for r in general_results:
                    for box in r.boxes:
                        try:
                            cls = int(box.cls[0])
                            label = general_model.names[cls]
                            confidence = float(box.conf[0])
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            # Check for overlap with custom detections
                            is_overlap = False
                            for det in all_detections:
                                if det['source'] == 'custom':
                                    dx1, dy1, dx2, dy2 = det['box']
                                    ix1 = max(x1, dx1)
                                    iy1 = max(y1, dy1)
                                    ix2 = min(x2, dx2)
                                    iy2 = min(y2, dy2)
                                    
                                    if ix1 < ix2 and iy1 < iy2:
                                        inter_area = (ix2 - ix1) * (iy2 - iy1)
                                        box1_area = (x2 - x1) * (y2 - y1)
                                        box2_area = (dx2 - dx1) * (dy2 - dy1)
                                        iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)
                                        
                                        if iou > 0.3:
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
                        except Exception as e:
                            print(f"⚠ General detection error: {e}")
                            continue
            
            total_objects = len(all_detections)
            defect_count = 0
            non_defect_count = 0
            last_detections = all_detections.copy()
            
            # === DRAW DETECTIONS ===
            for detection in all_detections:
                try:
                    x1, y1, x2, y2 = detection['box']
                    label = detection['label']
                    confidence = detection['confidence']
                    is_defect = detection['is_defect']
                    
                    # Calculate dimensions
                    w_px = x2 - x1
                    h_px = y2 - y1
                    area_px = w_px * h_px
                    
                    distance_mm = calibrator.calculate_adaptive_distance(
                        w_px, h_px, label, confidence, CAMERA_FOCAL_LENGTH_PX
                    )
                    
                    dimensions = calibrator.calculate_adaptive_dimensions(
                        w_px, h_px, distance_mm, CAMERA_FOCAL_LENGTH_PX, label, confidence
                    )
                    
                    distance_cm = distance_mm / 10
                    
                    # Determine severity
                    if is_defect:
                        severity, color = calculate_severity(
                            dimensions['width_mm'], 
                            dimensions['height_mm'],
                            CRITICAL_THRESHOLD,
                            MODERATE_THRESHOLD
                        )
                        defect_count += 1
                    else:
                        severity = "N/A"
                        color = (0, 255, 0)
                        non_defect_count += 1
                    
                    # Draw detection
                    draw_detection_box(
                        frame, x1, y1, x2, y2, label, dimensions,
                        distance_mm, confidence, severity, color, is_defect
                    )
                    
                    # Log record
                    record = {
                        "timestamp": datetime.now().isoformat(),
                        "frame_number": frame_count,
                        "object_type": label,
                        "is_defect": is_defect,
                        "bbox_x1": x1,
                        "bbox_y1": y1,
                        "bbox_x2": x2,
                        "bbox_y2": y2,
                        "width_px": w_px,
                        "height_px": h_px,
                        "area_px": area_px,
                        "width_mm": round(dimensions['width_mm'], 2),
                        "height_mm": round(dimensions['height_mm'], 2),
                        "area_mm2": round(dimensions['area_mm2'], 2),
                        "diagonal_mm": round(dimensions['diagonal_mm'], 2),
                        "perimeter_mm": round(dimensions['perimeter_mm'], 2),
                        "distance_mm": round(distance_mm, 2),
                        "distance_cm": round(distance_cm, 2),
                        "confidence": round(confidence, 3),
                        "severity": severity,
                        "measurement_quality": dimensions['measurement_quality']
                    }
                    
                    records.append(record)
                    
                    # Write to CSV
                    with open(csv_file, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(record.values())
                
                except Exception as e:
                    print(f"⚠ Detection drawing error: {e}")
                    continue
            
            # === DRAW STATUS ===
            draw_status_bar(frame, defect_count, fps, total_objects, non_defect_count)
            
            # === DISPLAY FRAME ===
            cv2.imshow(WINDOW_NAME, frame)
            
            # === HANDLE KEYBOARD ===
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC
                print("\nStopping detection...")
                break
            
            elif key == ord('s') or key == ord('S'):
                screenshot = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(screenshot, frame)
                print(f"✓ Screenshot saved: {screenshot}")
            
            elif key == ord('g') or key == ord('G'):
                show_grid = not show_grid
                print(f"✓ Grid overlay: {'ON' if show_grid else 'OFF'}")
            
            elif key == ord('b') or key == ord('B'):
                if last_detections:
                    print("\n" + "="*50)
                    print("CAPTURING BLUEPRINTS")
                    print("="*50)
                    for i, det in enumerate(last_detections):
                        label = det['label']
                        bbox = det['box']
                        success = blueprint_manager.capture_blueprint(frame, bbox, label)
                        if success:
                            print(f"  ✓ [{i+1}] Captured: {label}")
                    print("="*50)
                else:
                    print("⚠ No objects detected to capture")
    
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()
    
    finally:
        # === CLEANUP ===
        print("\nCleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        
        # === SAVE SUMMARY ===
        if records:
            defect_records = [r for r in records if r['is_defect']]
            non_defect_records = [r for r in records if not r['is_defect']]
            
            distances = [r['distance_cm'] for r in records]
            areas = [r['area_mm2'] for r in records]
            
            summary = {
                "session_info": {
                    "start_time": timestamp,
                    "end_time": datetime.now().isoformat(),
                    "total_frames": frame_count,
                    "total_objects_detected": len(records),
                    "total_defects": len(defect_records),
                    "total_non_defects": len(non_defect_records),
                    "model_path": MODEL_PATH,
                    "calibration": {
                        "camera_focal_length_px": CAMERA_FOCAL_LENGTH_PX,
                        "detection_confidence": DETECTION_CONFIDENCE,
                        "iou_threshold": IOU_THRESHOLD
                    }
                },
                "statistics": {
                    "avg_distance_cm": round(np.mean(distances), 2) if distances else 0,
                    "min_distance_cm": round(np.min(distances), 2) if distances else 0,
                    "max_distance_cm": round(np.max(distances), 2) if distances else 0,
                    "avg_area_mm2": round(np.mean(areas), 2) if areas else 0
                },
                "records": records
            }
            
            with open(json_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\n{'='*70}")
            print("SESSION SUMMARY")
            print(f"{'='*70}")
            print(f"Total Frames Processed: {frame_count}")
            print(f"Total Objects Detected: {len(records)}")
            print(f"  - Defects: {len(defect_records)}")
            print(f"  - Non-Defects: {len(non_defect_records)}")
            print(f"\nData saved to:")
            print(f"  CSV: {csv_file}")
            print(f"  JSON: {json_file}")
            print(f"{'='*70}")
        
        print("\n✓ Shutdown complete")


if __name__ == "__main__":
    main()
