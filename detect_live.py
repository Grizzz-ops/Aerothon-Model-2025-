import cv2
import numpy as np
import csv
import json
from ultralytics import YOLO
from datetime import datetime
import winsound
from collections import deque

# =========================
# CONFIGURATION
# =========================
MODEL_PATH = r"C:\Users\Admin\OneDrive\Desktop\metal_defect_detection\runs\detect\train2\weights\best.pt"
SCALE_MM_PER_PIXEL = 0.2  # Calibration factor
BEEP_FREQ = 1200
BEEP_DURATION = 150

# Severity thresholds (in mm)
CRITICAL_THRESHOLD = 20
MODERATE_THRESHOLD = 10

# Display settings
WINDOW_NAME = "AI Metal Defect Inspection System"
FONT = cv2.FONT_HERSHEY_SIMPLEX
STATS_UPDATE_INTERVAL = 30  # frames

# =========================
# LOAD MODEL
# =========================
print("Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("Model loaded successfully!")

# =========================
# OUTPUT FILES
# =========================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_file = f"inspection_{timestamp}.csv"
json_file = f"inspection_{timestamp}.json"

records = []
defect_history = deque(maxlen=100)  # Track recent defects for statistics

# Initialize CSV
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "timestamp", "frame_number", "defect_type",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "width_px", "height_px", "area_px",
        "width_mm", "height_mm", "area_mm2",
        "confidence", "severity"
    ])

# =========================
# HELPER FUNCTIONS
# =========================
def calculate_severity(width_mm, height_mm):
    """Determine defect severity based on dimensions"""
    max_dimension = max(width_mm, height_mm)
    
    if max_dimension > CRITICAL_THRESHOLD:
        return "CRITICAL", (0, 0, 255)  # Red
    elif max_dimension > MODERATE_THRESHOLD:
        return "MODERATE", (0, 165, 255)  # Orange
    else:
        return "MINOR", (0, 255, 255)  # Yellow

def draw_info_panel(frame, defect_count, fps, severity_counts):
    """Draw information panel with statistics"""
    h, w = frame.shape[:2]
    panel_height = 120
    
    # Semi-transparent panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - panel_height), (w, h), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Statistics
    y_offset = h - 95
    cv2.putText(frame, f"Total Defects: {defect_count}", 
                (20, y_offset), FONT, 0.7, (255, 255, 255), 2)
    
    cv2.putText(frame, f"FPS: {fps:.1f}", 
                (20, y_offset + 30), FONT, 0.7, (0, 255, 255), 2)
    
    # Severity breakdown
    critical = severity_counts.get('CRITICAL', 0)
    moderate = severity_counts.get('MODERATE', 0)
    minor = severity_counts.get('MINOR', 0)
    
    cv2.putText(frame, f"Critical: {critical}", 
                (w - 400, y_offset), FONT, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, f"Moderate: {moderate}", 
                (w - 400, y_offset + 25), FONT, 0.6, (0, 165, 255), 2)
    cv2.putText(frame, f"Minor: {minor}", 
                (w - 400, y_offset + 50), FONT, 0.6, (0, 255, 255), 2)
    
    # Timestamp
    current_time = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, current_time, 
                (w - 150, y_offset + 30), FONT, 0.6, (255, 255, 255), 2)

def draw_enhanced_bbox(frame, x1, y1, x2, y2, label, width_mm, height_mm, 
                       area_mm2, confidence, severity, color):
    """Draw enhanced bounding box with detailed information"""
    # Main bounding box
    thickness = 3 if severity == "CRITICAL" else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Corner markers for better visibility
    corner_length = 15
    cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, thickness + 1)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, thickness + 1)
    cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, thickness + 1)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, thickness + 1)
    cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, thickness + 1)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, thickness + 1)
    cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, thickness + 1)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, thickness + 1)
    
    # Information background
    info_height = 95
    cv2.rectangle(frame, (x1, y1 - info_height), (x2, y1), (0, 0, 0), -1)
    cv2.rectangle(frame, (x1, y1 - info_height), (x2, y1), color, 2)
    
    # Defect type and confidence
    cv2.putText(frame, f"{label.upper()} ({confidence:.1%})", 
                (x1 + 5, y1 - 70), FONT, 0.5, (255, 255, 255), 2)
    
    # Dimensions
    cv2.putText(frame, f"W: {width_mm:.2f}mm  H: {height_mm:.2f}mm", 
                (x1 + 5, y1 - 48), FONT, 0.45, (0, 255, 255), 1)
    
    # Area
    cv2.putText(frame, f"Area: {area_mm2:.2f}mm2", 
                (x1 + 5, y1 - 28), FONT, 0.45, (0, 255, 255), 1)
    
    # Severity badge
    severity_y = y1 - 8
    cv2.putText(frame, f"[{severity}]", 
                (x1 + 5, severity_y), FONT, 0.55, color, 2)
    
    # Dimension arrows
    # Width arrow
    cv2.arrowedLine(frame, (x1, y2 + 15), (x2, y2 + 15), (255, 255, 0), 2, tipLength=0.03)
    cv2.putText(frame, f"{width_mm:.1f}mm", 
                ((x1 + x2) // 2 - 40, y2 + 35), FONT, 0.5, (255, 255, 0), 2)
    
    # Height arrow
    cv2.arrowedLine(frame, (x2 + 15, y1), (x2 + 15, y2), (255, 255, 0), 2, tipLength=0.03)
    cv2.putText(frame, f"{height_mm:.1f}mm", 
                (x2 + 20, (y1 + y2) // 2), FONT, 0.5, (255, 255, 0), 2)

def draw_status_banner(frame, defects_in_frame):
    """Draw status banner at the top"""
    h, w = frame.shape[:2]
    
    if defects_in_frame > 0:
        banner_text = f"⚠ DEFECT DETECTED ({defects_in_frame})"
        banner_color = (0, 0, 255)
    else:
        banner_text = "✓ OK - NO DEFECTS"
        banner_color = (0, 255, 0)
    
    # Banner background
    cv2.rectangle(frame, (0, 0), (w, 60), banner_color, -1)
    
    # Banner text
    text_size = cv2.getTextSize(banner_text, FONT, 1.3, 3)[0]
    text_x = (w - text_size[0]) // 2
    cv2.putText(frame, banner_text, (text_x, 40), FONT, 1.3, (255, 255, 255), 3)

# =========================
# CAMERA INITIALIZATION
# =========================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

print("Camera initialized successfully!")
print(f"Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
print("\nControls:")
print("  Q/ESC - Quit")
print("  S - Save current frame")
print("  R - Reset statistics")
print("\nStarting inspection...")

# =========================
# STATISTICS
# =========================
defect_count = 0
frame_count = 0
severity_counts = {'CRITICAL': 0, 'MODERATE': 0, 'MINOR': 0}
fps = 0
fps_start_time = datetime.now()

# =========================
# MAIN DETECTION LOOP
# =========================
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame_count += 1
        
        # Calculate FPS
        if frame_count % STATS_UPDATE_INTERVAL == 0:
            fps_end_time = datetime.now()
            fps = STATS_UPDATE_INTERVAL / (fps_end_time - fps_start_time).total_seconds()
            fps_start_time = fps_end_time
        
        # Run detection
        results = model(frame, conf=0.4, verbose=False)
        defects_in_frame = 0
        
        # Process detections
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                confidence = float(box.conf[0])
                
                # Bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Pixel dimensions
                w_px = x2 - x1
                h_px = y2 - y1
                area_px = w_px * h_px
                
                # Real-world dimensions (mm)
                w_mm = w_px * SCALE_MM_PER_PIXEL
                h_mm = h_px * SCALE_MM_PER_PIXEL
                area_mm2 = area_px * (SCALE_MM_PER_PIXEL ** 2)
                
                # Determine severity
                severity, color = calculate_severity(w_mm, h_mm)
                
                defects_in_frame += 1
                defect_count += 1
                severity_counts[severity] += 1
                defect_history.append(severity)
                
                # Draw enhanced bounding box with all information
                draw_enhanced_bbox(frame, x1, y1, x2, y2, label, 
                                 w_mm, h_mm, area_mm2, confidence, 
                                 severity, color)
                
                # Audio alert for critical defects
                if severity == "CRITICAL":
                    try:
                        winsound.Beep(BEEP_FREQ, BEEP_DURATION)
                    except:
                        pass
                
                # Save record
                record = {
                    "timestamp": datetime.now().isoformat(),
                    "frame_number": frame_count,
                    "defect_type": label,
                    "bbox_x1": x1, "bbox_y1": y1,
                    "bbox_x2": x2, "bbox_y2": y2,
                    "width_px": w_px,
                    "height_px": h_px,
                    "area_px": area_px,
                    "width_mm": round(w_mm, 2),
                    "height_mm": round(h_mm, 2),
                    "area_mm2": round(area_mm2, 2),
                    "confidence": round(confidence, 3),
                    "severity": severity
                }
                records.append(record)
                
                # Append to CSV
                with open(csv_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(record.values())
        
        # Draw UI elements
        draw_status_banner(frame, defects_in_frame)
        draw_info_panel(frame, defect_count, fps, severity_counts)
        
        # Display frame
        cv2.imshow(WINDOW_NAME, frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC
            print("\nStopping inspection...")
            break
        elif key == ord('s') or key == ord('S'):  # Save screenshot
            screenshot_file = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(screenshot_file, frame)
            print(f"Screenshot saved: {screenshot_file}")
        elif key == ord('r') or key == ord('R'):  # Reset stats
            defect_count = 0
            severity_counts = {'CRITICAL': 0, 'MODERATE': 0, 'MINOR': 0}
            print("Statistics reset")

except KeyboardInterrupt:
    print("\nInterrupted by user")

finally:
    # =========================
    # CLEANUP
    # =========================
    cap.release()
    cv2.destroyAllWindows()
    
    # Save JSON report
    summary = {
        "session_info": {
            "start_time": timestamp,
            "end_time": datetime.now().isoformat(),
            "total_frames": frame_count,
            "total_defects": defect_count
        },
        "severity_summary": severity_counts,
        "defects": records
    }
    
    with open(json_file, "w") as f:
        json.dump(summary, f, indent=4)
    
    print("\n" + "="*50)
    print("INSPECTION COMPLETE")
    print("="*50)
    print(f"Total frames processed: {frame_count}")
    print(f"Total defects detected: {defect_count}")
    print(f"  - Critical: {severity_counts['CRITICAL']}")
    print(f"  - Moderate: {severity_counts['MODERATE']}")
    print(f"  - Minor: {severity_counts['MINOR']}")
    print(f"\nResults saved:")
    print(f"  CSV: {csv_file}")
    print(f"  JSON: {json_file}")
    print("="*50)
