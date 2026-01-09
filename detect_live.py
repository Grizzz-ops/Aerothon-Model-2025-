"""
Smart Live Detection - WITH BLUEPRINT REFERENCE SYSTEM
=======================================================
Captures reference objects and detects structural differences in real-time
"""

import cv2
import numpy as np
import csv
import json
from datetime import datetime
import os
import sys

# =============================================
# BLUEPRINT COMPARISON FUNCTIONS (NEW)
# =============================================

class BlueprintManager:
    """Manages reference blueprints and comparison"""
    
    def __init__(self):
        self.blueprints = {}  # Store multiple blueprints by class name
        self.blueprint_features = {}  # Store features for comparison
        self.comparison_enabled = False
        
        # Create blueprints directory
        os.makedirs("blueprints", exist_ok=True)
        
        # Initialize feature detectors
        self.orb = cv2.ORB_create(nfeatures=500)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    def capture_blueprint(self, frame, bbox, class_name):
        """Capture a region as blueprint reference"""
        x1, y1, x2, y2 = bbox
        
        # Extract region with padding
        padding = 10
        h, w = frame.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        blueprint_img = frame[y1:y2, x1:x2].copy()
        
        if blueprint_img.size == 0:
            return False
        
        # Store blueprint
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"blueprints/{class_name}_{timestamp}.jpg"
        cv2.imwrite(filename, blueprint_img)
        
        self.blueprints[class_name] = {
            'image': blueprint_img,
            'filename': filename,
            'timestamp': timestamp,
            'bbox_size': (x2-x1, y2-y1)
        }
        
        # Extract features for comparison
        gray = cv2.cvtColor(blueprint_img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        self.blueprint_features[class_name] = {
            'keypoints': keypoints,
            'descriptors': descriptors,
            'edges': cv2.Canny(gray, 50, 150),
            'contours': self._extract_contours(gray),
            'text_regions': self._detect_text_regions(gray)
        }
        
        print(f"\n✓ Blueprint captured for '{class_name}'")
        print(f"  Saved: {filename}")
        print(f"  Features: {len(keypoints)} keypoints")
        print(f"  Text regions: {len(self.blueprint_features[class_name]['text_regions'])}")
        
        return True
    
    def _extract_contours(self, gray_img):
        """Extract structural contours"""
        _, binary = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter significant contours
        significant = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:  # Minimum area threshold
                significant.append({
                    'contour': cnt,
                    'area': area,
                    'perimeter': cv2.arcLength(cnt, True)
                })
        
        return significant
    
    def _detect_text_regions(self, gray_img):
        """Detect potential text regions using MSER"""
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray_img)
        
        text_regions = []
        for region in regions:
            if len(region) > 10:  # Minimum region size
                x, y, w, h = cv2.boundingRect(region)
                # Filter by aspect ratio (typical for text)
                aspect_ratio = w / float(h) if h > 0 else 0
                if 0.1 < aspect_ratio < 10:
                    text_regions.append({'bbox': (x, y, w, h), 'points': len(region)})
        
        return text_regions
    
    def compare_to_blueprint(self, frame, bbox, class_name):
        """Compare detected object to blueprint"""
        if class_name not in self.blueprints:
            return None, "No blueprint available"
        
        x1, y1, x2, y2 = bbox
        padding = 10
        h, w = frame.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        current_img = frame[y1:y2, x1:x2].copy()
        
        if current_img.size == 0:
            return None, "Invalid region"
        
        # Resize to match blueprint size
        blueprint_img = self.blueprints[class_name]['image']
        current_img_resized = cv2.resize(current_img, 
                                         (blueprint_img.shape[1], blueprint_img.shape[0]))
        
        # Convert to grayscale
        gray_current = cv2.cvtColor(current_img_resized, cv2.COLOR_BGR2GRAY)
        gray_blueprint = cv2.cvtColor(blueprint_img, cv2.COLOR_BGR2GRAY)
        
        # Initialize results
        differences = {
            'feature_match_score': 0,
            'structural_similarity': 0,
            'missing_text_regions': [],
            'contour_differences': 0,
            'overall_defect': False,
            'defect_reasons': []
        }
        
        # 1. Feature Matching (ORB)
        kp_current, desc_current = self.orb.detectAndCompute(gray_current, None)
        blueprint_features = self.blueprint_features[class_name]
        
        if desc_current is not None and blueprint_features['descriptors'] is not None:
            matches = self.bf_matcher.match(blueprint_features['descriptors'], desc_current)
            match_ratio = len(matches) / max(len(blueprint_features['keypoints']), 1)
            differences['feature_match_score'] = match_ratio
            
            if match_ratio < 0.3:  # Less than 30% features matched
                differences['overall_defect'] = True
                differences['defect_reasons'].append(f"Low feature match: {match_ratio:.1%}")
        
        # 2. Structural Similarity (SSIM)
        differences['structural_similarity'] = self._compute_ssim(gray_blueprint, gray_current)
        
        if differences['structural_similarity'] < 0.7:  # Less than 70% similar
            differences['overall_defect'] = True
            differences['defect_reasons'].append(
                f"Low structural similarity: {differences['structural_similarity']:.1%}"
            )
        
        # 3. Text Region Comparison
        current_text_regions = self._detect_text_regions(gray_current)
        blueprint_text_count = len(blueprint_features['text_regions'])
        current_text_count = len(current_text_regions)
        
        if current_text_count < blueprint_text_count:
            missing_count = blueprint_text_count - current_text_count
            differences['missing_text_regions'] = missing_count
            differences['overall_defect'] = True
            differences['defect_reasons'].append(
                f"Missing text regions: {missing_count} (expected {blueprint_text_count}, found {current_text_count})"
            )
        
        # 4. Contour Comparison
        current_contours = self._extract_contours(gray_current)
        blueprint_contour_count = len(blueprint_features['contours'])
        current_contour_count = len(current_contours)
        
        contour_diff = abs(blueprint_contour_count - current_contour_count)
        differences['contour_differences'] = contour_diff
        
        if contour_diff > 2:  # More than 2 contours different
            differences['overall_defect'] = True
            differences['defect_reasons'].append(
                f"Structural elements differ: {contour_diff} components (expected {blueprint_contour_count}, found {current_contour_count})"
            )
        
        # 5. Pixel-level difference
        diff_img = cv2.absdiff(gray_blueprint, gray_current)
        diff_percentage = (np.count_nonzero(diff_img > 30) / diff_img.size) * 100
        differences['pixel_difference'] = diff_percentage
        
        if diff_percentage > 40:  # More than 40% pixels different
            differences['overall_defect'] = True
            differences['defect_reasons'].append(f"High pixel difference: {diff_percentage:.1f}%")
        
        return differences, current_img_resized
    
    def _compute_ssim(self, img1, img2):
        """Compute Structural Similarity Index"""
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        
        mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return float(np.mean(ssim_map))
    
    def has_blueprint(self, class_name):
        """Check if blueprint exists for class"""
        return class_name in self.blueprints
    
    def list_blueprints(self):
        """List all stored blueprints"""
        return list(self.blueprints.keys())
    
    def delete_blueprint(self, class_name):
        """Delete a blueprint"""
        if class_name in self.blueprints:
            del self.blueprints[class_name]
            del self.blueprint_features[class_name]
            print(f"✓ Deleted blueprint for '{class_name}'")
            return True
        return False

# =============================================
# ORIGINAL CODE CONTINUES BELOW
# =============================================

print("="*60)
print("AI DEFECT DETECTION - WITH BLUEPRINT SYSTEM")
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

print("\nLoading YOLOv8 for general object detection (pencils, pens, etc.)...")
try:
    general_model = YOLO('yolov8n.pt')
    print("✓ General detection model loaded!")
    print(f"  Can detect: person, pen, pencil, book, bottle, phone, etc.")
    USE_GENERAL_MODEL = True
except Exception as e:
    print(f"⚠ Warning: Could not load general model: {e}")
    print("  Only custom defect detection will be available")
    USE_GENERAL_MODEL = False

# =============================================
# CONFIGURATION (ORIGINAL)
# =============================================
SCALE_MM_PER_PIXEL = 0.2
KNOWN_OBJECT_WIDTH_MM = 50.0
CAMERA_FOCAL_LENGTH_PX = 800
DETECTION_CONFIDENCE = 0.25
IOU_THRESHOLD = 0.45
CRITICAL_THRESHOLD = 20
MODERATE_THRESHOLD = 10

NON_DEFECT_CLASSES = [
    'person', 'human', 'people', 'man', 'woman', 'child',
    'face', 'hand', 'body', 'pedestrian',
    'pencil', 'pen', 'book', 'bottle', 'cup', 'phone', 'cell phone',
    'keyboard', 'mouse', 'laptop', 'monitor', 'chair', 'table',
    'scissors', 'ruler', 'stapler', 'paper', 'notebook'
]

WINDOW_NAME = "AI Defect Detection - WITH BLUEPRINT (Press Q to Quit)"
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Initialize Blueprint Manager (NEW)
blueprint_manager = BlueprintManager()

# =============================================
# SETUP OUTPUT FILES (ORIGINAL)
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
        "confidence", "severity",
        "blueprint_comparison", "comparison_score", "defect_reasons"
    ])

# =============================================
# HELPER FUNCTIONS (ORIGINAL + ENHANCED)
# =============================================
def estimate_distance(width_px, known_width_mm=KNOWN_OBJECT_WIDTH_MM, 
                     focal_length=CAMERA_FOCAL_LENGTH_PX):
    if width_px == 0:
        return 0
    distance_mm = (known_width_mm * focal_length) / width_px
    return distance_mm

def calculate_dimensions(w_px, h_px, scale=SCALE_MM_PER_PIXEL):
    w_mm = w_px * scale
    h_mm = h_px * scale
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
                   distance_mm, confidence, severity, color, is_defect=True,
                   blueprint_result=None):
    """Enhanced drawing with blueprint comparison (ENHANCED)"""
    thickness = 3 if severity == "CRITICAL" else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Calculate info box height based on content
    info_height = 135
    if blueprint_result and blueprint_result[0]:
        info_height += 60  # Extra space for blueprint info
    
    cv2.rectangle(frame, (x1, y1 - info_height), (x2, y1), (0, 0, 0), -1)
    cv2.rectangle(frame, (x1, y1 - info_height), (x2, y1), color, 2)
    
    y_offset = y1 - info_height + 20
    
    # Object type indicator
    type_text = "DEFECT" if is_defect else "NON-DEFECT"
    cv2.putText(frame, f"[{type_text}]", 
                (x1 + 5, y_offset), FONT, 0.4, (255, 255, 255), 2)
    y_offset += 20
    
    # Label and confidence
    cv2.putText(frame, f"{label.upper()} ({confidence:.1%})", 
                (x1 + 5, y_offset), FONT, 0.45, (255, 255, 255), 2)
    y_offset += 20
    
    # Dimensions
    cv2.putText(frame, f"W:{dimensions['width_mm']:.1f}mm H:{dimensions['height_mm']:.1f}mm", 
                (x1 + 5, y_offset), FONT, 0.4, (0, 255, 255), 1)
    y_offset += 20
    
    cv2.putText(frame, f"Area:{dimensions['area_mm2']:.1f}mm² Diag:{dimensions['diagonal_mm']:.1f}mm", 
                (x1 + 5, y_offset), FONT, 0.35, (255, 255, 0), 1)
    y_offset += 20
    
    # Distance
    distance_cm = distance_mm / 10
    cv2.putText(frame, f"Distance: {distance_cm:.1f}cm", 
                (x1 + 5, y_offset), FONT, 0.4, (255, 128, 0), 1)
    y_offset += 20
    
    # Blueprint comparison results (NEW)
    if blueprint_result and blueprint_result[0]:
        diff = blueprint_result[0]
        if diff['overall_defect']:
            cv2.putText(frame, "BLUEPRINT: DEFECT FOUND", 
                       (x1 + 5, y_offset), FONT, 0.4, (0, 0, 255), 2)
            y_offset += 18
            # Show first defect reason
            if diff['defect_reasons']:
                reason = diff['defect_reasons'][0][:40]  # Truncate if too long
                cv2.putText(frame, f"  {reason}", 
                           (x1 + 5, y_offset), FONT, 0.3, (255, 100, 100), 1)
        else:
            cv2.putText(frame, f"BLUEPRINT: OK ({diff['structural_similarity']:.0%})", 
                       (x1 + 5, y_offset), FONT, 0.4, (0, 255, 0), 1)
        y_offset += 20
    
    # Severity
    if is_defect:
        cv2.putText(frame, f"Severity: {severity}", 
                    (x1 + 5, y_offset), FONT, 0.5, color, 2)
    else:
        cv2.putText(frame, "OK - Tracking Only", 
                    (x1 + 5, y_offset), FONT, 0.4, (0, 255, 0), 1)

def draw_status(frame, defect_count, fps, total_objects, non_defect_count, blueprint_mode):
    """Enhanced status with blueprint info (ENHANCED)"""
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
    
    # Blueprint indicator
    if blueprint_mode:
        cv2.putText(frame, "[BLUEPRINT MODE]", (10, 55), FONT, 0.5, (255, 255, 0), 2)
    
    # Bottom info bar
    cv2.rectangle(frame, (0, h - 70), (w, h), (40, 40, 40), -1)
    
    # First row
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, h - 45), FONT, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"Total: {total_objects}", (150, h - 45), FONT, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"Defects: {defect_count}", (300, h - 45), FONT, 0.6, (255, 0, 0), 2)
    cv2.putText(frame, f"Non-Defects: {non_defect_count}", (480, h - 45), FONT, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Conf: {DETECTION_CONFIDENCE}", (720, h - 45), FONT, 0.6, (255, 128, 255), 2)
    
    # Second row - Blueprint info
    blueprints = blueprint_manager.list_blueprints()
    if blueprints:
        bp_text = f"Blueprints: {', '.join(blueprints)}"
        cv2.putText(frame, bp_text, (10, h - 15), FONT, 0.5, (255, 255, 0), 1)
    else:
        cv2.putText(frame, "No blueprints captured", (10, h - 15), FONT, 0.5, (128, 128, 128), 1)

def draw_reference_grid(frame, spacing=100):
    """Draw reference grid for visual measurement aid (ORIGINAL)"""
    h, w = frame.shape[:2]
    
    for x in range(0, w, spacing):
        cv2.line(frame, (x, 0), (x, h), (80, 80, 80), 1)
    
    for y in range(0, h, spacing):
        cv2.line(frame, (0, y), (w, y), (80, 80, 80), 1)

# =============================================
# START CAMERA (ORIGINAL)
# =============================================
print("\n" + "="*60)
print("STARTING DETECTION WITH BLUEPRINT SYSTEM")
print("="*60)
print("\nNew Blueprint Controls:")
print("  B - Capture blueprint of currently detected object")
print("  C - Enable/disable blueprint comparison mode")
print("  L - List all captured blueprints")
print("  D - Delete last blueprint")
print("\nOriginal Controls:")
print("  Q/ESC - Quit")
print("  S - Screenshot")
print("  R - Reset statistics")
print("  G - Toggle reference grid")
print("\nHow to use:")
print("  1. Show a GOOD reference object to the camera")
print("  2. Press 'B' to capture it as blueprint")
print("  3. Press 'C' to enable comparison mode")
print("  4. Show other objects - system will detect differences")
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
blueprint_mode = False  # NEW
last_detections = []  # Store last frame's detections for blueprint capture

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1280, 720)

# =============================================
# MAIN LOOP (ENHANCED)
# =============================================
try:
    print("\n✓ Detection started! Press Q to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        if show_grid:
            draw_reference_grid(frame)
        
        if frame_count % 30 == 0:
            fps_end = datetime.now()
            fps = 30 / (fps_end - fps_start).total_seconds()
            fps_start = fps_end
        
        # Run detections (ORIGINAL)
        results = model(frame, conf=DETECTION_CONFIDENCE, iou=IOU_THRESHOLD, verbose=False)
        
        if USE_GENERAL_MODEL:
            general_results = general_model(frame, conf=DETECTION_CONFIDENCE, iou=IOU_THRESHOLD, verbose=False)
        else:
            general_results = []
        
        detections_this_frame = 0
        non_defects_this_frame = 0
        total_objects = 0
        
        all_detections = []
        
        # Process custom model detections
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
        
        # Process general model detections
        if USE_GENERAL_MODEL:
            for r in general_results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    label = general_model.names[cls]
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
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
                                iou = inter_area / (box1_area + box2_area - inter_area)
                                
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
        
        total_objects = len(all_detections)
        last_detections = all_detections.copy()  # Store for blueprint capture
        
        # Draw all detections (ENHANCED WITH BLUEPRINT COMPARISON)
        for detection in all_detections:
            x1, y1, x2, y2 = detection['box']
            label = detection['label']
            confidence = detection['confidence']
            is_defect = detection['is_defect']
            
            w_px = x2 - x1
            h_px = y2 - y1
            area_px = w_px * h_px
            
            dimensions = calculate_dimensions(w_px, h_px)
            distance_mm = estimate_distance(w_px)
            distance_cm = distance_mm / 10
            
            # Blueprint comparison (NEW)
            blueprint_result = None
            comparison_score = None
            defect_reasons = []
            
            if blueprint_mode and blueprint_manager.has_blueprint(label):
                blueprint_result = blueprint_manager.compare_to_blueprint(
                    frame, (x1, y1, x2, y2), label
                )
                if blueprint_result[0]:
                    diff = blueprint_result[0]
                    comparison_score = diff['structural_similarity']
                    defect_reasons = diff['defect_reasons']
                    
                    # Override is_defect if blueprint shows defect
                    if diff['overall_defect']:
                        is_defect = True
            
            # Calculate severity
            if is_defect:
                severity, color = calculate_severity(
                    dimensions['width_mm'], 
                    dimensions['height_mm']
                )
                detections_this_frame += 1
                total_defects_detected += 1
            else:
                severity = "N/A"
                color = (0, 255, 0)
                non_defects_this_frame += 1
                total_non_defects_detected += 1
            
            draw_detection(frame, x1, y1, x2, y2, label, dimensions,
                         distance_mm, confidence, severity, color, is_defect,
                         blueprint_result)
            
            # Save record (ENHANCED WITH BLUEPRINT DATA)
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
                "severity": severity,
                "blueprint_comparison": "Yes" if blueprint_result else "No",
                "comparison_score": round(comparison_score, 3) if comparison_score else None,
                "defect_reasons": "; ".join(defect_reasons) if defect_reasons else None
            }
            records.append(record)
            
            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(record.values())
        
        defect_count = detections_this_frame
        non_defect_count = non_defects_this_frame
        draw_status(frame, defect_count, fps, total_objects, non_defect_count, blueprint_mode)
        cv2.imshow(WINDOW_NAME, frame)
        
        # Handle keys (ENHANCED)
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
        
        # NEW BLUEPRINT CONTROLS
        elif key == ord('b') or key == ord('B'):
            if last_detections:
                print("\nCapturing blueprint...")
                print(f"Found {len(last_detections)} object(s)")
                for i, det in enumerate(last_detections):
                    label = det['label']
                    bbox = det['box']
                    success = blueprint_manager.capture_blueprint(frame, bbox, label)
                    if success:
                        print(f"  [{i+1}] Captured: {label}")
                print("✓ Blueprint capture complete!")
            else:
                print("⚠ No objects detected to capture as blueprint")
        
        elif key == ord('c') or key == ord('C'):
            blueprint_mode = not blueprint_mode
            if blueprint_mode:
                if blueprint_manager.list_blueprints():
                    print(f"✓ Blueprint comparison ENABLED")
                    print(f"  Active blueprints: {', '.join(blueprint_manager.list_blueprints())}")
                else:
                    print("⚠ Blueprint comparison enabled but NO blueprints captured yet!")
                    print("  Press 'B' to capture a blueprint first")
            else:
                print("✓ Blueprint comparison DISABLED")
        
        elif key == ord('l') or key == ord('L'):
            blueprints = blueprint_manager.list_blueprints()
            if blueprints:
                print(f"\nCaptured blueprints ({len(blueprints)}):")
                for bp in blueprints:
                    info = blueprint_manager.blueprints[bp]
                    print(f"  - {bp}: {info['timestamp']} ({info['filename']})")
            else:
                print("\nNo blueprints captured yet. Press 'B' to capture.")
        
        elif key == ord('d') or key == ord('D'):
            blueprints = blueprint_manager.list_blueprints()
            if blueprints:
                last_bp = blueprints[-1]
                blueprint_manager.delete_blueprint(last_bp)
            else:
                print("No blueprints to delete")

except KeyboardInterrupt:
    print("\nInterrupted by user")

except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()

finally:
    cap.release()
    cv2.destroyAllWindows()
    
    # Save summary (ENHANCED WITH BLUEPRINT DATA)
    if records:
        defect_records = [r for r in records if r['is_defect']]
        non_defect_records = [r for r in records if not r['is_defect']]
        
        distances = [r['distance_cm'] for r in records]
        areas = [r['area_mm2'] for r in records]
        
        defect_distances = [r['distance_cm'] for r in defect_records]
        non_defect_distances = [r['distance_cm'] for r in non_defect_records]
        
        # Blueprint statistics
        blueprint_compared = [r for r in records if r['blueprint_comparison'] == 'Yes']
        blueprint_defects = [r for r in blueprint_compared if r['is_defect']]
        
        summary = {
            "session_info": {
                "start_time": timestamp,
                "end_time": datetime.now().isoformat(),
                "total_frames": frame_count,
                "total_objects_detected": len(records),
                "total_defects": len(defect_records),
                "total_non_defects": len(non_defect_records),
                "calibration": {
                    "scale_mm_per_pixel": SCALE_MM_PER_PIXEL,
                    "known_object_width_mm": KNOWN_OBJECT_WIDTH_MM,
                    "camera_focal_length_px": CAMERA_FOCAL_LENGTH_PX
                },
                "blueprint_info": {
                    "blueprints_captured": blueprint_manager.list_blueprints(),
                    "total_blueprint_comparisons": len(blueprint_compared),
                    "blueprint_defects_found": len(blueprint_defects)
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
                "total_non_defects": 0,
                "blueprint_info": {
                    "blueprints_captured": blueprint_manager.list_blueprints()
                }
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
    
    if blueprint_manager.list_blueprints():
        print(f"\nBlueprints captured: {', '.join(blueprint_manager.list_blueprints())}")
    
    print(f"\nResults saved:")
    print(f"  {csv_file}")
    print(f"  {json_file}")
    print(f"  Blueprints folder: blueprints/")
    print("="*60)