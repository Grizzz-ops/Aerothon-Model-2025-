from ultralytics import YOLO
import cv2
import csv
import json
import winsound
from datetime import datetime

# =============================
# LOAD TRAINED MODEL
# =============================
model = YOLO(
    r"C:\Users\Admin\OneDrive\Desktop\metal_defect_detection\runs\detect\train2\weights\best.pt"
)

# =============================
# CSV + JSON SETUP
# =============================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_file = f"inspection_{timestamp}.csv"
json_file = f"inspection_{timestamp}.json"
json_data = []

with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Time", "Defect", "Confidence", "X1", "Y1", "X2", "Y2"])

# =============================
# CAMERA
# =============================
cap = cv2.VideoCapture(0)

print("Press Q or ESC to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO inference (tuned for demo)
    results = model(frame, conf=0.15, imgsz=832)

    defect_count = 0

    for r in results:
        for box in r.boxes:
            defect_count += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

            now = datetime.now().strftime("%H:%M:%S")

            # CSV log
            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([now, label, conf, x1, y1, x2, y2])

            # JSON log
            json_data.append({
                "time": now,
                "defect": label,
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2]
            })

    # =============================
    # OK / DEFECT BANNER
    # =============================
    if defect_count == 0:
        banner_text = "OK - NO DEFECTS"
        banner_color = (0, 200, 0)
    else:
        banner_text = "DEFECT DETECTED"
        banner_color = (0, 0, 255)
        winsound.Beep(1000, 200)

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), banner_color, -1)
    cv2.putText(
        frame,
        banner_text,
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (255, 255, 255),
        3
    )

    # =============================
    # DEFECT COUNT
    # =============================
    cv2.putText(
        frame,
        f"Defect Count: {defect_count}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        2
    )

    cv2.imshow("AI Metal Defect Inspection", frame)

    key = cv2.waitKey(1)
    if key == ord('q') or key == ord('Q') or key == 27:
        break

# =============================
# CLEANUP
# =============================
cap.release()
cv2.destroyAllWindows()

with open(json_file, "w") as f:
    json.dump(json_data, f, indent=4)

print("Inspection completed")
print("CSV saved:", csv_file)
print("JSON saved:", json_file)
