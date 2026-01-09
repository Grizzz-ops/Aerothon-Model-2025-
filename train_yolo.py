from ultralytics import YOLO

print("Starting YOLO training on CPU...")

model = YOLO("yolov8m.pt")

model.train(
    data="dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    device="cpu"   
)
