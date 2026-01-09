"""
Smart YOLO Training Script - Only trains if needed
===================================================
This version checks if a trained model exists before training
"""

from ultralytics import YOLO
import os
from pathlib import Path

print("="*60)
print("SMART YOLO TRAINING SCRIPT")
print("="*60)

# =============================================
# CHECK IF MODEL ALREADY EXISTS
# =============================================
POSSIBLE_MODEL_PATHS = [
    "runs/detect/train_fixed/weights/best.pt",
    "runs/detect/train_fixed/weights/last.pt",
    "runs/detect/train2/weights/best.pt",
    "best.pt",
    "last.pt",
]

existing_model = None
for path in POSSIBLE_MODEL_PATHS:
    if os.path.exists(path):
        existing_model = path
        break

if existing_model:
    print(f"\n✓ Found existing trained model at: {existing_model}")
    print("\nOptions:")
    print("  1. Use existing model (skip training)")
    print("  2. Train new model (will take time)")
    print("  3. Continue training existing model")
    
    choice = input("\nEnter your choice (1/2/3): ").strip()
    
    if choice == "1":
        print("\n✓ Using existing model - no training needed!")
        print(f"\nTo use this model for detection:")
        print(f"  python detect_live_FIXED.py")
        print("\nOr update MODEL_PATH in your detection script to:")
        print(f"  MODEL_PATH = r'{existing_model}'")
        exit(0)
    
    elif choice == "3":
        print("\n→ Continuing training from existing model...")
        RESUME_TRAINING = True
        PRETRAINED_MODEL = existing_model
    
    else:
        print("\n→ Starting fresh training...")
        RESUME_TRAINING = False
        PRETRAINED_MODEL = "yolov8n.pt"
else:
    print("\nNo existing model found - starting fresh training...")
    RESUME_TRAINING = False
    PRETRAINED_MODEL = "yolov8n.pt"

# =============================================
# VERIFY DATASET
# =============================================
print("\n" + "="*60)
print("CHECKING DATASET STRUCTURE")
print("="*60)

required_paths = [
    "dataset.yaml",
    "data/images/train",
    "data/images/val",
    "data/labels/train",
    "data/labels/val"
]

# Check for augmented data first
if os.path.exists("data/images/train_augmented"):
    print("\n✓ Found augmented training data!")
    print("  Using: data/images/train_augmented")
    required_paths[1] = "data/images/train_augmented"
    required_paths[3] = "data/labels/train_augmented"

missing_paths = []
for path in required_paths:
    if not os.path.exists(path):
        missing_paths.append(path)
        print(f"  ❌ MISSING: {path}")
    else:
        print(f"  ✓ Found: {path}")

if missing_paths:
    print("\n⚠️  ERROR: Missing required paths!")
    print("\nPlease ensure your dataset structure is correct.")
    print("Run this first if you haven't:")
    print("  python augment_dataset_fixed.py --input data/images/train --output data/images/train_augmented")
    exit(1)

# Count images
train_path = "data/images/train_augmented" if os.path.exists("data/images/train_augmented") else "data/images/train"
train_images = len(list(Path(train_path).glob("*.jpg"))) + \
               len(list(Path(train_path).glob("*.png")))
val_images = len(list(Path("data/images/val").glob("*.jpg"))) + \
             len(list(Path("data/images/val").glob("*.png")))

print(f"\nDataset Statistics:")
print(f"  Training images: {train_images}")
print(f"  Validation images: {val_images}")

if train_images == 0:
    print("\n❌ ERROR: No training images found!")
    exit(1)

if val_images == 0:
    print("\n⚠️  WARNING: No validation images found!")
    response = input("Continue anyway? (y/n): ").strip().lower()
    if response != 'y':
        exit(1)

# =============================================
# TRAINING CONFIGURATION
# =============================================
print("\n" + "="*60)
print("TRAINING CONFIGURATION")
print("="*60)

print(f"\nModel: YOLOv8n (nano - optimized for CPU)")
print(f"Device: CPU")
print(f"Epochs: 100")
print(f"Batch size: 4")
print(f"Image size: 640")
print(f"Early stopping: 20 epochs patience")

if RESUME_TRAINING:
    print(f"\n→ Resuming from: {PRETRAINED_MODEL}")
else:
    print(f"\n→ Starting with pretrained: {PRETRAINED_MODEL}")

response = input("\nStart training? (y/n): ").strip().lower()
if response != 'y':
    print("Training cancelled.")
    exit(0)

# =============================================
# LOAD MODEL
# =============================================
print(f"\nLoading model...")

try:
    if RESUME_TRAINING:
        model = YOLO(PRETRAINED_MODEL)
        print(f"✓ Loaded existing model for continued training")
    else:
        model = YOLO("yolov8n.pt")
        print(f"✓ Loaded pretrained YOLOv8n model")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit(1)

# =============================================
# START TRAINING
# =============================================
print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)
print("\nThis will take a while...")
print("Press Ctrl+C to stop early (model will be saved)")
print("\n")

try:
    results = model.train(
        data="dataset.yaml",
        epochs=100,
        imgsz=640,
        batch=4,
        device="cpu",
        patience=20,
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
        val=True,
        project="runs/detect",
        name="train_fixed",
        exist_ok=True,
        pretrained=not RESUME_TRAINING,
        optimizer='Adam',
        lr0=0.001,
        cos_lr=True,
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        resume=RESUME_TRAINING,  # Resume if continuing
    )
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    best_model_path = "runs/detect/train_fixed/weights/best.pt"
    last_model_path = "runs/detect/train_fixed/weights/last.pt"
    
    print(f"\nTrained models saved:")
    print(f"  Best model: {best_model_path}")
    print(f"  Last model: {last_model_path}")
    
    print(f"\nTraining results:")
    print(f"  Results directory: runs/detect/train_fixed/")
    print(f"  Check these files:")
    print(f"    - results.png (training metrics)")
    print(f"    - confusion_matrix.png")
    print(f"    - val_batch0_pred.jpg (predictions)")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("\n1. Review training results in: runs/detect/train_fixed/")
    print("\n2. Test the model:")
    print("     python detect_live_FIXED.py")
    print("\n3. If results are poor, try:")
    print("     - More training data")
    print("     - More augmentation")
    print("     - Different confidence threshold")
    print("="*60)

except KeyboardInterrupt:
    print("\n\n⚠️  Training interrupted by user!")
    print("Partial model may be saved in: runs/detect/train_fixed/weights/last.pt")
    
except Exception as e:
    print(f"\n❌ Training failed with error:")
    print(f"   {e}")
    import traceback
    traceback.print_exc()
    exit(1)