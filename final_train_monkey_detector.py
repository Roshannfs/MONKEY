#!/usr/bin/env python3
"""
Simple Monkey Detection Model Training Script
"""

def train_monkey_detector():
    print("🐒 Monkey Detection Model Training")
    print("=" * 50)

    try:
        from ultralytics import YOLO

        # Load base model
        print("📥 Loading YOLOv8 base model...")
        model = YOLO('yolov8n.pt')

        # Check if dataset exists
        import os
        if not os.path.exists('monkey_dataset/data.yaml'):
            print("❌ Dataset not found!")
            print("Run: python setup_monkey_ai_project.py first")
            return

        print("🚀 Starting training...")
        print("This will take time depending on your dataset size and hardware")

        # Start training
        results = model.train(
            data='monkey_dataset/data.yaml',
            epochs=100,
            batch=16,
            imgsz=640,
            name='monkey_detector_v1',
            save=True,
            plots=True
        )

        print("\n✅ Training completed!")
        print("📁 Results saved in: runs/detect/monkey_detector_v1")
        print("🧠 Best model: runs/detect/monkey_detector_v1/weights/best.pt")

        return results

    except ImportError:
        print("❌ ultralytics package not installed")
        print("Run: pip install ultralytics")
        return None
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None

if __name__ == "__main__":
    train_monkey_detector()
