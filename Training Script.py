# train_monkey_detector.py
from ultralytics import YOLO
import matplotlib.pyplot as plt

class MonkeyDetectorTrainer:
    def __init__(self):
        print("🐒 Monkey Detection Model Trainer")
        print("=" * 50)

    def train_model(self, dataset_path, epochs=100, batch_size=16):
        """Train YOLOv8 model on monkey dataset"""

        print(f"Starting training with:")
        print(f"📁 Dataset: {dataset_path}")
        print(f"🔄 Epochs: {epochs}")
        print(f"📦 Batch Size: {batch_size}")
        print()

        # Load pre-trained YOLOv8 model
        model = YOLO('yolov8n.pt')  # Nano version for faster training

        # Train the model
        results = model.train(
            data=dataset_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=640,
            name='monkey_detector_v1',
            save=True,
            plots=True
        )

        print("✅ Training completed!")
        return model, results

    def evaluate_model(self, model, test_data):
        """Evaluate trained model performance"""

        print("🔍 Evaluating model performance...")

        # Run validation
        metrics = model.val()

        print(f"📊 Results:")
        print(f"   mAP@0.5: {metrics.box.map50:.3f}")
        print(f"   mAP@0.5:0.95: {metrics.box.map:.3f}")
        print(f"   Precision: {metrics.box.mp:.3f}")
        print(f"   Recall: {metrics.box.mr:.3f}")

        return metrics

    def test_detection(self, model, test_image_path):
        """Test detection on single image"""

        print(f"🧪 Testing detection on: {test_image_path}")

        # Run inference
        results = model(test_image_path)

        # Display results
        for result in results:
            # Plot image with detections
            result.show()

            # Print detections
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    conf = box.conf.item()
                    cls = int(box.cls.item())
                    print(f"   Detected: {model.names[cls]} (confidence: {conf:.3f})")
            else:
                print("   No monkeys detected")

# Usage example
if __name__ == "__main__":
    trainer = MonkeyDetectorTrainer()

    # Train model
    model, results = trainer.train_model(
        dataset_path='monkey_dataset/data.yaml',
        epochs=100,
        batch_size=16
    )

    # Evaluate model
    metrics = trainer.evaluate_model(model, 'monkey_dataset/data.yaml')

    # Test on new image
    trainer.test_detection(model, 'test_monkey_image.jpg')
