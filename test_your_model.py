
"""
Quick Test Script for Your Trained Monkey Detection Model
Run this after training completes
"""

from ultralytics import YOLO
import cv2
import os

def test_trained_model():
    print("🧪 Testing Your Trained Monkey Detection Model")
    print("="*50)

    # Check if model exists
    model_path = r'C:\m\runs\detect\monkey_detector_v1\weights\best.pt'

    if not os.path.exists(model_path):
        print("❌ Model not found!")
        print(f"Looking for: {model_path}")
        print("Make sure training completed successfully")
        return

    # Load your trained model
    print("📥 Loading your custom trained model...")
    model = YOLO(model_path)
    print("✅ Model loaded successfully!")

    # Test on webcam
    print("🎥 Starting webcam test...")
    print("Press 'q' to quit, 's' to save detection")

    cap = cv2.VideoCapture(0)
    detection_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to access webcam")
            break

        # Run your AI detection
        results = model(frame, conf=0.5)  # 50% confidence threshold

        # Draw results
        annotated_frame = results[0].plot()

        # Count detections
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            detection_count += 1
            print(f"🐒 Monkey detected! (Detection #{detection_count})")

            # Show detection details
            for box in boxes:
                conf = box.conf.item()
                print(f"   Confidence: {conf:.3f}")

        # Display
        cv2.imshow('Your Trained Monkey Detector - Press q to quit', annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame
            filename = f'detection_result_{detection_count}.jpg'
            cv2.imwrite(filename, annotated_frame)
            print(f"💾 Saved: {filename}")

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n🎉 Test completed!")
    print(f"Total detections: {detection_count}")
    print("Your AI model is working! 🚀")

if __name__ == "__main__":
    test_trained_model()
