# Advanced training configuration
def train_advanced_monkey_detector():
    model = YOLO('yolov8s.pt')  # Small model (better accuracy)

    # Advanced training parameters
    results = model.train(
        data='monkey_dataset/data.yaml',
        epochs=200,                    # More epochs for better learning
        batch=8,                       # Adjust based on GPU memory
        imgsz=640,                     # Image size
        lr0=0.01,                      # Learning rate
        weight_decay=0.0005,           # Regularization
        mosaic=1.0,                    # Data augmentation
        mixup=0.1,                     # Advanced augmentation
        copy_paste=0.1,                # Copy-paste augmentation
        degrees=10,                    # Rotation augmentation
        translate=0.1,                 # Translation augmentation
        scale=0.5,                     # Scale augmentation
        fliplr=0.5,                    # Horizontal flip probability
        flipud=0.1,                    # Vertical flip probability
        hsv_h=0.015,                   # HSV-Hue augmentation
        hsv_s=0.7,                     # HSV-Saturation augmentation
        hsv_v=0.4,                     # HSV-Value augmentation
        name='advanced_monkey_detector',
        save=True,
        plots=True,
        val=True                       # Validate during training
    )

    return model, results
