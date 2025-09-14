# Test on real images/video
def test_real_world_detection():
    model = YOLO('runs/detect/monkey_detector/weights/best.pt')

    # Test on images
    test_images = ['test1.jpg', 'test2.jpg', 'test3.jpg']

    for img_path in test_images:
        print(f"Testing: {img_path}")

        results = model(img_path)

        # Save annotated image
        for i, result in enumerate(results):
            result.save(f'results_{img_path}')

    # Test on webcam (real-time)
    results = model.predict(source=0, show=True)  # 0 for webcam

    return results
