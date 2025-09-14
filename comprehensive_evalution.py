# Detailed model evaluation
def evaluate_monkey_detector(model_path, test_dataset):
    model = YOLO(model_path)

    # Run validation
    results = model.val(data=test_dataset)

    print("📊 Evaluation Results:")
    print("=" * 30)
    print(f"mAP@0.5: {results.box.map50:.3f}")
    print(f"mAP@0.5:0.95: {results.box.map:.3f}")
    print(f"Precision: {results.box.mp:.3f}")
    print(f"Recall: {results.box.mr:.3f}")
    print(f"F1-Score: {2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr):.3f}")

    return results
