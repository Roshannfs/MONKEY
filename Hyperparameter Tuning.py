# Automated hyperparameter tuning
def tune_hyperparameters():
    model = YOLO('yolov8n.pt')

    # Define search space
    search_space = {
        'lr0': [0.001, 0.01, 0.1],
        'batch': [8, 16, 32],
        'epochs': [50, 100, 200]
    }

    best_map = 0
    best_params = {}

    for lr in search_space['lr0']:
        for batch in search_space['batch']:
            for epochs in search_space['epochs']:
                print(f"Testing: lr={lr}, batch={batch}, epochs={epochs}")

                results = model.train(
                    data='monkey_dataset/data.yaml',
                    lr0=lr,
                    batch=batch,
                    epochs=epochs,
                    name=f'tune_lr{lr}_b{batch}_e{epochs}'
                )

                # Get validation mAP
                map_score = results.results_dict['metrics/mAP50']

                if map_score > best_map:
                    best_map = map_score
                    best_params = {'lr0': lr, 'batch': batch, 'epochs': epochs}

    print(f"Best parameters: {best_params}")
    print(f"Best mAP: {best_map:.3f}")

    return best_params
