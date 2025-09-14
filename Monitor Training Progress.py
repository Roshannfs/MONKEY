# Real-time training monitoring
import matplotlib.pyplot as plt
import pandas as pd

def plot_training_results(results_folder):
    """Plot training metrics"""

    # Load training results
    results_csv = f"{results_folder}/results.csv"
    df = pd.read_csv(results_csv)

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Plot loss
    axes[0,0].plot(df['epoch'], df['train/box_loss'], label='Train')
    axes[0,0].plot(df['epoch'], df['val/box_loss'], label='Validation')
    axes[0,0].set_title('Box Loss')
    axes[0,0].legend()

    # Plot mAP
    axes[0,1].plot(df['epoch'], df['metrics/mAP50'])
    axes[0,1].set_title('mAP@0.5')

    # Plot precision/recall
    axes[1,0].plot(df['epoch'], df['metrics/precision'])
    axes[1,0].set_title('Precision')

    axes[1,1].plot(df['epoch'], df['metrics/recall'])
    axes[1,1].set_title('Recall')

    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.show()

# Monitor training
plot_training_results('runs/detect/monkey_detector')
