import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt

def parse_args():
    desc = "Visualize the plot of your training model from a JSON file"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-J', '--json', type=str, default='', help='Path to the model.json file Inside your training folder?')
    parser.add_argument('-M', '--metric', type=str, default='accuracy', help='accuracy or loss?', choices=['accuracy', 'loss'])

    return parser.parse_args()


def main(args):
    model_path = Path(args.json)
    with open(model_path, 'r') as file:
        data = json.load(file)

    # Extract epochs, train_accuracy, and val_accuracy
    epochs = list(data['epochs'].keys())
    train_metric = [data['epochs'][epoch][f'train_{args.metric}'] for epoch in epochs]
    val_metric = [data['epochs'][epoch][f'val_{args.metric}'] for epoch in epochs]

    # Convert epoch strings to integers
    epochs = [int(epoch) for epoch in epochs]

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_metric, label='Train Metric')
    plt.plot(epochs, val_metric, label='Validation Metric')
    plt.xlabel('Epochs')
    plt.ylabel(args.metric.capitalize() + ' Value')
    plt.title('Training, Validation Plot')
    plt.legend()
    plt.xticks(epochs)
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    main(args)