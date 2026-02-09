import argparse
import os
import pathlib
import torch
import json
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.multiprocessing import freeze_support
from model import MLP
from early_stopping import EarlyStopping

# Set random seed for reproducibility
torch.manual_seed(42)

# Hyperparameters
input_size = 28 * 28
hidden_size = 64
output_size = 10
learning_rate = 0.001
batch_size = 4
epochs = 5


def parse_args():
    desc = "Train a model to classify images of dogs and cats"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-E', '--epochs', type=int, default=10, help='Number of epochs to train the model: Default is 10')
    parser.add_argument('-B', '--batch_size', type=int, default=4, help='Batch size for training: Default is 64')
    parser.add_argument('-L', '--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer: Default is 0.001')
    parser.add_argument('-D', '--data_dir', type=pathlib.Path, default='./dataset', help='Path to the dataset directory: Default is ./dataset')

    return parser.parse_args()


def generate_metrics_plot(train, val, target_directory, filename):
    epochs_range = range(1, len(train) + 1)
    plt.figure(figsize=(10, 5))

    plt.plot(epochs_range, train, 'r', label='Training')
    plt.plot(epochs_range, val, 'b', label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Training, Validation Plot')

    plt.tight_layout()
    plt.savefig(os.path.join(target_directory, filename))
    plt.show()


def main(args):
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load MNIST dataset
    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.FashionMNIST(root=args.data_dir, train=True, download=True, transform=transform)

    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Initialize the model, loss function, and optimizer
    model = MLP(input_size, hidden_size, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

    # Directory to save model checkpoints
    checkpoint_dir = 'model_checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training information
    training_info = {
        'total_epochs': args.epochs,
        'epochs': {},
    }

    # Early stopping object
    early_stopping = EarlyStopping(patience=5, min_delta=0.01)
    current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    target_directory = os.path.join(checkpoint_dir, current_date)
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.view(-1, input_size)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)

        # Validation loop
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.view(-1, input_size)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        val_accuracy = 100 * correct_val / total_val
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch {epoch + 1}/{args.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%')

        # Save the model checkpoint
        checkpoint_path = os.path.join(target_directory, f'model_epoch_{epoch + 1}.pth')
        torch.save(model.state_dict(), checkpoint_path)

        # Store the training information
        training_info['epochs'][epoch] = {
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy
        }

        # Early stopping
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("Early stopping in epoch: ", epoch)
            # break

    with open(os.path.join(target_directory, 'model.json'), 'w') as file:
        json.dump(training_info, file)

    # Save the training plot metrics
    generate_metrics_plot(train_losses, val_losses, target_directory, 'training_loss_plot.png')
    generate_metrics_plot(train_accuracies, val_accuracies, target_directory, 'training_accuracy_plot.png')


if __name__ == '__main__':
    # Call the main function
    args = parse_args()

    # Add the freeze_support() call for Windows
    freeze_support()
    
    # Call the main function
    main(args)
