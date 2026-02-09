import argparse
import os
import pathlib
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import DogVsCats
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def parse_args():
    desc = "Train a model to classify images of dogs and cats"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-E', '--epochs', type=int, default=10, help='Number of epochs to train the model: Default is 10')
    parser.add_argument('-B', '--batch_size', type=int, default=32, help='Batch size for training: Default is 64')
    parser.add_argument('-L', '--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer: Default is 0.001')
    parser.add_argument('-D', '--data_dir', type=pathlib.Path, default='./dataset', help='Path to the dataset directory: Default is ./dataset')

    return parser.parse_args()


def main(args):
    # Define transformations for the dataset
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize images to 64x64
        # transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
        # transforms.RandomRotation(10),  # Randomly rotate images by up to 10 degrees
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize pixel values
    ])

    # Load the dataset
    train_dataset = datasets.ImageFolder(root=args.data_dir, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Instantiate the model
    model = DogVsCats()

    # Check if GPU is available and move the model to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Directory to save model checkpoints
    checkpoint_dir = 'model_checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Train the model
    for epoch in tqdm(range(args.epochs)):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to GPU

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # if i % 100 == 99:  # Print every 100 mini-batches
            #     print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            #     running_loss = 0.0

        # Save the model checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch + 1}.pth')
        torch.save(model.state_dict(), checkpoint_path)

    print('Finished Training')


if __name__ == '__main__':
    # Call the main function
    args = parse_args()
    main(args)
