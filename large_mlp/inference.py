import argparse
import pathlib
import torch
import os
import time

import torch.nn as nn
import torch.nn.utils.prune as tprune

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from zeus.monitor import ZeusMonitor
from model import SetMLP

input_size = 28 * 28
hidden_size = 16
output_size = 10
batch_size = 4

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_args():
    desc = "PyTorch Model Inference"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-M', '--model_path', type=pathlib.Path, default='./model_checkpoints', help='Path to the model checkpoints directory: Default is ./model_checkpoints')
    parser.add_argument('-D', '--device', choices=['cpu', 'cuda'], default='cuda', help='Device to use: cpu or cuda')

    return parser.parse_args()


def remove_pruning(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if hasattr(module, 'weight_mask'):
                print(f"Removing pruning from {name}")
                tprune.remove(module, 'weight')


def clean_state_dict(state_dict):
    new_state_dict = {}
    for key in state_dict:
        # If we encounter 'weight_orig', apply the mask
        if 'weight_orig' in key:
            mask_key = key.replace('weight_orig', 'weight_mask')
            if mask_key in state_dict:
                # Apply the mask to the original weights
                pruned_weight = state_dict[key] * state_dict[mask_key]
                new_key = key.replace('weight_orig', 'weight')
                new_state_dict[new_key] = pruned_weight
        # Skip 'weight_mask' keys
        elif 'weight_mask' not in key:
            new_state_dict[key] = state_dict[key]
    return new_state_dict


def main(args):
    # Check if GPU is available
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(args.device)
    print(f'Using device: {device}')

    # Load MNIST dataset
    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST(root='./dataset', train=True, download=True, transform=transform)
    monitor = ZeusMonitor(gpu_indices=[0])

    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = int((len(dataset) - train_size) / 2)
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initialize the model, loss function, and optimizer
    model = SetMLP(input_size, output_size).to(device)
    state_dict = torch.load(args.model_path)
    model.load_state_dict(clean_state_dict(state_dict))

    model.eval()
    correct_test = 0
    total_test = 0
    all_labels_test = []
    all_predictions_test = []

    total_inference_time = 0
    total_inference_energy = 0
    num_batches = 0

    with torch.no_grad():  # No need to calculate gradients during testing
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.view(-1, input_size)

            monitor.begin_window("inference")
            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()
            inference_measurement = monitor.end_window("inference")

            batch_inference_time = end_time - start_time
            total_inference_time += batch_inference_time
            total_inference_energy += inference_measurement.total_energy
            num_batches += 1

            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

            # Store labels and predictions for metrics calculation
            all_labels_test.extend(labels.cpu().numpy())
            all_predictions_test.extend(predicted.cpu().numpy())

    # Calculate average inference time per batch
    avg_inference_time_per_batch = total_inference_time / num_batches
    avg_inference_energy_per_batch = total_inference_energy / num_batches

    test_accuracy = 100 * correct_test / total_test

    # Calculate additional metrics for test data
    test_precision = precision_score(all_labels_test, all_predictions_test, average='macro', zero_division=0)
    test_recall = recall_score(all_labels_test, all_predictions_test, average='macro')
    test_f1 = f1_score(all_labels_test, all_predictions_test, average='macro')

    print('=====================================')
    print('Test Results:')
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    print(f'Test Precision: {test_precision:.4f}')
    print(f'Test Recall: {test_recall:.4f}')
    print(f'Test F1 Score: {test_f1:.4f}')
    print(f'Total Inference Time: {total_inference_time:.6f} seconds')
    print(f'Total Inference Energy: {total_inference_energy:.6f} J')
    print(f'Total Number of Batches: {num_batches}')
    print(f'Average Inference Time per Batch: {avg_inference_time_per_batch:.6f} seconds')
    print(f'Average Inference Energy per Batch: {avg_inference_energy_per_batch:.6f} J')
    print('=====================================')



if __name__ == '__main__':
    # Call the main function
    args = parse_args()
    main(args)