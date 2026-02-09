import argparse
import pathlib
import torch
import os
import time
import medmnist

import torch.nn as nn
import torch.nn.utils.prune as tprune

from medmnist import INFO
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from zeus.monitor import ZeusMonitor
from model import SetMLP

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
batch_size = 4


def parse_args():
    desc = "PyTorch Model Inference"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-M', '--model_path', type=pathlib.Path, default='./model_checkpoints', help='Path to the model checkpoints directory: Default is ./model_checkpoints')
    parser.add_argument('-K', '--dataset', type=str, default='bloodmnist', help='Choose DataSet Name From MedMNIST Dataset', choices=['bloodmnist', 'breastmnist', 'chestmnist', 'dermamnist', 'octmnist', 'organamnist', 'organcmnist', 'organsmnist', 'pathmnist', 'pneumoniamnist', 'retinamnist', 'tissuemnist'])
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

    # Extract the dataset and its metadata
    info = INFO[args.dataset]
    task, in_channel, num_classes = info['task'], info['n_channels'], len(info['label'])
    DataClass = getattr(medmnist, info['python_class'])
    input_size = 420 * 420 * 1

    total_padding = max(0, 224 - 28)
    padding_left, padding_top = total_padding // 2, total_padding // 2
    padding_right, padding_bottom = total_padding - padding_left, total_padding - padding_top

    # Load MNIST dataset
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomRotation(10),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.Pad((padding_left, padding_top, padding_right, padding_bottom), fill=0, padding_mode='constant')
    ])

    test_dataset = DataClass(split='test', transform=data_transform, download=True, as_rgb=False, size=224, root='/data/users/adouzandehzenoozi/Datasets')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    monitor = ZeusMonitor(gpu_indices=[0])

    # Initialize the model, loss function, and optimizer
    model = SetMLP(input_size, num_classes).to(device)
    state_dict = torch.load(args.model_path)
    model.load_state_dict(clean_state_dict(state_dict))

    model.eval()
    correct_test = 0
    total_test = 0
    all_labels_test = []
    all_predictions_test = []


    zeus_result_list = []
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
            zeus_result_list.append(inference_measurement)
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

    # Calculate the total energy and time by Zeus
    total_energy_by_zeus = sum(map(lambda m: m.total_energy, zeus_result_list))
    total_time_by_zeus = sum(map(lambda m: m.time, zeus_result_list))
    avg_energy_by_zeus = total_energy_by_zeus / len(zeus_result_list)
    avg_time_by_zeus = total_time_by_zeus / len(zeus_result_list)

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
    print(f'Total Energy by Zeus: {total_energy_by_zeus:.6f} J')
    print(f'Total Time by Zeus: {total_time_by_zeus:.6f} seconds')
    print(f'Average Energy by Zeus: {avg_energy_by_zeus:.6f} J')
    print(f'Average Time by Zeus: {avg_time_by_zeus:.6f} seconds')
    print('=====================================')



if __name__ == '__main__':
    # Call the main function
    args = parse_args()
    main(args)