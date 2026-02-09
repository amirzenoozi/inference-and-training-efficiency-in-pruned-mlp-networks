import argparse
import os
import pathlib
import torch
import json
import time
import torchprofile

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.utils.prune as tprune

from codecarbon import EmissionsTracker
from datetime import datetime
from zeus.monitor import ZeusMonitor
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.multiprocessing import freeze_support
from model import DynamicSparseMLP
from erdos_renyi_pruning import ErdosRenyiPruningMethod
from early_stopping import EarlyStopping

# Set random seed for reproducibility
torch.manual_seed(42)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_args():
    desc = "Train a model to classify images of dogs and cats"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-E', '--epochs', type=int, default=10, help='Number of epochs to train the model: Default is 10')
    parser.add_argument('-B', '--batch_size', type=int, default=4, help='Batch size for training: Default is 64')
    parser.add_argument('-L', '--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer: Default is 0.001')
    parser.add_argument('-D', '--data_dir', type=pathlib.Path, default='./dataset', help='Path to the dataset directory: Default is ./dataset')
    parser.add_argument('-P', '--pruning_type', type=str, default='before', help='Type of pruning?', choices=['before', 'after', 'manual', 'none', 'erdos'])
    parser.add_argument('-K', '--dataset', type=str, default='mnist', help='Choose DataSet Name', choices=['mnist', 'fmnist', 'emnist', 'cifar10'])
    parser.add_argument('-S', '--sparsity', type=float, default=0.5, help='Sparsity level for pruning: Default is 0.5')
    parser.add_argument('-M', '--model_dir', type=pathlib.Path, default='./model_checkpoints', help='Path to the model checkpoints directory: Default is ./model_checkpoints')
    parser.add_argument('-W', '--layer_wise', type=int, default=0, help='Apply pruning layer-wise: Default is 0', choices=[0, 1])

    return parser.parse_args()


def modify_weights_mocanu(model, sparsity, prune_percentage=0.2, filter_layers=False):
    layer_names = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    if bool(filter_layers):
        ignored_layers_name = [layer_names[0], layer_names[-1]]
    else:
        ignored_layers_name = []

    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if name not in ignored_layers_name:
                    if hasattr(module, 'weight_mask') and hasattr(module, 'weight_orig'):
                        # Access weight_mask and weight_orig
                        weight_mask = module.weight_mask
                        weight_orig = module.weight_orig
                        weights = module.weight

                        # Get the device of the weight tensor
                        device = weights.device

                        not_pruned_indices = (weight_mask == 1).nonzero(as_tuple=True)
                        not_pruned_weights = weight_orig[not_pruned_indices]

                        num_to_prune = int(not_pruned_weights.numel() * prune_percentage)
                        positive_num = num_to_prune // 2
                        negative_num = num_to_prune - positive_num

                        if num_to_prune > 0:
                            positive_weights = not_pruned_weights[not_pruned_weights > 0]
                            if positive_weights.numel() > 0:
                                positive_num = min(positive_num, positive_weights.numel())
                                pos_indices = (not_pruned_weights > 0).nonzero(as_tuple=True)[0].to(device)
                                smallest_pos_indices = pos_indices[torch.topk(positive_weights, positive_num, largest=False).indices].to(device)
                            else:
                                smallest_pos_indices = torch.tensor([], dtype=torch.long, device=device)

                            negative_weights = not_pruned_weights[not_pruned_weights < 0]
                            if negative_weights.numel() > 0:
                                negative_num = min(negative_num, negative_weights.numel())
                                neg_indices = (not_pruned_weights < 0).nonzero(as_tuple=True)[0].to(device)
                                largest_neg_indices = neg_indices[torch.topk(negative_weights, negative_num, largest=True).indices].to(device)
                            else:
                                largest_neg_indices = torch.tensor([], dtype=torch.long, device=device)

                        random_positive_weights = torch.randn(positive_num, device=device)
                        random_negative_weights = -torch.randn(negative_num, device=device)

                        prune_indices = torch.cat((smallest_pos_indices, largest_neg_indices))
                        random_weights = torch.cat((random_positive_weights, random_negative_weights))

                        if prune_indices.numel() > 0:
                            row_indices = not_pruned_indices[0][prune_indices]
                            col_indices = not_pruned_indices[1][prune_indices]

                            # Remove the smallest weights that we found
                            weight_mask.index_put_((row_indices, col_indices), torch.zeros_like(row_indices, dtype=weight_mask.dtype, device=device))

                        # Generate new connections
                        zero_mask_indices = (weight_mask == 0).nonzero(as_tuple=True)
                        if len(zero_mask_indices[0]) >= num_to_prune:
                            random_zero_indices = torch.randperm(len(zero_mask_indices[0]))[:num_to_prune]
                            selected_zero_mask_indices = (zero_mask_indices[0][random_zero_indices], zero_mask_indices[1][random_zero_indices])
                            weight_mask.index_put_(selected_zero_mask_indices, torch.ones(num_to_prune, dtype=weight_mask.dtype, device=device))


                        # if name == 'fc3':
                    #     print(positive_num, negative_num)
                    #     print(positive_weights)
                    #     print(negative_weights)
                    #     print(weight_orig)
                    #     print(row_indices, col_indices)


def apply_pruning(model, amount, filter_layers=False):
    layer_names = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    if bool(filter_layers):
        ignored_layers_name = [layer_names[0], layer_names[-1]]
    else:
        ignored_layers_name = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if name not in ignored_layers_name:
                tprune.l1_unstructured(module, name='weight', amount=amount)


def erdos_renyi_pruning(model, amount, filter_layers=False):
    """
    Apply Erdos-Renyi based pruning to a tensor in a module.

    Args:
        module (nn.Module): The module containing the tensor to prune.
        p (float): The target sparsity level, a probability of pruning a given connection.
    """
    layer_names = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    if bool(filter_layers):
        ignored_layers_name = [layer_names[0], layer_names[-1]]
    else:
        ignored_layers_name = []

    name = 'weight'
    with torch.no_grad():
        for layer_name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if layer_name not in ignored_layers_name:
                    ErdosRenyiPruningMethod.apply(module, name=name, p=amount)
                    # Make sure the `weight` attribute exists after pruning
                    setattr(module, name, module._parameters[name + '_orig'] * module._buffers[name + '_mask'])
    return module


def apply_random_pruning(model, amount, filter_layers=False):
    layer_names = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    if bool(filter_layers):
        ignored_layers_name = [layer_names[0], layer_names[-1]]
    else:
        ignored_layers_name = []
    # Apply Erdős–Rényi random graph initialization
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if name not in ignored_layers_name:
                tprune.random_unstructured(module, name='weight', amount=amount)


def remove_pruning(model, filter_layers=False):
    layer_names = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    if bool(filter_layers):
        ignored_layers_name = [layer_names[0], layer_names[-1]]
    else:
        ignored_layers_name = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if name not in ignored_layers_name:
                tprune.remove(module, 'weight')


def clean_state_dict_last_version(state_dict):
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


def clean_state_dict(state_dict):
    new_state_dict = {}
    for key in state_dict:
        # Replace 'weight_orig' with 'weight'
        if 'weight_orig' in key:
            new_key = key.replace('weight_orig', 'weight')
            new_state_dict[new_key] = state_dict[key]
        # Skip 'weight_mask' keys
        elif 'weight_mask' not in key:
            new_state_dict[key] = state_dict[key]
    return new_state_dict


def generate_metrics_plot(train, val, target_directory, filename, early_stopping_epoch=None):
    epochs_range = range(1, len(train) + 1)
    plt.figure(figsize=(10, 5))

    plt.plot(epochs_range, train, 'r', label='Training')
    plt.plot(epochs_range, val, 'b', label='Validation')

    plt.xlabel('Epochs')
    plt.ylabel('Value')

    plt.legend()
    plt.title('Training, Validation Plot')
    plt.xticks(epochs_range)
    plt.grid(True)

    # Highlight the early stopping epoch if provided
    if early_stopping_epoch is not None:
        # Add a vertical line at the early stopping epoch
        plt.axvline(x=early_stopping_epoch, color='g', linestyle='--', label='Early Stopping')

        # Annotate the early stopping epoch with an arrow and text
        plt.annotate('Early Stopping\nEpoch: {}'.format(early_stopping_epoch),
                     xy=(early_stopping_epoch, val[early_stopping_epoch - 1]),
                     xytext=(early_stopping_epoch + 1, max(train + val) * 0.9),
                     arrowprops=dict(facecolor='green', shrink=0.05),
                     horizontalalignment='left',
                     verticalalignment='top')

    plt.tight_layout()
    plt.savefig(os.path.join(target_directory, filename), format='jpg', bbox_inches='tight')
    plt.savefig(os.path.join(target_directory, filename), format='eps', bbox_inches='tight')
    plt.savefig(os.path.join(target_directory, filename), format='pdf', bbox_inches='tight')
    # plt.show()


def count_nonzero_weights(model):
    total_params = 0
    nonzero_params = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if hasattr(module, 'weight_mask') and hasattr(module, 'weight_orig'):
                nonzero_params += torch.sum(module.weight_mask == 0).item()
                total_params += module.weight_mask.numel()
    return nonzero_params, total_params


def count_flops(model, inputs):
    # This function will simulate FLOPs measurement
    # It should ideally track the operations during the forward pass
    # We will use torchprofile or similar library to get this information
    flops = torchprofile.profile_macs(model, inputs)
    return flops


def main(args):
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load dataset
    if args.dataset != 'cifar10':
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    if args.dataset == 'mnist':
        dataset = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform)
    elif args.dataset == 'fmnist':
        dataset = datasets.FashionMNIST(root=args.data_dir, train=True, download=True, transform=transform)
    elif args.dataset == 'emnist':
        dataset = datasets.EMNIST(root=args.data_dir, split='balanced', train=True, download=True, transform=transform)
        # dataset = datasets.EMNIST(root=args.data_dir, split='byclass', train=True, download=True, transform=transform)
    elif args.dataset == 'cifar10':
        dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)

    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = int((len(dataset) - train_size) / 2)
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Initialize the model, loss function, and optimizer
    input_size = {
        'mnist': 28 * 28,
        'fmnist': 28 * 28,
        'emnist': 28 * 28,
        'cifar10': 32 * 32 * 3,
    }
    hidden_size = 16
    output_size = {
        'mnist': 10,
        'fmnist': 10,
        'emnist': 47,
        'cifar10': 10,
    }
    model = DynamicSparseMLP(input_size[args.dataset], hidden_size, output_size[args.dataset]).to(device)
    monitor = ZeusMonitor(gpu_indices=[0])

    if args.pruning_type == 'before':
        apply_pruning(model, args.sparsity, args.layer_wise)
    elif args.pruning_type == 'manual':
        apply_random_pruning(model, args.sparsity, args.layer_wise)
    elif args.pruning_type == 'erdos':
        erdos_renyi_pruning(model, args.sparsity, args.layer_wise)

    # debug: Number of non-zero weights before training
    # nonzero, total = count_nonzero_weights(model)
    # print(f'Before training: {nonzero}/{total} ({nonzero / total:.2%} non-zero)')
    # return False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

    # Directory to save model checkpoints
    checkpoint_dir = args.model_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    pruning_names = {
        'before': 'PreTraining',
        'none': 'None',
        'manual': 'SetMethod',
        'erdos': 'ErdosRenyi',
        'after': 'PostTraining',
    }
    dataset_names = {
        'mnist': 'MNIST',
        'fmnist': 'FashionMNIST',
        'emnist': 'EMNIST',
        'cifar10': 'CIFAR10',
    }
    current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    target_folder_name = f'Dataset-{dataset_names[args.dataset]}__Sparsity-{int(args.sparsity * 100)}%__Pruning-{pruning_names[args.pruning_type]}__{current_date}'
    target_directory = os.path.join(checkpoint_dir, target_folder_name)

    # Training information
    training_info = {
        'total_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'pruning_type': args.pruning_type,
        'final_sparsity': args.sparsity,
        'input_size': input_size[args.dataset],
        'hidden_size': hidden_size,
        'output_size': output_size[args.dataset],
        'total_energy': '',
        'dataset': args.dataset,
        'stopped_at': None,
        'training_duration': None,
        'max_memory_allocated': 0,
        'test_results': {},
        'epochs': {},
    }

    # Early stopping object
    early_stopping = EarlyStopping(patience=10, min_delta=0, save_path=os.path.join(target_directory, 'best_model.pth'))

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_precisions = []
    val_precisions = []
    train_recalls = []
    val_recalls = []
    train_f1s = []
    val_f1s = []
    epoch_memory_allocated = []

    # FLOPs tracking
    train_flps = []
    val_flps = []

    # Training loop
    epoch_flops = 0
    total_training_time = 0
    total_training_energy = 0
    total_training_co2 = 0
    monitor.begin_window("training")
    training_tracker = EmissionsTracker(save_to_file=False, allow_multiple_runs=True)
    for epoch in range(args.epochs):
        monitor.begin_window("epoch")
        training_tracker.start()
        torch.cuda.reset_peak_memory_stats()
        epoch_start_time = time.time()
        model.train()

        # Calculate current sparsity level based on schedule
        if args.pruning_type == 'manual' or args.pruning_type == 'erdos':
            modify_weights_mocanu(model, args.sparsity, 0.2, args.layer_wise)

        total_loss = 0
        correct_train = 0
        total_train = 0
        all_labels_train = []
        all_predictions_train = []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.view(-1, input_size[args.dataset])
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Measure FLOPs for the forward pass
            flops = count_flops(model, inputs)
            epoch_flops += flops

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # Store labels and predictions for metrics calculation
            all_labels_train.extend(labels.cpu().numpy())
            all_predictions_train.extend(predicted.cpu().numpy())

        # End timer for the current epoch
        epoch_training_time = time.time() - epoch_start_time
        total_training_time += epoch_training_time
        train_flps.append(epoch_flops)

        # End energy for the current epoch
        measurement = monitor.end_window("epoch")
        total_training_co2 += training_tracker.stop()
        total_training_energy += measurement.total_energy

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)

        # Calculate additional metrics for training data
        train_precision = precision_score(all_labels_train, all_predictions_train, average='macro', zero_division=0)
        train_precisions.append(train_precision)
        train_recall = recall_score(all_labels_train, all_predictions_train, average='macro')
        train_recalls.append(train_recall)
        train_f1 = f1_score(all_labels_train, all_predictions_train, average='macro')
        train_f1s.append(train_f1)

        # Validation loop
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        all_labels_val = []
        all_predictions_val = []
        epoch_val_flops = 0

        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.view(-1, input_size[args.dataset])

            # Measure FLOPs for the forward pass
            flops = count_flops(model, inputs)
            epoch_val_flops += flops

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

            # Store labels and predictions for metrics calculation
            all_labels_val.extend(labels.cpu().numpy())
            all_predictions_val.extend(predicted.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_flps.append(epoch_val_flops)

        val_accuracy = 100 * correct_val / total_val
        val_accuracies.append(val_accuracy)

        # Calculate additional metrics for validation data
        val_precision = precision_score(all_labels_val, all_predictions_val, average='macro', zero_division=0)
        val_precisions.append(val_precision)
        val_recall = recall_score(all_labels_val, all_predictions_val, average='macro')
        val_recalls.append(val_recall)
        val_f1 = f1_score(all_labels_val, all_predictions_val, average='macro')
        val_f1s.append(val_f1)

        # Calculate the epoch training duration
        epoch_end_time = time.time()

        # Calculate the peak memory allocated in the current epoch
        memory_allocated = torch.cuda.max_memory_allocated()
        epoch_memory_allocated.append(memory_allocated)

        print(f'Epoch {epoch + 1}/{args.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
              f'Train Accuracy: {train_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%, Energy: {measurement.total_energy}J')

        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        # Save the model checkpoint
        checkpoint_path = os.path.join(target_directory, f'model_epoch_{epoch + 1}.pth')
        torch.save(model.state_dict(), checkpoint_path)


        # Store the training information
        training_info['epochs'][epoch] = {
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'train_precision': train_precision,
            'val_precision': val_precision,
            'train_recall': train_recall,
            'val_recall': val_recall,
            'train_f1': train_f1,
            'val_f1': val_f1,
            'energy': f'{measurement.total_energy}J',
            'train_and_validation_duration': f'{epoch_end_time - epoch_start_time} seconds',
            'epoch_training_duration': f'{epoch_training_time} seconds',
            'max_memory_allocated': f'{memory_allocated / (1024 ** 2):.2f} MB'
        }

        # Early stopping
        if training_info['stopped_at'] is None and epoch > 10:
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                training_info['stopped_at'] = epoch
                print("Early stopping in epoch: ", epoch)

    # Calculate the total training energy
    measurement = monitor.end_window("training")
    training_info['total_energy'] = f'{total_training_energy} J'
    print(f"Entire Training Energy: {total_training_energy} J")
    print(f"Entire Training and Validation Energy: {measurement.total_energy} J")
    # Calculate the total training duration
    training_info['training_duration'] = f'{total_training_time} seconds'
    # Calculate the maximum memory allocated during training
    training_info['max_memory_allocated'] = f'{max(epoch_memory_allocated) / (1024 ** 2):.2f} MB'

    # Remove the pruning reparametrizations to make the model explicitly sparse
    if args.pruning_type in ['manual', 'before']:
        remove_pruning(model, args.layer_wise)

    if args.pruning_type == 'after':
        # Load the best model before pruning
        best_model_path = os.path.join(target_directory, f'model_epoch_{args.epochs}.pth')
        model.load_state_dict(torch.load(best_model_path))

        # Debug: Before pruning
        # nonzero, total = count_nonzero_weights(model)
        # print(f'Before pruning: {nonzero}/{total} ({nonzero / total:.2%} non-zero)')

        # Apply pruning
        apply_pruning(model, args.sparsity, args.layer_wise)
        remove_pruning(model, args.layer_wise)

        # Debug: After pruning
        # nonzero, total = count_nonzero_weights(model)
        # print(f'After pruning: {nonzero}/{total} ({nonzero / total:.2%} non-zero)')

        # Save the pruned best model
        checkpoint_path = os.path.join(target_directory, f'post_pruned_model.pth')
        torch.save(model.state_dict(), checkpoint_path)

    # Test loop (after training is complete)
    test_model = DynamicSparseMLP(input_size[args.dataset], hidden_size, output_size[args.dataset]).to(device)
    # if args.pruning_type in ['none', 'manual', 'before', 'erdos']:
    #     try:
    #         print("Loading the best model")
    #         cleaned_state_dict = torch.load(os.path.join(target_directory, 'best_model.pth'))
    #     except FileNotFoundError:
    #         print("Load the 'stopped_at' epoch")
    #         cleaned_state_dict = torch.load(os.path.join(target_directory, f'model_epoch_{training_info["stopped_at"]}.pth'))
    if args.pruning_type == 'after':
        try:
            print("Loading the post-pruned model")
            cleaned_state_dict = torch.load(os.path.join(target_directory, 'post_pruned_model.pth'))
        except FileNotFoundError:
            print("Load the 'stopped_at' epoch")
            # cleaned_state_dict = torch.load(os.path.join(target_directory, f'model_epoch_{training_info["stopped_at"]}.pth'))
    else:
        print("Loading the last saved model")
        cleaned_state_dict = torch.load(os.path.join(target_directory, f'model_epoch_{args.epochs}.pth'))

    test_model.load_state_dict(clean_state_dict_last_version(cleaned_state_dict))
    torch.save(test_model.state_dict(), os.path.join(target_directory, f'model_epoch_{args.epochs}_compressed.pth'))
    test_model.eval()
    correct_test = 0
    total_test = 0
    all_labels_test = []
    all_predictions_test = []
    test_flps = []

    total_inference_time = 0
    total_inference_energy = 0
    total_inference_co2 = 0
    num_batches = 0

    # FLOPs measurement for testing
    total_test_flops = 0
    inference_tracker = EmissionsTracker(save_to_file=False, allow_multiple_runs=True)

    with torch.no_grad():  # No need to calculate gradients during testing
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.view(-1, input_size[args.dataset])

            # Measure FLOPs for the forward pass
            flops = count_flops(test_model, inputs)
            total_test_flops += flops

            monitor.begin_window("inference")
            inference_tracker.start()
            start_time = time.time()
            outputs = test_model(inputs)
            end_time = time.time()
            total_inference_co2 += inference_tracker.stop()
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

    # Store FLOPs for the test phase
    test_flps.append(total_test_flops)

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
    print(f'Total Inference CO2 Emissions: {total_inference_co2}')
    print(f'Total Number of Batches: {num_batches}')
    print(f'Average Inference Time per Batch: {avg_inference_time_per_batch:.6f} seconds')
    print(f'Average Inference Energy per Batch: {avg_inference_energy_per_batch:.6f} J')
    print(f'Total Training Time: {total_training_time:.6f} seconds')
    print(f'Total Training CO2 Emissions: {total_training_co2}')
    print(f'Average Test FLOPs: {total_test_flops / total_test}')
    print('=====================================')
    print('Checkpoints saved in:', target_directory)
    print('=====================================')

    # Save the test results to the training_info dictionary
    training_info['test_results'] = {
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'total_inference_time': total_inference_time,
        'total_inference_energy': total_inference_energy,
        'num_batches': num_batches,
        'avg_inference_time_per_batch': avg_inference_time_per_batch,
        'avg_inference_energy_per_batch': avg_inference_energy_per_batch,
        'total_training_time': total_training_time,
        'avg_test_flops': (total_test_flops / total_test),
    }

    # Save the training plot metrics
    generate_metrics_plot(train_losses, val_losses, target_directory, 'training_loss_plot.png', training_info['stopped_at'])
    generate_metrics_plot(train_accuracies, val_accuracies, target_directory, 'training_accuracy_plot.png', training_info['stopped_at'])
    generate_metrics_plot(train_precisions, val_precisions, target_directory, 'training_precision_plot.png', training_info['stopped_at'])
    generate_metrics_plot(train_recalls, val_recalls, target_directory, 'training_recall_plot.png', training_info['stopped_at'])
    generate_metrics_plot(train_f1s, val_f1s, target_directory, 'training_f1_plot.png', training_info['stopped_at'])
    generate_metrics_plot(train_flps, val_flps, target_directory, 'training_flop_plot.png', training_info['stopped_at'])

    with open(os.path.join(target_directory, 'model.json'), 'w') as file:
        json.dump(training_info, file, indent=4)


if __name__ == '__main__':
    # Call the main function
    args = parse_args()

    # Add the freeze_support() call for Windows
    freeze_support()

    # Call the main function
    main(args)


