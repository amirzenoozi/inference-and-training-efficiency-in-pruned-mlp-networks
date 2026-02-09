import argparse
import os.path
import torch

import matrix as mx
import scripts as sc

from tqdm import tqdm
from pathlib import Path
from mnist_mlp.model import DynamicSparseMLP
from mnist_mlp.set_mlp_model import SetMLP


def parse_args():
    desc = "Visualize the Neural Network Architecture from a PyTorch Model Checkpoint"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-B', '--pth_base', type=str, default='./mnist/model_checkpoints', help='What is The Root Path?')
    parser.add_argument('-F', '--pth_start', type=int, default=1, help='What is The First Pth File?')
    parser.add_argument('-E', '--pth_end', type=int, default=30, help='What is The Last Pth File?')
    parser.add_argument('-W', '--weighted', type=int, default=1, help='Does it a Weighted Graph? (1/0)', choices=[0, 1])
    parser.add_argument('-M', '--model', type=str, default='MNIST', help='MNIST or DogVsCat?', choices=['MNIST'])

    return parser.parse_args()


def main(args):
    if args.model == 'MNIST':
        input_size = 28 * 28
        hidden_size = 16
        output_size = 10
        model = SetMLP(input_size, output_size)
    else:
        print('Model not found!')
        return

    # Extract the parent directory and the file name
    checkpoint_path = Path(args.pth_base)
    checkpoint_parent_dir = checkpoint_path.absolute()

    for pth_file in tqdm(range(args.pth_start, args.pth_end + 1)):
        # Load the saved model state dictionary from a .pth file
        checkpoint = torch.load(Path(f'{checkpoint_parent_dir}\model_epoch_{pth_file}.pth'))

        # Apply the masks to the original weights
        if args.model in ['MNIST']:
            for key in list(checkpoint.keys()):
                if 'weight_orig' in key:
                    # Compute the sparse weights
                    sparse_weights = checkpoint[key] * checkpoint[key.replace('weight_orig', 'weight_mask')]
                    # Assign the sparse weights to the corresponding key
                    checkpoint[key.replace('_orig', '')] = sparse_weights
                    # Remove the original and mask keys from the state_dict
                    del checkpoint[key]
                    del checkpoint[key.replace('weight_orig', 'weight_mask')]

        # Load the checkpoint into the model
        model.load_state_dict(checkpoint)

        # Extract the adjacency matrix from the model
        feature_layers, classification_layers = mx.split_layers(model)
        matrix, nodes, edges, layers = mx.extract_adjacency_matrix(classification_layers, bool(args.weighted))
        in_degree_centrality, out_degree_centrality, betweenness_centrality, eigenvector_centrality = mx.calculate_centrality_metrics(matrix, bool(args.weighted))

        # Sorted Degree Centrality Metrics
        sorted_in_degree_centrality = sorted(in_degree_centrality.items())
        in_degree_centrality_values = [item[1] for item in sorted_in_degree_centrality]

        # Sorted Degree Centrality Metrics
        sorted_out_degree_centrality = sorted(out_degree_centrality.items())
        out_degree_centrality_values = [item[1] for item in sorted_out_degree_centrality]

        # Sorted Betweenness Centrality Metrics
        sorted_betweenness_centrality = sorted(betweenness_centrality.items())
        betweenness_centrality_values = [item[1] for item in sorted_betweenness_centrality]

        # Sorted Eigenvector Centrality Metrics
        sorted_eigenvector_centrality = sorted(eigenvector_centrality.items())
        eigenvector_centrality_values = [item[1] for item in sorted_eigenvector_centrality]

        # Write the metrics to a CSV file
        sc.write_all_epochs_to_csv(os.path.join(checkpoint_parent_dir, f'in_degree_centrality_{args.pth_start}_{args.pth_end}.csv'), pth_file, in_degree_centrality_values)
        sc.write_all_epochs_to_csv(os.path.join(checkpoint_parent_dir, f'out_degree_centrality_{args.pth_start}_{args.pth_end}.csv'), pth_file, out_degree_centrality_values)
        sc.write_all_epochs_to_csv(os.path.join(checkpoint_parent_dir, f'betweenness_centrality_{args.pth_start}_{args.pth_end}.csv'), pth_file, betweenness_centrality_values)
        sc.write_all_epochs_to_csv(os.path.join(checkpoint_parent_dir, f'eigenvector_centrality_{args.pth_start}_{args.pth_end}.csv'), pth_file, eigenvector_centrality_values)



if __name__ == '__main__':
    # Call the main function
    args = parse_args()
    main(args)

