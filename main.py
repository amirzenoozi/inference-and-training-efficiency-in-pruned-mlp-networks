import argparse
import torch
import visualizer

import matrix as mx
import scripts as sc

from pathlib import Path
from mnist_mlp.model import DynamicSparseMLP
from large_mlp.model import SetMLP
from timeit import default_timer as timer


def parse_args():
    desc = "Visualize the Neural Network Architecture from a PyTorch Model Checkpoint"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-P', '--pth', type=str, default='./models/mlp_model.pth', help='What is Your Model Path?')
    parser.add_argument('-M', '--model', type=str, default='MNIST', help='MNIST or DogVsCat?', choices=['MNIST'])
    parser.add_argument('-J', '--json', type=int, default=0, help='Save the Json Parameters? (1/0)', choices=[0, 1])
    parser.add_argument('-C', '--csv', type=int, default=0, help='Save the Matrix as a CSV File? (1/0)', choices=[0, 1])
    parser.add_argument('-V', '--visualize', type=int, default=0, help='Visualize the Network? (1/0)', choices=[0, 1])
    parser.add_argument('-W', '--weighted', type=int, default=1, help='Does it a Weighted Graph? (1/0)', choices=[0, 1])

    return parser.parse_args()


def main(args):
    if args.model == 'MNIST':
        input_size = 28 * 28
        hidden_size = 16
        output_size = 10
        # model = SetMLP(input_size, output_size)
        model = DynamicSparseMLP(input_size, hidden_size, output_size)
    else:
        print('Model not found!')
        return

    # Extract the parent directory and the file name
    checkpoint_path = Path(args.pth)
    checkpoint_parent_dir = checkpoint_path.parent.absolute()
    file_name = checkpoint_path.name.split('.')[0]
    output_filename = f'{checkpoint_parent_dir}/{file_name}'

    # Load the saved model state dictionary from a .pth file
    checkpoint = torch.load(checkpoint_path)

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
    start_time = timer()
    feature_layers, classification_layers = mx.split_layers(model)
    matrix, nodes, edges, layers = mx.extract_adjacency_matrix(classification_layers, bool(args.weighted))

    if args.visualize == 1:
        print(f"Number of Nodes: {nodes}")
        print(f"Number of Edges: {edges}")

    end_time = timer()

    # Model Parameters to be stored in JSON
    if args.json == 1:
        print('Storing the Model Parameters in a JSON File...')
        degrees, probabilities = mx.calculate_degree_distribution(matrix, bool(args.weighted))
        in_degree_centrality, out_degree_centrality, betweenness_centrality, eigenvector_centrality = mx.calculate_centrality_metrics(matrix, bool(args.weighted))
        largest_cc_size = mx.calculate_largest_connected_component_size(matrix, True)
        model_parameters = {
            'nodes': nodes,
            'edges': edges,
            'time': end_time - start_time,
            'csv': f'{output_filename}.csv',
            'model_path': args.pth,
            'model_name': args.model,
            'largest_connected_component_size': largest_cc_size,
        }

        sc.store_matrix_parameters(model_parameters, f'{output_filename}.json')
        mx.plot_degree_distribution(degrees, probabilities, file_name, f'{output_filename}_degree_distribution.png', args.visualize)

        # Centrality Ranking of the Nodes with 3 different methods
        in_degree_ranking = mx.rank_nodes(in_degree_centrality)
        out_degree_ranking = mx.rank_nodes(out_degree_centrality)
        betweenness_ranking = mx.rank_nodes(betweenness_centrality)
        eigenvector_ranking = mx.rank_nodes(eigenvector_centrality)

        # Draw ranking Plot
        mx.plot_centrality_histogram(in_degree_centrality, 'InDegree Centrality Distribution', f'{output_filename}_in_degree_centrality.png', args.visualize)
        mx.plot_centrality_histogram(out_degree_centrality, 'OutDegree Centrality Distribution', f'{output_filename}_out_degree_centrality.png', args.visualize)
        mx.plot_centrality_histogram(betweenness_centrality, 'Betweenness Centrality Distribution', f'{output_filename}_betweenness_centrality.png', args.visualize)
        mx.plot_centrality_histogram(eigenvector_centrality, 'Eigenvector Centrality Distribution', f'{output_filename}_eigenvector_centrality.png', args.visualize)
        mx.store_graph_in_gephi_format(matrix, f'{output_filename}_gephi.gexf')

    if args.csv == 1:
        print('Storing the Adjacency Matrix as a CSV File...')
        sc.store_adjacency_matrix(matrix, f'{output_filename}.csv')

    print('Visualizing the Network...')
    visualizer.draw_network_from_adjacency_matrix(matrix, layers, f'{output_filename}.png', args.visualize)


if __name__ == '__main__':
    # Call the main function
    args = parse_args()
    main(args)
