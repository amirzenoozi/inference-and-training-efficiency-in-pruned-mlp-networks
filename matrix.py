import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


def split_layers(model):
    feature_layers = []
    classification_layers = []
    is_classification = False

    for layer in model.children():
        if isinstance(layer, nn.Linear):
            is_classification = True

        if is_classification:
            classification_layers.append(layer)
        else:
            feature_layers.append(layer)

    return feature_layers, classification_layers


def extract_adjacency_matrix(model, isWeightedGraph=True):
    # Map neuron indices to unique integers
    neuron_indices = {}
    current_index = 0
    last_layer = None
    layers_nodes = []
    size = 0

    last_layer, layers_nodes, size = extract_layer_information(model, layers_nodes, size, last_layer)

    size += last_layer.weight.shape[0]
    layers_nodes.append(last_layer.weight.shape[0])

    # Create an empty square adjacency matrix (dense representation)
    adjacency_matrix = np.zeros((size, size), dtype=float)

    # Update the adjacency matrix based on connections
    layer_number = 0
    for layer in list(model):
        try:
            num_outputs, num_inputs, *rest = layer.weight.shape
            for i in range(num_inputs):
                for j in range(num_outputs):
                    i_index, j_index = sum(layers_nodes[0: layer_number]) + i, sum(layers_nodes[0: layer_number + 1]) + j

                    # For convolutional layers, reduce kernel to single value
                    if len(rest) > 0:
                        weight = layer.weight[j, i].view(-1).mean().item()

                    # For fully connected layers
                    else:
                        weight = layer.weight[j, i].item()

                    if isWeightedGraph:
                        adjacency_matrix[i_index, j_index] = weight
                    else:
                        adjacency_matrix[i_index, j_index] = 1 if weight != 0 else 0

            layer_number += 1
        except AttributeError:
            pass

    num_nodes = size
    num_edges = np.count_nonzero(adjacency_matrix)

    return adjacency_matrix, num_nodes, num_edges, layers_nodes


def extract_layer_information(model_layers, layers_nodes, size, last_layer):
    local_last_layer = last_layer
    local_layers_nodes = layers_nodes
    local_size = size

    for layer in model_layers:
        try:
            if not isinstance(layer, nn.Sequential):
                num_outputs, num_inputs, *rest = layer.weight.shape
                local_layers_nodes.append(num_inputs)
                local_size += num_inputs
                local_last_layer = layer
            else:
                local_last_layer, local_layers_nodes, local_size = extract_layer_information(layer.children(), local_layers_nodes, local_size, local_last_layer)
        except AttributeError:
            pass

    return local_last_layer, local_layers_nodes, local_size


def calculate_degree_distribution(adjacency_matrix, weighted=False):
    degrees = []
    if weighted:
        # Weighted graph: Sum the weights for each node
        degrees = np.sum(adjacency_matrix, axis=1)
    else:
        # Unweighted graph: Count the non-zero entries for each node
        degrees = np.count_nonzero(adjacency_matrix, axis=1)

    if weighted:
        # For weighted graphs, use histogram to calculate distribution
        hist, bin_edges = np.histogram(degrees, bins='auto', density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return bin_centers, hist
    else:
        # For unweighted graphs, use bincount to calculate distribution
        degree_distribution = np.bincount(degrees)
        probabilities = degree_distribution / len(adjacency_matrix)
        return np.arange(len(probabilities)), probabilities


def plot_degree_distribution(degrees, probabilities, title, file_name, visualize):
    plt.figure(figsize=(10, 6))
    bar_width = 0.02  # Set bar width less than 1 to add space between bars
    plt.plot(degrees, probabilities, marker='o', linestyle='-', color='#D4D4D4', markersize=5, label='Degree Probability')
    plt.bar(degrees, probabilities, width=bar_width, color='#1F77B4', align='center', edgecolor='#113652', label='Degree Probability')
    plt.xlabel('Degree', fontsize=14)
    plt.ylabel('Probability', fontsize=14)
    plt.title(f'Degree Distribution of: {title}', fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for i, txt in enumerate(probabilities):
        plt.annotate(f'{txt:.2f}', (degrees[i], probabilities[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(file_name)

    if visualize == 1:
        plt.show()


def plot_centrality_histogram(centrality_metric, title, file_name, visualize):
    plt.figure(figsize=(10, 6))
    bar_width = 0.02  # Set bar width less than 1 to add space between bars
    values = list(centrality_metric.values())
    plt.hist(values, bins=10, edgecolor='#113652', color='#1F77B4', alpha=0.7)

    # Add labels and title
    plt.xlabel('Centrality Value', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title(title, fontsize=16)

    # Add grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Annotate histogram bars with frequency values
    counts, bins, patches = plt.hist(values, bins=10, edgecolor='#113652', color='#1F77B4', alpha=0.7)
    for count, patch in zip(counts, patches):
        height = patch.get_height()
        plt.text(patch.get_x() + patch.get_width() / 2, height + 0.5, f'{int(height)}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(file_name)

    if visualize == 1:
        plt.show()


def store_graph_in_gephi_format(adjacency_matrix, file_name):
    # Create the graph from the adjacency matrix
    G = create_graph_from_adjacency_matrix(adjacency_matrix)
    nx.write_gexf(G, file_name)


def create_graph_from_adjacency_matrix(adjacency_matrix):
    G = nx.DiGraph()
    size = len(adjacency_matrix)
    for i in range(size):
        for j in range(size):
            if adjacency_matrix[i, j] != 0:
                G.add_edge(i, j, weight=adjacency_matrix[i, j])
    return G


def rank_nodes(centrality_metric):
    return sorted(centrality_metric.items(), key=lambda item: item[1], reverse=True)


def calculate_centrality_metrics(adjacency_matrix, isWeightedGraph=True):
    # Create the graph from the adjacency matrix
    G = create_graph_from_adjacency_matrix(adjacency_matrix)

    # Degree centrality
    in_degree_centrality = nx.in_degree_centrality(G)
    out_degree_centrality = nx.out_degree_centrality(G)

    if isWeightedGraph:
        # Betweenness centrality
        betweenness_centrality = nx.betweenness_centrality(G, weight='weight', normalized=True)
        # Eigenvector centrality
        eigenvector_centrality = nx.eigenvector_centrality(G, weight='weight', max_iter=1000, tol=1e-06)
    else:
        # Betweenness centrality
        betweenness_centrality = nx.betweenness_centrality(G, normalized=True)
        # Eigenvector centrality
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06)

    return in_degree_centrality, out_degree_centrality, betweenness_centrality, eigenvector_centrality


def calculate_largest_connected_component_size(adjacency_matrix, is_DiGraph=True):
    # Create the graph from the adjacency matrix
    G = create_graph_from_adjacency_matrix(adjacency_matrix)

    # Find the largest connected component
    if is_DiGraph:
        largest_cc = max(nx.strongly_connected_components(G), key=len)
    else:
        largest_cc = max(nx.connected_components(G), key=len)

    # Return the size of the largest connected component
    return len(largest_cc)
