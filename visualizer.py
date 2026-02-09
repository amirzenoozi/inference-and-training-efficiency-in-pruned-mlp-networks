import matplotlib.pyplot as plt
import networkx as nx


def draw_network_from_adjacency_matrix(adjacency_matrix, layers=None, output_filename=None, visualize=False):
    # Create a graph from the adjacency matrix
    G = nx.DiGraph(adjacency_matrix)

    # Set the node attributes
    largest_layer_index = layers.index(max(layers))
    for index, layer in enumerate(layers):
        for node in range(sum(layers[0:index]), sum(layers[0:index + 1])):
            G.nodes[node]["name"] = f"l_{index + 1}_n_{node + 1}"
            scale_factor = 2.5 if index == largest_layer_index else 1
            G.nodes[node]["pos"] = (index, node * scale_factor)

    # Draw the graph
    # pos = nx.spring_layout(G)  # You can choose a different layout if needed
    nx.draw(
        G=G,
        pos=nx.get_node_attributes(G, 'pos'),
        with_labels=True,
        labels=nx.get_node_attributes(G, "name"),
        font_weight='bold',
        node_size=400,
        node_color='skyblue',
        font_size=8,
        edge_color='gray'
    )

    # Add edge labels
    edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, nx.get_node_attributes(G, 'pos'), edge_labels=edge_labels, font_color='red')

    # Get the number of nodes and edges
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    # print(f"G Nodes: {num_nodes}")
    # print(f"G Edges: {num_edges}")

    # Save the plot
    plt.savefig(output_filename)

    if visualize:
        plt.show()
