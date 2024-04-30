import matplotlib.pyplot as plt
import torch

def plot_graph_positions(node_positions, title="Graph", save_path=None):
    """
    Plot the graph node positions.

    Parameters:
    - node_positions: A tensor of shape [n_nodes, 2], where each row represents the x and y coordinates.
    - title: The plot title.
    - save_path: If provided, saves the plot to the specified path. Otherwise, shows the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(node_positions[:, 0].cpu().numpy(), node_positions[:, 1].cpu().numpy(), s=10)
    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    # plt.grid(True)
    if save_path:
        plt.savefig('/home/pietro_devecchi/GNN_autoencoder/plot_nodes/'+save_path)
        plt.close()
    else:
        plt.show()

def plot_mapping(node_positions_fine, node_positions_coarse, fine2coarse_map, title="Mapping", save_path=None):
    """
    Plot the mapping between fine and coarse nodes.

    Parameters:
    - node_positions_fine: A tensor of shape [n_fine_nodes, 2], where each row represents the x and y coordinates.
    - node_positions_coarse: A tensor of shape [n_coarse_nodes, 2], where each row represents the x and y coordinates.
    - fine2coarse_map: A dictionary mapping each fine node index to its corresponding coarse node index.
    - title: The plot title.
    - save_path: If provided, saves the plot to the specified path. Otherwise, shows the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(node_positions_fine[:, 0].cpu().numpy(), node_positions_fine[:, 1].cpu().numpy(), s=10, label="Fine nodes")
    plt.scatter(node_positions_coarse[:, 0].cpu().numpy(), node_positions_coarse[:, 1].cpu().numpy(), s=50, label="Coarse nodes", marker="x")
    for fine_index in range(node_positions_fine.shape[0]):
        plt.plot([node_positions_fine[fine_index, 0].cpu().numpy(), node_positions_coarse[fine2coarse_map[fine_index], 0].cpu().numpy()],
                 [node_positions_fine[fine_index, 1].cpu().numpy(), node_positions_coarse[fine2coarse_map[fine_index], 1].cpu().numpy()], c="black", alpha=0.5)
    
    # for fine_idx, coarse_idx in fine2coarse_map.items():
    #     plt.plot([node_positions_fine[fine_idx, 0], node_positions_coarse[coarse_idx, 0]],
    #              [node_positions_fine[fine_idx, 1], node_positions_coarse[coarse_idx, 1]], c="black", alpha=0.5)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(title)
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.legend()
    # plt.grid(True)
    if save_path:
        plt.savefig('/home/pietro_devecchi/GNN_autoencoder/plot_nodes/'+save_path)
        plt.close()
    else:
        plt.show()

def plot_graph(node_positions, edge_index, title = "Graph", save_path=None):
    """
    Plot the graph structure.

    Parameters:
    - node_positions: A tensor of shape [n_nodes, 2], where each row represents the x and y coordinates.
    - edge_index: A tensor of shape [2, n_edges], where each column represents an edge between two nodes.
    - title: The plot title.
    - save_path: If provided, saves the plot to the specified path. Otherwise, shows the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(node_positions[:, 0].cpu().numpy(), node_positions[:, 1].cpu().numpy(), s=10)
    # for edge in edge_index.t().cpu().numpy():
    #     plt.plot([node_positions[edge[0], 0].cpu().numpy(), node_positions[edge[1], 0].cpu().numpy()],
    #                 [node_positions[edge[0], 1].cpu().numpy(), node_positions[edge[1], 1].cpu().numpy()], c="black", alpha=0.5)

    for edge in edge_index.t().cpu().numpy():
        x_start, y_start = node_positions[edge[0], 0].cpu().numpy(), node_positions[edge[0], 1].cpu().numpy()
        x_end, y_end = node_positions[edge[1], 0].cpu().numpy(), node_positions[edge[1], 1].cpu().numpy()
        plt.arrow(x_start, y_start, x_end - x_start, y_end - y_start, color="black", alpha=0.5, head_width=0.02, head_length=0.02)
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(title)
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    # plt.grid(True)
    if save_path:
        plt.savefig('/home/pietro_devecchi/GNN_autoencoder/plot_nodes/'+save_path)
        plt.close()
    else:
        plt.show()