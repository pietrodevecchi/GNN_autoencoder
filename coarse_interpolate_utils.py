import dolfin
import torch
import torch.nn as nn
from torch_scatter import scatter_add
from torch.nn import Linear, ModuleList
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from dataset import create_dataset
from scipy.spatial import Delaunay
import matplotlib.path as mpath
from scipy.spatial import ConvexHull
from core_model import MLP


# class CoarsenGraphWithMLP(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
#         super().__init__()
#         self.mlp = MLP(num_layers, in_channels + 1, hidden_channels, out_channels)  # +1 for distance feature

#     def forward(self, nodes, edges, length_scale, node_features, edge_attr):
#         voxel_centroids, closest_centroid_indices, distances = create_voxel_grid_with_distances(nodes, length_scale)
#         node_assignments = assign_nodes_to_voxels(nodes, voxel_centroids)

#         # Assuming node_features shape is [num_nodes, num_timesteps, num_features]
#         # And distances shape is [num_nodes]

#         # # Convert distances to a PyTorch tensor and add dimensions to match node_features
#         # distances_tensor = torch.tensor(distances, device=node_features.device).unsqueeze(1).unsqueeze(-1)  # Now [num_nodes, 1, 1]

#         # # Expand distances to match the timesteps dimension of node_features
#         # distances_expanded = distances_tensor.expand(-1, node_features.size(1), -1)  # Now [num_nodes, num_timesteps, 1]

#         # # Concatenate expanded distances with node_features
#         # extended_node_features = torch.cat([node_features, distances_expanded], dim=-1)

#         # Prepare extended node features (original features + distances)
#         extended_node_features = torch.cat([node_features, torch.tensor(distances).unsqueeze(-1).to(node_features.device)], dim=-1)

#         # Process extended features with MLP
#         processed_features = self.mlp(extended_node_features)

#         # Initialize feature aggregator for coarse nodes
#         coarse_features_sum = torch.zeros((len(voxel_centroids), processed_features.size(-1))).to(processed_features.device)
#         coarse_features_count = torch.zeros(len(voxel_centroids)).to(processed_features.device)

#         # Aggregate features
#         for idx, assignment in enumerate(closest_centroid_indices):
#             coarse_features_sum[assignment] += processed_features[idx]
#             coarse_features_count[assignment] += 1

#         # Avoid division by zero
#         coarse_features_count[coarse_features_count == 0] = 1
#         coarse_features_avg = coarse_features_sum / coarse_features_count.unsqueeze(-1)

#         # Here, coarse_features_avg contains the averaged features for each coarse node
#         # Continue with creating coarse edges and attributes as necessary
#         # This example stops at generating coarse node features

#         return coarse_features_avg


# Function definitions remain unchanged
# def create_voxel_grid_with_distances(nodes, length_scale):
#     min_x, min_y = np.min(nodes, axis=0)
#     max_x, max_y = np.max(nodes, axis=0)
#     grid_x = np.arange(min_x + length_scale / 2, max_x, length_scale)
#     grid_y = np.arange(min_y + length_scale / 2, max_y, length_scale)
#     voxel_centroids = np.vstack([np.repeat(grid_x, len(grid_y)), np.tile(grid_y, len(grid_x))]).T
    
#     # Use KDTree to find the closest centroid and distance to each node
#     centroid_tree = cKDTree(voxel_centroids)
#     distances, closest_centroid_indices = centroid_tree.query(nodes)
    
#     # Identify unique centroids that have at least one node assigned to them
#     unique_assigned_centroids = np.unique(closest_centroid_indices)
    
#     # Filter the voxel_centroids to keep only those with nodes assigned
#     final_voxel_centroids = voxel_centroids[unique_assigned_centroids]

#     fin_centroid_tree = cKDTree(final_voxel_centroids)

#     fin_distances, fin_closest_centroid_indices = fin_centroid_tree.query(nodes)
    
#     return final_voxel_centroids, fin_closest_centroid_indices, fin_distances


# def create_voxel_grid_with_distances(nodes, length_scale):
#     min_x, min_y = np.min(nodes, axis=0)
#     max_x, max_y = np.max(nodes, axis=0)
#     grid_x = np.arange(min_x + length_scale / 2, max_x, length_scale)
#     grid_y = np.arange(min_y + length_scale / 2, max_y, length_scale)
#     voxel_centroids = np.vstack([np.repeat(grid_x, len(grid_y)), np.tile(grid_y, len(grid_x))]).T
    
#     # Use KDTree to find the closest centroid and distance to each node
#     centroid_tree = cKDTree(voxel_centroids)
#     distances, closest_centroid_indices = centroid_tree.query(nodes)
    
#     # Identify unique centroids that have at least one node assigned to them
#     unique_assigned_centroids = np.unique(closest_centroid_indices)
    
#     # Filter the voxel_centroids to keep only those with nodes assigned
#     final_voxel_centroids = voxel_centroids[unique_assigned_centroids]

#     fin_centroid_tree = cKDTree(final_voxel_centroids)

#     fin_distances, fin_closest_centroid_indices = fin_centroid_tree.query(nodes)

#     # Convert fin_distances to a PyTorch tensor
#     fin_distances = torch.from_numpy(fin_distances).float()  # Ensure the type is float for compatibility
#     fin_distances = fin_distances.unsqueeze(0).unsqueeze(-1)

#     return final_voxel_centroids, fin_closest_centroid_indices, fin_distances


# def create_voxel_grid_with_distances(nodes, length_scale):
#     # nodes = torch.from_numpy(nodes) if isinstance(nodes, np.ndarray) else nodes
#     if not isinstance(nodes, torch.Tensor):
#         nodes = torch.tensor(nodes, dtype=torch.float)
#     min_x, min_y = torch.min(nodes, dim=0).values
#     max_x, max_y = torch.max(nodes, dim=0).values
#     grid_x = torch.arange(min_x + length_scale / 2, max_x, length_scale)
#     grid_y = torch.arange(min_y + length_scale / 2, max_y, length_scale)
#     voxel_centroids = torch.stack([grid_x.repeat(len(grid_y)), grid_y.repeat_interleave(len(grid_x))], dim=1)
    
#     # Use KDTree to find the closest centroid and distance to each node, requires conversion to numpy
#     centroid_tree = cKDTree(voxel_centroids.numpy())
#     distances, closest_centroid_indices = centroid_tree.query(nodes.numpy())
    
#     # Identify unique centroids that have at least one node assigned to them
#     unique_assigned_centroids = np.unique(closest_centroid_indices)
    
#     # Filter the voxel_centroids to keep only those with nodes assigned
#     # final_voxel_centroids = voxel_centroids[torch.tensor(unique_assigned_centroids, dtype=torch.long)]

#     final_voxel_centroids = voxel_centroids[torch.tensor(unique_assigned_centroids, dtype=torch.float)]


#     fin_centroid_tree = cKDTree(final_voxel_centroids.numpy())

#     fin_distances, fin_closest_centroid_indices = fin_centroid_tree.query(nodes.numpy())

#     # Convert to PyTorch tensors
#     # final_voxel_centroids = torch.tensor(final_voxel_centroids, dtype=torch.float)
#     # fin_closest_centroid_indices = torch.tensor(fin_closest_centroid_indices, dtype=torch.long)
#     fin_closest_centroid_indices = torch.tensor(fin_closest_centroid_indices, dtype=torch.float)

#     fin_distances = torch.tensor(fin_distances, dtype=torch.float).unsqueeze(0).unsqueeze(-1)

#     return final_voxel_centroids, fin_closest_centroid_indices, fin_distances


def coarsen_graph(nodes, edges, length_scale, edge_attr):
    batch_size = edge_attr.size(0)
    num_attrs = edge_attr.size(2)  
    # Use 'create_voxel_grid_with_distances' to get centroids, indices, and distances as PyTorch tensors
    voxel_centroids, closest_centroid_indices, distances = create_voxel_grid_with_distances(nodes, length_scale)

    # Coarse nodes are directly the voxel centroids
    coarse_nodes = voxel_centroids

    # Prepare lists for storing start and end voxels for coarse edges and their attributes
    start_voxels_list = []
    end_voxels_list = []
    coarse_edge_attrs_list = []

    # Initialize a dictionary for mapping voxel pairs to their corresponding fine edges
    edge_index_map = {}

    # Iterate over each edge to populate the map
    for i in range(edges.size(1)):
        start_node, end_node = edges[:, i]
        start_voxel, end_voxel = closest_centroid_indices[start_node].item(), closest_centroid_indices[end_node].item()

        if start_voxel != end_voxel:
            key = tuple(sorted([start_voxel, end_voxel]))
            edge_index_map.setdefault(key, []).append(i)

    # Process each edge in the map to calculate averaged attributes
    # for (start_voxel, end_voxel), indices in edge_index_map.items():
    #     indices = torch.tensor(indices, dtype=torch.long)  # Use long for indexing
    #     selected_attrs = edge_attr[:, indices].mean(dim=1)  # Average over the dimension of edges
    #     # Append start and end voxels separately
    #     start_voxels_list.append(start_voxel)
    #     end_voxels_list.append(end_voxel)
    #     coarse_edge_attrs_list.append(selected_attrs.unsqueeze(0))

    for batch_idx in range(batch_size):
        batch_edge_attrs_list = []
        for (start_voxel, end_voxel), indices in edge_index_map.items():
            indices_tensor = torch.tensor(indices, dtype=torch.long)
            # Select and average attributes for current batch and indices
            selected_attrs = edge_attr[batch_idx, indices_tensor].mean(dim=0)  # Shape: [8]
            batch_edge_attrs_list.append(selected_attrs.unsqueeze(0))  # Add batch dimension back
            if batch_idx==0:
                start_voxels_list.append(start_voxel)
                end_voxels_list.append(end_voxel)

        # Concatenate all edge attributes for the current batch
        if batch_edge_attrs_list:
            batch_coarse_edge_attrs = torch.cat(batch_edge_attrs_list, dim=0)  # Shape: [M, 8] for current batch
            coarse_edge_attrs_list.append(batch_coarse_edge_attrs.unsqueeze(0)) 

    # Convert lists to tensors

    if start_voxels_list:
        # Convert lists to tensors and then stack to get shape [2, N]
        start_voxels = torch.tensor(start_voxels_list, dtype=torch.long)
        end_voxels = torch.tensor(end_voxels_list, dtype=torch.long)
        coarse_edges = torch.stack([start_voxels, end_voxels], dim=0)
    else:
        coarse_edges = torch.empty((2, 0), dtype=torch.long)  # Ensure dtype matches index dtype

    # print('coarse_edges: ', type(coarse_edges), coarse_edges.size(), '\n')

    # print('coarse_edge_attrs_list: ', len(coarse_edge_attrs_list), len(coarse_edge_attrs_list[0]), \
    #     len(coarse_edge_attrs_list[0][0]), '\n')

    if coarse_edge_attrs_list:
        coarse_edge_attrs = torch.cat(coarse_edge_attrs_list, dim=0) #.repeat(edge_attr.size(0), 1, 1) # Shape: [1, N, Attrs]
    else:
        coarse_edge_attrs = torch.empty((1, 0, edge_attr.size(-1)), dtype=edge_attr.dtype)

    return coarse_nodes, coarse_edges, closest_centroid_indices, coarse_edge_attrs, distances

def create_maps_distances(nodes, scales):
    fine2coarse_list = {}
    distances = {}
    
    for length_scale in scales:
        nodes, fine2coarse, dist = create_voxel_grid_with_distances(nodes, length_scale)
        fine2coarse_list[length_scale]=fine2coarse
        distances[length_scale]=dist
    return fine2coarse_list, distances

def create_voxel_grid_with_distances(nodes, length_scale):
    if not isinstance(nodes, torch.Tensor):
        nodes = torch.tensor(nodes, dtype=torch.float)
    min_x, min_y = torch.min(nodes, dim=0).values
    max_x, max_y = torch.max(nodes, dim=0).values
    grid_x = torch.arange(min_x + length_scale / 2, max_x, length_scale)
    grid_y = torch.arange(min_y + length_scale / 2, max_y, length_scale)
    voxel_centroids = torch.stack([grid_x.repeat(len(grid_y)), grid_y.repeat_interleave(len(grid_x))], dim=1)
    
    centroid_tree = cKDTree(voxel_centroids.numpy())
    distances, closest_centroid_indices = centroid_tree.query(nodes.numpy())
    
    unique_assigned_centroids = torch.tensor(np.unique(closest_centroid_indices), dtype=torch.long)
    final_voxel_centroids = voxel_centroids[unique_assigned_centroids]
    
    fin_centroid_tree = cKDTree(final_voxel_centroids.numpy())
    fin_distances, fin_closest_centroid_indices = fin_centroid_tree.query(nodes.numpy())
    
    fin_closest_centroid_indices = torch.tensor(fin_closest_centroid_indices, dtype=torch.long)
    fin_distances = torch.tensor(fin_distances, dtype=torch.float).unsqueeze(0).unsqueeze(-1)

    return final_voxel_centroids, fin_closest_centroid_indices, fin_distances

# def create_voxel_grid_with_distances(nodes, length_scale):
#     if not isinstance(nodes, torch.Tensor):
#         nodes = torch.tensor(nodes, dtype=torch.float)
#     min_x, min_y = torch.min(nodes, dim=0).values
#     max_x, max_y = torch.max(nodes, dim=0).values
#     grid_x = torch.arange(min_x + length_scale / 2, max_x, length_scale)
#     grid_y = torch.arange(min_y + length_scale / 2, max_y, length_scale)
#     voxel_centroids = torch.stack([grid_x.repeat(len(grid_y)), grid_y.repeat_interleave(len(grid_x))], dim=1)
    
#     centroid_tree = cKDTree(voxel_centroids.numpy())
#     distances, closest_centroid_indices = centroid_tree.query(nodes.numpy())
    
#     unique_assigned_centroids = torch.tensor(np.unique(closest_centroid_indices), dtype=torch.long)
#     final_voxel_centroids = voxel_centroids[unique_assigned_centroids]
    
#     fin_centroid_tree = cKDTree(final_voxel_centroids.numpy())
#     fin_distances, fin_closest_centroid_indices = fin_centroid_tree.query(nodes.numpy())
    
#     fin_closest_centroid_indices = torch.tensor(fin_closest_centroid_indices, dtype=torch.long)
#     fin_distances = torch.tensor(fin_distances, dtype=torch.float).unsqueeze(0).unsqueeze(-1)

#     return final_voxel_centroids, fin_closest_centroid_indices, fin_distances


# def assign_nodes_to_voxels(nodes, voxel_centroids):
#     tree = cKDTree(voxel_centroids)
#     _, node_assignments = tree.query(nodes)
#     return node_assignments

# def coarsen_graph(nodes, edges, length_scale, edge_attr):
#     # Assuming `create_voxel_grid_with_distances` now returns PyTorch tensors.
#     voxel_centroids, closest_centroid_indices, distances = create_voxel_grid_with_distances(nodes, length_scale)

#     # Adjust `assign_nodes_to_voxels` to accept and return PyTorch tensors (adjustment not shown here).
#     # node_assignments = assign_nodes_to_voxels(nodes, voxel_centroids)

#     # Instead of a dictionary, directly use a tensor for coarse nodes.
#     coarse_nodes = voxel_centroids

#     # Prepare tensors for storing coarse edges and their attributes.
#     start_voxels_list = []
#     end_voxels_list = []
#     coarse_edge_attrs_list = []

#     # Maps voxel pairs to their corresponding fine edges. Initialize as a dictionary for easy mapping.
#     edge_index_map = {}

#     for i in range(edges.size(1)):
#         start_node, end_node = edges[:, i]
#         start_voxel, end_voxel = closest_centroid_indices[start_node], closest_centroid_indices[end_node]

#         if start_voxel != end_voxel:
#             key = tuple(sorted([start_voxel.item(), end_voxel.item()]))  # Ensure items are Python ints for the dictionary.
#             edge_index_map.setdefault(key, []).append(i)

#     # Process each edge in the map to calculate averaged attributes.
#     for (start_voxel, end_voxel), indices in edge_index_map.items():
#         # indices = torch.tensor(indices, dtype=torch.long)
#         indices = torch.tensor(indices, dtype=torch.float)
#         selected_attrs = edge_attr[:, indices].mean(dim=1)  # Average over the dimension of edges.
#         # Append start and end voxels separately
#         start_voxels_list.append(start_voxel)
#         end_voxels_list.append(end_voxel)
#         coarse_edge_attrs_list.append(selected_attrs)

#     # Convert lists to tensors.
#     if coarse_edge_attrs_list:
#         coarse_edge_attrs = torch.cat(coarse_edge_attrs_list, dim=0).unsqueeze(0)  # [1, N, Attrs]
#     else:
#         coarse_edge_attrs = torch.empty((1, 0, edge_attr.size(-1)), dtype=edge_attr.dtype)

#     if start_voxels_list:
#         # Stack start and end voxels to get [2, N] shape for edges
#         # start_voxels = torch.tensor(start_voxels_list, dtype=torch.long)
#         # end_voxels = torch.tensor(end_voxels_list, dtype=torch.long)
#         start_voxels = torch.tensor(start_voxels_list, dtype=torch.float)
#         end_voxels = torch.tensor(end_voxels_list, dtype=torch.float)
#         coarse_edges = torch.stack([start_voxels, end_voxels], dim=0)
#     else:
#         # coarse_edges = torch.empty((2, 0), dtype=torch.long)
#         coarse_edges = torch.empty((2, 0), dtype=torch.float)

#     return coarse_nodes, coarse_edges, closest_centroid_indices, coarse_edge_attrs, distances


# def coarsen_graph(nodes, edges, length_scale, edge_attr):
#     voxel_centroids, closest_centroid_indices, distances = create_voxel_grid_with_distances(nodes, length_scale)
#     # node_assignments = assign_nodes_to_voxels(nodes, voxel_centroids)

#     coarse_nodes = {i: centroid for i, centroid in enumerate(voxel_centroids)}
#     coarse_edges = []
#     coarse_edge_attrs = []

#     edge_index_map = {}  # Maps voxel pairs to their corresponding fine edges
#     start_nodes = edges[0]  # Extract start nodes
#     end_nodes = edges[1]  # Extract end nodes
    
#     for i in range(edges.size(1)):
#         start_node = start_nodes[i].item()
#         end_node = end_nodes[i].item()
#         start_voxel = closest_centroid_indices[start_node]
#         end_voxel = closest_centroid_indices[end_node]
        
#         if start_voxel != end_voxel:
#             key = tuple(sorted([start_voxel, end_voxel]))
#             if key not in edge_index_map:
#                 edge_index_map[key] = []
#             edge_index_map[key].append(i)

#     # print('indices', type(indices), len(indices), '\n')

#     for key, indices in edge_index_map.items():
        
#         selected_attrs = edge_attr[0, indices, :]
#         avg_attr = selected_attrs.mean(dim=0)  # Compute mean across the selected edges
#         coarse_edges.append(key)
#         coarse_edge_attrs.append(avg_attr.unsqueeze(0)) 

#     # coarse_edge_attrs = torch.stack(coarse_edge_attrs, dim=0) if coarse_edge_attrs else torch.empty(0, edge_attr.size(1))
#     if coarse_edge_attrs:
#         # Ensure all tensors in the list are of the same shape for torch.stack
#         coarse_edge_attrs = torch.cat(coarse_edge_attrs, dim=0)
#     else:
#         coarse_edge_attrs = torch.empty(0, edge_attr.size(2)) 

#     return coarse_nodes, coarse_edges, closest_centroid_indices, coarse_edge_attrs, distances

# def coarsen_graph(nodes, edges, length_scale, edge_attr):
#     voxel_centroids, closest_centroid_indices, distances = create_voxel_grid_with_distances(nodes, length_scale)
#     # node_assignments = assign_nodes_to_voxels(nodes, voxel_centroids)
#     node_assignments = assign_nodes_to_voxels(nodes, voxel_centroids)

#     coarse_nodes = {i: centroid for i, centroid in enumerate(voxel_centroids)}
#     coarse_edges = []
#     coarse_edge_attrs = []

#     edge_index_map = {}  # Maps voxel pairs to their corresponding fine edges
#     for i, (start_node, end_node) in enumerate(edges):
#         # start_voxel = node_assignments[start_node]
#         # end_voxel = node_assignments[end_node]
#         start_voxel = closest_centroid_indices[start_node]
#         end_voxel = closest_centroid_indices[end_node]
#         if start_voxel != end_voxel:
#             key = tuple(sorted([start_voxel, end_voxel]))
#             if key not in edge_index_map:
#                 edge_index_map[key] = []
#             edge_index_map[key].append(i)

#     for key, indices in edge_index_map.items():
#         # Calculate the average attribute for edges between these voxels
#         avg_attr = edge_attr[indices].mean(dim=0)
#         coarse_edges.append(key)
#         coarse_edge_attrs.append(avg_attr)

#     coarse_edge_attrs = torch.stack(coarse_edge_attrs, dim=0) if coarse_edge_attrs else torch.empty(0, edge_attr.size(1))

#     # return nodes, edges, coarse_nodes, coarse_edges, closest_centroid_indices, coarse_edge_attrs
#     return coarse_nodes, coarse_edges, closest_centroid_indices, coarse_edge_attrs, distances

# def visualize_graphs(original_nodes, original_edges, coarse_nodes, coarse_edges, node_assignments):
#     plt.figure(figsize=(12, 6))
    
#     plt.subplot(1, 2, 1)
#     plt.title('Original Graph')
#     for edge in original_edges:
#         plt.plot(original_nodes[edge, 0], original_nodes[edge, 1], 'k-', alpha=0.5)
#     plt.scatter(original_nodes[:, 0], original_nodes[:, 1], c='b')
    
#     plt.subplot(1, 2, 2)
#     plt.title('Coarsened Graph')
#     for edge in coarse_edges:
#         plt.plot([coarse_nodes[edge[0]][0], coarse_nodes[edge[1]][0]], 
#                  [coarse_nodes[edge[0]][1], coarse_nodes[edge[1]][1]], 'k-', alpha=0.5)
#     plt.scatter([coarse_nodes[i][0] for i in coarse_nodes], 
#                 [coarse_nodes[i][1] for i in coarse_nodes], c='r')
    
#     for i, assignment in enumerate(node_assignments):
#         plt.plot([original_nodes[i, 0], coarse_nodes[assignment][0]],
#                  [original_nodes[i, 1], coarse_nodes[assignment][1]], 'g--', alpha=0.5)
    
#     plt.show()

# Adaptation for using `in_nodes` to get node positions
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# train_size = 70
# problem = 'AD'

# # Create dataset
# train_data, test_data = create_dataset(device, problem, train_size)

# # Select a simulation for visualization
# sim = 10  # Assuming we want to visualize the first simulation in test data

# # Assuming `meshes` is accessible and contains the mesh for each simulation
# # If not directly accessible here, you may need to adjust how you obtain the mesh object for the selected simulation
# mesh = test_data['mesh'][sim]  # or however you access the mesh object for the given simulation

# # Extracting node positions for interior nodes
# in_node_indices = test_data['in_nodes'][sim]  # Assuming this gives the indices of interior nodes
# node_positions = mesh.coordinates()[in_node_indices]  # Fetching positions for interior nodes

# # Extracting edge_index for the graph structure
# edge_index = test_data['edge_index'][sim].cpu().numpy()

# # Convert edge_index to a list of edge tuples
# edges = [(int(edge[0]), int(edge[1])) for edge in edge_index.T]

# edge_attr = torch.cat(test_data['edge_weights'], dim=0)

# node_attr = test_data['trajs'][sim]

# index_mapping = {original_index: new_index for new_index, original_index in enumerate(in_node_indices)}

# # Filter and remap edges to only include those with both nodes in `in_node_indices`
# filtered_edges = []
# for start_node, end_node in edges:
#     if start_node in index_mapping and end_node in index_mapping:
#         # Remap node indices to match their positions in the filtered list
#         remapped_start = index_mapping[start_node]
#         remapped_end = index_mapping[end_node]
#         filtered_edges.append((remapped_start, remapped_end))

# coarse_nodes, coarse_edges, node_assignments, coarse_edge_attrs = coarsen_graph(
#     node_positions, filtered_edges, 0.1, edge_attr
# )

# # Visualization remains the same; additional code required to visualize edge attributes if needed
# visualize_graphs(node_positions, filtered_edges, coarse_nodes, coarse_edges, node_assignments)
# # print(node_positions)
# # print(edges)

# print("nuovo test\n\n")

# in_channels = node_attr.size(-1)  # Assuming node_attr is a tensor
# hidden_channels = 2 * in_channels  # Example choice
# out_channels = in_channels  # Maintain the same size
# num_layers = 2  # Example choice

# Assume MLP and CoarsenGraphWithMLP are defined as discussed

# Initialize the module
# coarsen_mlp_module = CoarsenGraphWithMLP(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers)

# # Prepare a test sample
# test_node_positions = node_positions  # From your dataset preparation step
# test_edges = np.array(filtered_edges)  # Ensure this is in a NumPy array format
# test_edge_attr = edge_attr  # Assuming this is already prepared
# test_node_attr = node_attr  # Node features for the test sample

# # Convert test data to the required format if not already
# # e.g., test_node_positions might need conversion to a tensor

# # Run the coarsening and MLP feature processing
# coarse_node_features = coarsen_mlp_module(test_node_positions, test_edges, 0.05, test_node_attr, test_edge_attr)
# # def forward(self, nodes, edges, length_scale, node_features, edge_attr):
# # Print or visualize the results
# print("Coarse node features shape:", coarse_node_features.shape)
# # Additional visualization can be implemented as needed
