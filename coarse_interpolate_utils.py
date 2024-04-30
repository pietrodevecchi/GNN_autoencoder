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

from plot_nodes import plot_graph_positions, plot_mapping, plot_graph

plot = 0

device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''

def create_voxel_grid_with_distances(nodes, length_scale):
    """
    Creates a voxel grid with distances for the given nodes.

    Args:
        nodes (torch.Tensor or array-like): The input nodes.
        length_scale (float): The length scale for creating the voxel grid.

    Returns:
        tuple: A tuple containing the final voxel centroids, the indices of the closest centroids for each node,
               and the distances between each node and its closest centroid.
    """
    if not isinstance(nodes, torch.Tensor):
        nodes = torch.tensor(nodes, dtype=torch.float)
    nodes = nodes.cpu()
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
    
    fin_closest_centroid_indices = torch.tensor(fin_closest_centroid_indices, dtype=torch.long).to(device)
    fin_distances = torch.tensor(fin_distances, dtype=torch.float).unsqueeze(0).unsqueeze(-1).to(device)

    

    if plot:
        final_voxel_centroids = final_voxel_centroids.to(device)
        plot_graph_positions(final_voxel_centroids, title="coarsen nodes", save_path='coarsen_nodes_{}.png'.format(length_scale))

    return final_voxel_centroids, fin_closest_centroid_indices, fin_distances
    

def create_maps_distances(nodes, scales):
    """
    Creates maps and distances for given nodes and scales.

    Args:
        nodes (torch.Tensor or array-like): The input nodes.
        scales (list): List of length scales.

    Returns:
        tuple: A tuple containing two dictionaries:
            - fine2coarse_list: A dictionary mapping each length scale to its corresponding fine-to-coarse map.
            - distances: A dictionary mapping each length scale to its corresponding distances.

    """
    fine2coarse_list = {}
    distances = {}

    if plot:
        plot_graph_positions(nodes, title="Original Graph", save_path='original_nodes.png')

    if plot > 1:
        for length_scale in scales:
            nodes_coarse, fine2coarse, dist = create_voxel_grid_with_distances(nodes, length_scale)
            fine2coarse_list[length_scale]=fine2coarse
            distances[length_scale]=dist
            plot_mapping(nodes, nodes_coarse, fine2coarse, title="Mapping", save_path='mapping_{}.png'.format(length_scale))
            nodes = nodes_coarse

    else:
        for length_scale in scales:
            nodes, fine2coarse, dist = create_voxel_grid_with_distances(nodes, length_scale)
            fine2coarse_list[length_scale]=fine2coarse
            distances[length_scale]=dist


    return fine2coarse_list, distances



'''




    # def coarsen_graph(nodes, edges, length_scale, edge_attr):
#     """
#     Coarsens a graph by creating voxel grids and calculating averaged attributes for coarse edges.

#     Args:
#         nodes (torch.Tensor): Tensor representing the nodes of the graph.
#         edges (torch.Tensor): Tensor representing the edges of the graph.
#         length_scale (float): Length scale used for creating voxel grids.
#         edge_attr (torch.Tensor): Tensor representing the attributes of the edges.

#     Returns:
#         tuple: A tuple containing the following elements:
#             - coarse_nodes (torch.Tensor): Tensor representing the coarse nodes.
#             - coarse_edges (torch.Tensor): Tensor representing the coarse edges.
#             - closest_centroid_indices (torch.Tensor): Tensor representing the indices of the closest centroids.
#             - coarse_edge_attrs (torch.Tensor): Tensor representing the attributes of the coarse edges.
#             - distances (torch.Tensor): Tensor representing the distances between nodes and centroids.
#     """

#     batch_size = edge_attr.size(0)
#     num_attrs = edge_attr.size(2)  
#     # Use 'create_voxel_grid_with_distances' to get centroids, indices, and distances as PyTorch tensors
#     voxel_centroids, closest_centroid_indices, distances = create_voxel_grid_with_distances(nodes, length_scale)

#     # Coarse nodes are directly the voxel centroids
#     coarse_nodes = voxel_centroids

#     # Prepare lists for storing start and end voxels for coarse edges and their attributes
#     start_voxels_list = []
#     end_voxels_list = []
#     coarse_edge_attrs_list = []

#     # Initialize a dictionary for mapping voxel pairs to their corresponding fine edges
#     edge_index_map = {}

#     # Iterate over each edge to populate the map
#     for i in range(edges.size(1)):
#         start_node, end_node = edges[:, i]
#         start_voxel, end_voxel = closest_centroid_indices[start_node].item(), closest_centroid_indices[end_node].item()

#         if start_voxel != end_voxel:
#             key = tuple(sorted([start_voxel, end_voxel]))
#             edge_index_map.setdefault(key, []).append(i)

#     # Process each edge in the map to calculate averaged attributes
#     # for (start_voxel, end_voxel), indices in edge_index_map.items():
#     #     indices = torch.tensor(indices, dtype=torch.long)  # Use long for indexing
#     #     selected_attrs = edge_attr[:, indices].mean(dim=1)  # Average over the dimension of edges
#     #     # Append start and end voxels separately
#     #     start_voxels_list.append(start_voxel)
#     #     end_voxels_list.append(end_voxel)
#     #     coarse_edge_attrs_list.append(selected_attrs.unsqueeze(0))

#     for batch_idx in range(batch_size):
#         batch_edge_attrs_list = []
#         for (start_voxel, end_voxel), indices in edge_index_map.items():
#             indices_tensor = torch.tensor(indices, dtype=torch.long)
#             # Select and average attributes for current batch and indices
#             selected_attrs = edge_attr[batch_idx, indices_tensor].mean(dim=0)  # Shape: [8]
#             batch_edge_attrs_list.append(selected_attrs.unsqueeze(0))  # Add batch dimension back
#             if batch_idx==0:
#                 start_voxels_list.append(start_voxel)
#                 end_voxels_list.append(end_voxel)

#         # Concatenate all edge attributes for the current batch
#         if batch_edge_attrs_list:
#             batch_coarse_edge_attrs = torch.cat(batch_edge_attrs_list, dim=0).to(device)  # Shape: [M, 8] for current batch
#             coarse_edge_attrs_list.append(batch_coarse_edge_attrs.unsqueeze(0)) 

#     # Convert lists to tensors

#     if start_voxels_list:
#         # Convert lists to tensors and then stack to get shape [2, N]
#         start_voxels = torch.tensor(start_voxels_list, dtype=torch.long)
#         end_voxels = torch.tensor(end_voxels_list, dtype=torch.long)
#         coarse_edges = torch.stack([start_voxels, end_voxels], dim=0).to(device)
#     else:
#         coarse_edges = torch.empty((2, 0), dtype=torch.long)  # Ensure dtype matches index dtype

#     # print('coarse_edges: ', type(coarse_edges), coarse_edges.size(), '\n')

#     # print('coarse_edge_attrs_list: ', len(coarse_edge_attrs_list), len(coarse_edge_attrs_list[0]), \
#     #     len(coarse_edge_attrs_list[0][0]), '\n')

#     if coarse_edge_attrs_list:
#         coarse_edge_attrs = torch.cat(coarse_edge_attrs_list, dim=0) #.repeat(edge_attr.size(0), 1, 1) # Shape: [1, N, Attrs]
#     else:
#         coarse_edge_attrs = torch.empty((1, 0, edge_attr.size(-1)), dtype=edge_attr.dtype)

#     return coarse_nodes, coarse_edges, closest_centroid_indices, coarse_edge_attrs, distances


def create_voxel_grid_with_distances(nodes, length_scale):
    """
    Creates a voxel grid with distances for the given nodes.

    Args:
        nodes (torch.Tensor or array-like): The input nodes.
        length_scale (float): The length scale for creating the voxel grid.

    Returns:
        tuple: A tuple containing the final voxel centroids, the indices of the closest centroids for each node,
               and the distances between each node and its closest centroid.
    """
    if not isinstance(nodes, torch.Tensor):
        nodes = torch.tensor(nodes, dtype=torch.float)
    nodes = nodes.cpu()
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
    
    fin_closest_centroid_indices = torch.tensor(fin_closest_centroid_indices, dtype=torch.long).to(device)
    fin_distances = torch.tensor(fin_distances, dtype=torch.float).unsqueeze(0).unsqueeze(-1).to(device)

    

    if plot:
        final_voxel_centroids = final_voxel_centroids.to(device)
        plot_graph_positions(final_voxel_centroids, title="coarsen nodes", save_path='coarsen_nodes_{}.png'.format(length_scale))

    return final_voxel_centroids, fin_closest_centroid_indices, fin_distances
    

def create_maps_distances(nodes, scales, edge_index):
    """
    Creates maps and distances for given nodes and scales.

    Args:
        nodes (torch.Tensor or array-like): The input nodes.
        scales (list): List of length scales.

    Returns:
        tuple: A tuple containing two dictionaries:
            - fine2coarse_list: A dictionary mapping each length scale to its corresponding fine-to-coarse map.
            - distances: A dictionary mapping each length scale to its corresponding distances.

    """
    fine2coarse_list = {}
    distances = {}
    edge_index_coarse_list = {}
    fine2coarse_edges_list = {}

    if plot:
        plot_graph_positions(nodes, title="Original Graph", save_path='original_nodes.png')

    
    for length_scale in scales:
        nodes_coarse, fine2coarse, dist = create_voxel_grid_with_distances(nodes, length_scale)

        edge_index_coarse, fine2coarse_edges = coarsen_edges(edge_index, fine2coarse)

        fine2coarse_list[length_scale]=fine2coarse
        distances[length_scale]=dist

        edge_index_coarse_list[length_scale]=edge_index_coarse
        fine2coarse_edges_list[length_scale]=fine2coarse_edges

        if plot > 1:
            plot_graph(nodes_coarse, edge_index_coarse, title="Coarsen Graph", save_path='coarsen_graph_{}.png'.format(length_scale))
            plot_mapping(nodes, nodes_coarse, fine2coarse, title="Mapping", save_path='mapping_{}.png'.format(length_scale))
        
        nodes = nodes_coarse
        edge_index = edge_index_coarse

 


    return fine2coarse_list, distances, edge_index_coarse_list, fine2coarse_edges_list


def coarsen_edges(edges, fine2coarse_index):
    '''
    param edges:                        torch.Tensor with shape [2 ,n_edges].
    param fine2coarse_index:            torch.Tensor with shape [n_edges_coarse].
    param edge_attr:                    torch.Tensor with shape [batch_size-1, n_edges, hidden_channels].

    '''
    start_nodes, end_nodes = edges
    # if verbose:
    #     print('fine2coarse_index:', type(fine2coarse_index), fine2coarse_index.size(), fine2coarse_index.device, '\n')
    #     print('start_nodes:', type(start_nodes), start_nodes.size(), start_nodes.device, '\n')
    start_voxels = fine2coarse_index[start_nodes].to(device)
    end_voxels = fine2coarse_index[end_nodes].to(device)
    # if verbose:
    #     print('start_voxels:', type(start_voxels), start_voxels.size(), start_voxels.device, '\n')
    #     print('end_voxels:', type(end_voxels), end_voxels.size(), end_voxels.device, '\n')

    # Ensure edge_index_coarse and fine2coarse_edges are on the correct device
    # edge_index_coarse, fine2coarse_edges = torch.unique(torch.stack([torch.min(start_voxels, end_voxels), 
    #                                                         torch.max(start_voxels, end_voxels)], dim=0), 
    #                                             dim=1, return_inverse=True)

    edge_index_coarse, fine2coarse_edges = torch.unique(torch.stack([start_voxels, end_voxels], dim=0), dim=1, return_inverse=True)
    
    # edge_index_coarse = torch.cat([edge_index_coarse, edge_index_coarse.flip([0])], dim=1)
    # fine2coarse_edges = torch.cat([fine2coarse_edges, fine2coarse_edges+edges.size(1)], dim=0)sr
    a=0

    return edge_index_coarse, fine2coarse_edges