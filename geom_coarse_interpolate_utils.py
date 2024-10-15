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
# from core_model import MLP

from plot_utils.plot_nodes import plot_graph_positions, plot_mapping, plot_graph

plot = 0

device= torch.device("cuda" if torch.cuda.is_available() else "cpu")



def create_voxel_grid_with_distances(nodes, length_scale, batch_size):
    """
    Creates a voxel grid with distances for the given nodes.

    Args:
        nodes (torch.Tensor or array-like): The input nodes.
        length_scale (float): The length scale for creating the voxel grid.

    Returns:
        tuple: A tuple containing the final voxel centroids, the indices of the closest centroids for each node,
               and the distances between each node and its closest centroid.
    """
    # select nodeds for first batch
    # nodes_batch = nodes.size(0)//batch_size
    # nodes = nodes[:nodes_batch]

    if not isinstance(nodes, torch.Tensor):
        nodes = torch.tensor(nodes, dtype=torch.float)
    nodes = nodes.cpu()
    # min_x, min_y = torch.min(nodes, dim=0).values
    # max_x, max_y = torch.max(nodes, dim=0).values
    min_x, min_y, max_x, max_y = 0.0, 0.0, 1.0, 1.0
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
    fin_distances = torch.tensor(fin_distances, dtype=torch.float).unsqueeze(-1).to(device)

    if plot:
        final_voxel_centroids = final_voxel_centroids.to(device)
        plot_graph_positions(final_voxel_centroids, title="coarsen nodes", save_path='coarsen_nodes_{}.png'.format(length_scale))

    # Create a tensor to store the updated indices
    updated_fin_closest_centroid_indices = torch.zeros(batch_size * fin_closest_centroid_indices.size(0), dtype=torch.int64)

    for i in range(batch_size):
        # Add i * final_voxel_centroid.size(0) to fin_closest_centroid_indices and store it in the updated tensor
        updated_fin_closest_centroid_indices[i * fin_closest_centroid_indices.size(0): (i+1) * fin_closest_centroid_indices.size(0)] = fin_closest_centroid_indices + i * final_voxel_centroids.size(0)
    # Replace fin_closest_centroid_indices with the updated tensor
    fin_closest_centroid_indices = updated_fin_closest_centroid_indices.to(device)
    # Repeat fin_distances for each batch
    fin_distances = fin_distances.repeat(batch_size, 1).to(device)

    
    a=0

    

    return final_voxel_centroids, fin_closest_centroid_indices, fin_distances
    

def create_maps_distances(nodes, scales, edge_index, batch_size=1):
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
    nodes_coarse_list = {}

    if plot:
        plot_graph_positions(nodes, title="Original Graph", save_path='original_nodes.png')

    
    for length_scale in scales:
        nodes_coarse, fine2coarse, dist = create_voxel_grid_with_distances(nodes, length_scale, batch_size=batch_size)

        edge_index_coarse, fine2coarse_edges = coarsen_edges(edge_index, fine2coarse)

        fine2coarse_list[length_scale]=fine2coarse
        distances[length_scale]=dist
        nodes_coarse_list[length_scale]=nodes_coarse

        edge_index_coarse_list[length_scale]=edge_index_coarse
        fine2coarse_edges_list[length_scale]=fine2coarse_edges

        if plot > 1:
            plot_graph(nodes_coarse, edge_index_coarse, title="Coarsen Graph", save_path='coarsen_graph_{}.png'.format(length_scale))
            plot_mapping(nodes, nodes_coarse, fine2coarse, title="Mapping", save_path='mapping_{}.png'.format(length_scale))
        
        nodes = nodes_coarse
        edge_index = edge_index_coarse

 


    return fine2coarse_list, distances, edge_index_coarse_list, fine2coarse_edges_list, nodes_coarse_list


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