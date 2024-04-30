import torch
import torch.nn as nn
from torch_scatter import scatter_mean
from torch_scatter import scatter_add
from torch.nn import Linear, ModuleList
# from coarse_interpolate_utils import *
import tracemalloc
from scipy.spatial import cKDTree
import numpy as np

from plot_nodes import plot_graph_positions, plot_mapping, plot_graph

import time
import torch

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

verbose = 0
plot = 0
stopwatch = 0


class MLP(nn.Module):
    '''Class for creating a Multi-Layer Perceptron
          Attributes
            layers      (List)      A list of layers transforms a tensor x into f(Wx + b), where
                                    f is SiLU activation function, W is the weight matrix and b the bias tensor.


    '''
    def __init__(self, num_layers,in_channels,hidden_channels,out_channels):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_channels,hidden_channels))
        self.layers.append(nn.SiLU())
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_channels,hidden_channels))
            self.layers.append(nn.SiLU())
        self.layers.append(nn.Linear(hidden_channels, out_channels))

               
    def forward(self, x):
        '''
        :param x: torch.Tensor
        :return: torch.Tensor
        '''
        for layer in self.layers:
            x = layer(x)
        return x
    

class NodeModel(torch.nn.Module):
    '''Class for creating a model for the nodes
        Attributes
            node_mlp      (object)    A MLP object that combines and tranforms node and edge features
    '''
    def __init__(self, mlp_layers, hidden_channels):
        super(NodeModel, self).__init__()
        self.node_mlp = MLP(mlp_layers,2*hidden_channels,hidden_channels,hidden_channels)

    def forward(self, x, edge_index, edge_attr):
        '''
        :param x:           torch.Tensor with shape [batch_size-1,n_nodes,hidden_channels].
        :param edge_index:  torch.Tensor with shape [2,n_edges].
        :param edge_attr:   torch.Tensor with shape [batch_size-1,n_edges,hidden_channels].
        :return:            torch.Tensor with shape [batch_size-1,n_nodes,hidden_channels]
        '''
        _, dest = edge_index
        # qui somma i valori degli edges che hanno come destination lo stesso nodo,
        # risulta un tensore con righe quanti nodi dove c'e' la somma delle features degli
        # edges che arrivano in quel nodo

        # cerco di aggiungere edges come fossero undirected
        # counts_src = scatter_add(torch.ones_like(edge_attr), src, dim=1, dim_size=x.size(1))
        # counts_dest = scatter_add(torch.ones_like(edge_attr), dest, dim=1, dim_size=x.size(1))

        # # aggiungo normalizzazione
        # out_src = scatter_add(edge_attr, src, dim=1, dim_size=x.size(1)) / counts_src.clamp(min=1)
        # out_dest = scatter_add(edge_attr, dest, dim=1, dim_size=x.size(1)) / counts_dest.clamp(min=1)

        # out_sum = out_src + out_dest

        out_sum = scatter_add(edge_attr, dest, dim=1, dim_size=x.size(1))


        out = torch.cat([x, out_sum], dim=2)
        out = self.node_mlp(out)
        return out


class EdgeModel(torch.nn.Module):
    '''Class for creating a model for the edges
        Attributes
            edge_mlp      (object)    A MLP object that tranforms edge features

    '''
    def __init__(self, mlp_layers, hidden_channels):
        super(EdgeModel, self).__init__()
        self.edge_mlp = MLP(mlp_layers, 3*hidden_channels,hidden_channels,hidden_channels)

    def forward(self, src, dest, edge_attr):
        '''
        :param src:         torch.Tensor with shape [batch_size-1,n_edges,hidden_channels]. Node features
                            corresponding to source nodes of the edges.
        :param dest:        torch.Tensor with shape [batch_size-1,n_edges,hidden_channels]. Node features
                            corresponding to destination nodes of the edges.
        :param edge_attr:   torch.Tensor with shape [batch_size-1,n_edges,hidden_channels].
        :return:            torch.Tensor with shape [batch_size-1,n_edges,hidden_channels].
        '''
        out = torch.cat([edge_attr, src, dest], dim=2)
        out = self.edge_mlp(out)
        return out
    


class MPLayer(torch.nn.Module):
    '''Class for creating a single message passing layer
    Attributes
            edge_model      (object)    A edge_model object that transforms the current edge_features
            node_model      (object)    A node_model object that combines node_features and edge_features and
                                        transforms them.


    '''

    def __init__(self, edge_model=None, node_model=None):
        super().__init__()
        self.edge_model = edge_model
        self.node_model = node_model

    def forward(self, x, edge_index, edge_attr):
        '''Performs a message passing forward pass

        :param x:           torch.Tensor with shape [batch_size-1,n_nodes,hidden_channels].
        :param edge_index:  torch.Tensor with shape [2,n_edges].
        :param edge_attr:   torch.Tensor with shape [batch_size-1,n_edges,hidden_channels]
        :return:            (torch.Tensor with shape [batch_size-1,n_nodes,hidden_channels],
                            torch.Tensor with shape [batch_size-1,n_nodes,hidden_channels])
        '''


        src = edge_index[0]
        dest = edge_index[1]

        edge_attr = self.edge_model(x[:,src], x[:,dest], edge_attr)
        x = self.node_model(x, edge_index, edge_attr)

        return x, edge_attr
    

class MP_block(torch.nn.Module):

    def __init__(self, MPblock_layers, mlp_layers, hidden_channels):
        super(MP_block, self).__init__()

        self.message_passing_layers = nn.ModuleList()

        for _ in range(MPblock_layers):
            node_model = NodeModel(mlp_layers, hidden_channels)
            edge_model = EdgeModel(mlp_layers, hidden_channels)
            self.message_passing_layers.append(MPLayer(node_model=node_model, edge_model=edge_model))

    def forward(self, x, edge_index, edge_attr, node_positions):

        for GraphNet in self.message_passing_layers:
            x, edge_attr = GraphNet(x, edge_index, edge_attr)

        return x, edge_attr



class Down_NodeModel(torch.nn.Module):
    '''Class for creating a model for the nodes coarsening
        Attributes
            node_mlp      (object)    A MLP object that combines and tranforms node and edge features
    '''
    def __init__(self, mlp_layers, hidden_channels):
        super(Down_NodeModel, self).__init__()
        self.node_mlp = MLP(mlp_layers,hidden_channels+1,hidden_channels,hidden_channels)

    def forward(self, x, distances):
        '''

        # :param x:           torch.Tensor with shape [batch_size-1,n_nodes,hidden_channels].
        # :param distances:   torch.Tensor with shape [1, n_nodes, 1] distance between the node and its 
        # :return:            torch.Tensor with shape [batch_size-1,n_nodes,hidden_channels]
        # '''

        if verbose > 1:
            print('distances_down_model:', type(distances), distances.size(), distances.device, '\n')
            print('x_down_model:', type(x), x.size(), x.device, '\n')        

        # Now you can concatenate 'x' and 'distances_expanded' along dim=2
        out = torch.cat([x, distances.expand(x.size(0), -1, -1)], dim=2)
        
        out = self.node_mlp(out)

        return out


class Up_NodeModel(torch.nn.Module):
    '''Class for creating a model for the nodes interpolating
        Attributes
            node_mlp      (object)    A MLP object that combines and tranforms node and edge features
    '''
    def __init__(self, mlp_layers, hidden_channels):
        super(Up_NodeModel, self).__init__()
        self.node_mlp = MLP(mlp_layers,2*hidden_channels+1,hidden_channels,hidden_channels)

    def forward(self, x, x_scale, fine2coarse_index, distances):
        '''

        :param x:                 torch.Tensor with shape [batch_size-1,n_nodes,hidden_channels].
        :param x_scale:           torch.Tensor with shape [batch_size-1,n_nodes,hidden_channels].
        :param distances:         torch.Tensor with shape [1,n_nodes,1].
        :param fine2coarse_index: torch.Tensor with shape [n_nodes].
        :return:                  torch.Tensor with shape [batch_size-1,n_nodes,hidden_channels]
        '''

        x = x[:, fine2coarse_index, :]

        if(verbose>1):
            print('up fine2coarse_index: ', type(fine2coarse_index), fine2coarse_index.size(), '\n')
            print('up x: ', type(x), x.size(), '\n')
            print('up x_scale: ', type(x_scale), x_scale.size(), '\n')
            print('up distances: ', type(distances), distances.size(), '\n')

        out = torch.cat([x, x_scale, distances.expand(x.size(0), -1, -1)], dim = 2)
        
        out = self.node_mlp(out)

        return out
    

class Coarse_layer(nn.Module):
    def __init__(self, down_node_model=None):

        super(Coarse_layer, self).__init__()
        self.down_node_model = down_node_model

    def forward(self, x, coarse_edge_index, edge_attr, scale, fine2coarse_index, fine2coarse_edges,
                distances, pool_indices=None, pool_edges=None, num_nodes=None, num_edges=None):
        '''Performs a coarsening operation 

        :param x:                   torch.Tensor with shape [batch_size-1,n_nodes,hidden_channels].
        :param edge_index:          torch.Tensor with shape [2, n_edges].
        :param edge_attr:           torch.Tensor with shape [batch_size-1,n_edges,hidden_channels].
        :param fine2coarse_index:   torch.Tensor with shape [n_edges_coarse].
        :distances:                 torch.Tensor with shape [1, n_edges_coarse, 1]

        :return:                    torch.Tensor with shape [batch_size-1,n_nodes,hidden_channels]
        '''

        removed_indices = None

        if pool_indices is not None:
            # Update fine2coarse_index and distances
            fine2coarse_index = fine2coarse_index[pool_indices[0]]
            distances = distances[:, pool_indices[0]]

            ''''''
            if verbose > 1:
                print('pool_edges:', type(pool_edges), pool_edges.size(), pool_edges.device, '\n')

            fine2coarse_edges = fine2coarse_edges[pool_edges]
            ''''''

        '''
        coarse_edge_index, coarse_edge_attrs = self.coarsen_edges(edge_index, fine2coarse_index, edge_attr)
        '''
        coarse_edge_attrs = self.coarsen_edges2(coarse_edge_index, fine2coarse_index, edge_attr, fine2coarse_edges, num_edges)

        fine_node_process = self.down_node_model(x, distances)

        if num_nodes is not None:
            coarse_node_attrs = scatter_add(fine_node_process, fine2coarse_index, dim=1 , dim_size=num_nodes)
            counts = scatter_add(torch.ones_like(fine_node_process), fine2coarse_index, dim=1 , dim_size=num_nodes)

        else:
            coarse_node_attrs = scatter_add(fine_node_process, fine2coarse_index, dim=1 , dim_size=torch.max(fine2coarse_index).item()+1)
            counts = scatter_add(torch.ones_like(fine_node_process), fine2coarse_index, dim=1 , dim_size=torch.max(fine2coarse_index).item()+1)
        
        coarse_node_attrs_avg = coarse_node_attrs / counts.clamp(min=1)

        if verbose>1:
            print('fine2coarse_index:', type(fine2coarse_index), fine2coarse_index.size(), fine2coarse_index.device, '\n')
            print('distances:', type(distances), distances.size(), distances.device, '\n')
            print('x_coarse_node', type(coarse_node_attrs_avg), coarse_node_attrs_avg.size(), coarse_node_attrs_avg.device, '\n')
            print('removed_indices:', type(removed_indices), removed_indices, '\n')


        # return coarse_node_attrs_avg, coarse_edge_attrs, coarse_edge_index, removed_indices
        return coarse_node_attrs_avg, coarse_edge_attrs, coarse_edge_index


    def coarsen_edges2(self, coarse_edge_index, fine2coarse_index, edge_attr, fine2coarse_edges, num_edges):
        '''
        param edges:                        torch.Tensor with shape [2 ,n_edges].
        param fine2coarse_index:            torch.Tensor with shape [n_edges_coarse].
        param edge_attr:                    torch.Tensor with shape [batch_size-1, n_edges, hidden_channels].

        '''
        '''
        start_nodes, end_nodes = edges
        if verbose:
            print('fine2coarse_index:', type(fine2coarse_index), fine2coarse_index.size(), fine2coarse_index.device, '\n')
            print('start_nodes:', type(start_nodes), start_nodes.size(), start_nodes.device, '\n')
        start_voxels = fine2coarse_index[start_nodes].to(device)
        end_voxels = fine2coarse_index[end_nodes].to(device)
        if verbose:
            print('start_voxels:', type(start_voxels), start_voxels.size(), start_voxels.device, '\n')
            print('end_voxels:', type(end_voxels), end_voxels.size(), end_voxels.device, '\n')

        # Ensure coarse_edge_index and fine2coarse_edges are on the correct device
        coarse_edge_index, fine2coarse_edges = torch.unique(torch.stack([torch.min(start_voxels, end_voxels), 
                                                                torch.max(start_voxels, end_voxels)], dim=0), 
                                                    dim=1, return_inverse=True)

        '''

        if verbose>1:
            print('coarse_edge_index:', type(coarse_edge_index), coarse_edge_index.size(), coarse_edge_index.device, '\n')
            print('fine2coarse_edges:', type(fine2coarse_edges), fine2coarse_edges.size(), fine2coarse_edges.device, '\n')
            print('edge_attr:', type(edge_attr), edge_attr.size(), edge_attr.device, '\n')
            print('fine2coarse_index:', type(fine2coarse_index), fine2coarse_index.size(), fine2coarse_index.device, '\n')

        batch_size, num_edges_fine, _ = edge_attr.shape

        # Prepare for batched operation, ensuring tensors are on the correct device
        batch_indices = torch.arange(batch_size, device=device).view(-1, 1).expand(-1, num_edges_fine).reshape(-1).to(device)
        # edge_indices_expanded = fine2coarse_edges.unsqueeze(0).repeat(batch_size, 1)
        edge_indices_expanded = fine2coarse_edges.repeat(batch_size)

        # Initialize coarse_edge_attrs tensor on the correct device
        # coarse_edge_attrs = torch.zeros(batch_size * coarse_edge_index.size(1), edge_attr.size(-1), device=device)
        if num_edges is None:
            num_edges = coarse_edge_index.size(1)

        coarse_edge_attrs = torch.zeros(batch_size * num_edges, edge_attr.size(-1), device=device)

        # Use index_add_ ensuring operands are on the same device
        # coarse_edge_attrs.index_add_(0, batch_indices * coarse_edge_index.size(1) + edge_indices_expanded, 
        #                             edge_attr.reshape(-1, edge_attr.size(-1)))

        coarse_edge_attrs.index_add_(0, batch_indices * num_edges + edge_indices_expanded, 
                                    edge_attr.reshape(-1, edge_attr.size(-1)))

        # Post-processing steps, ensuring tensor operations remain on the correct device
        # coarse_edge_attrs = coarse_edge_attrs.view(batch_size, coarse_edge_index.size(1), -1)
        coarse_edge_attrs = coarse_edge_attrs.view(batch_size, num_edges, -1)

        # flat_indices = batch_indices * coarse_edge_index.size(1) + edge_indices_expanded
        # counts = torch.bincount(flat_indices, minlength=batch_size * coarse_edge_index.size(1)).to(device)
        
        # # Reshape counts to match `coarse_edge_attrs` structure for correct averaging
        # counts_reshaped = counts.view(batch_size, coarse_edge_index.size(1), 1)  # Add an extra dimension for broadcasting

        flat_indices = batch_indices * num_edges + edge_indices_expanded
        counts = torch.bincount(flat_indices, minlength=batch_size * num_edges).to(device)
        counts_reshaped = counts.view(batch_size, num_edges, 1)  # Add an extra dimension for broadcasting

        coarse_edge_attrs_avg = coarse_edge_attrs / counts_reshaped.clamp(min=1)

        return coarse_edge_attrs_avg

    '''

    def coarsen_edges(self, edges, fine2coarse_index, edge_attr):
        
        # param edges:                        torch.Tensor with shape [2 ,n_edges].
        # param fine2coarse_index:            torch.Tensor with shape [n_edges_coarse].
        # param edge_attr:                    torch.Tensor with shape [batch_size-1, n_edges, hidden_channels].

        
        start_nodes, end_nodes = edges
        if verbose:
            print('fine2coarse_index:', type(fine2coarse_index), fine2coarse_index.size(), fine2coarse_index.device, '\n')
            print('start_nodes:', type(start_nodes), start_nodes.size(), start_nodes.device, '\n')
        start_voxels = fine2coarse_index[start_nodes].to(device)
        end_voxels = fine2coarse_index[end_nodes].to(device)
        if verbose:
            print('start_voxels:', type(start_voxels), start_voxels.size(), start_voxels.device, '\n')
            print('end_voxels:', type(end_voxels), end_voxels.size(), end_voxels.device, '\n')

        # Ensure coarse_edge_index and fine2coarse_edges are on the correct device
        coarse_edge_index, fine2coarse_edges = torch.unique(torch.stack([torch.min(start_voxels, end_voxels), 
                                                                torch.max(start_voxels, end_voxels)], dim=0), 
                                                    dim=1, return_inverse=True)
        batch_size, num_edges, _ = edge_attr.shape

        # Prepare for batched operation, ensuring tensors are on the correct device
        batch_indices = torch.arange(batch_size, device=device).view(-1, 1).expand(-1, num_edges).reshape(-1).to(device)
        edge_indices_expanded = fine2coarse_edges.repeat(batch_size)

        # Initialize coarse_edge_attrs tensor on the correct device
        coarse_edge_attrs = torch.zeros(batch_size * coarse_edge_index.size(1), edge_attr.size(-1), device=device)

        # Use index_add_ ensuring operands are on the same device
        coarse_edge_attrs.index_add_(0, batch_indices * coarse_edge_index.size(1) + edge_indices_expanded, 
                                    edge_attr.reshape(-1, edge_attr.size(-1)))

        # Post-processing steps, ensuring tensor operations remain on the correct device
        coarse_edge_attrs = coarse_edge_attrs.view(batch_size, coarse_edge_index.size(1), -1)

        flat_indices = batch_indices * coarse_edge_index.size(1) + edge_indices_expanded
        counts = torch.bincount(flat_indices, minlength=batch_size * coarse_edge_index.size(1)).to(device)
        
        # Reshape counts to match `coarse_edge_attrs` structure for correct averaging
        counts_reshaped = counts.view(batch_size, coarse_edge_index.size(1), 1)  # Add an extra dimension for broadcasting

        coarse_edge_attrs_avg = coarse_edge_attrs / counts_reshaped.clamp(min=1)

        return coarse_edge_index, coarse_edge_attrs_avg

    '''




class Interpolate_layer(torch.nn.Module):

    def __init__(self, up_node_model=None):
        super().__init__()
        self.up_node_model = up_node_model

    def forward(self, x, x_scale, fine2coarse_index, distances, pool_indices=None):

        '''Performs an interpolation 

        :param x:                   torch.Tensor with shape [batch_size-1,n_nodes,hidden_channels].
        :param x_scale:             torch.Tensor with shape [batch_size-1,n_nodes,hidden_channels].
        :param fine2coarse_index:   torch.Tensor with shape [n_nodes].
        :param distances:           torch.Tensor with shape [1,n_nodes,1].
        :return:                    torch.Tensor with shape [batch_size-1,n_nodes,hidden_channels]
        '''
        if pool_indices is not None:
            # AAAA capire pool_indices giusti da passare, qua stiamo facendo casino
            fine2coarse_index = fine2coarse_index[pool_indices[0]]
            distances = distances[:, pool_indices[0]]

        if x_scale.size(1) != distances.size(1):
            print("beware distances shrinked form {} to {} \n".format(distances.size(1), x_scale.size(1)))
            distances = distances[:, :x_scale.size(1), :]
            fine2coarse_index = fine2coarse_index[:x_scale.size(1)]
            

        interp_node_attrs = self.up_node_model(x, x_scale, fine2coarse_index, distances)

        return interp_node_attrs





class MultiscaleMessagePassing(nn.Module):
    def __init__(self, scales, mlp_layers, MPblock_layers, hidden_channels):
        super(MultiscaleMessagePassing, self).__init__()
        self.fine2coarse_list = {}
        self.fine2coarse_edges_list = {}
        self.edge_index_coarse_list = {}
        self.distances_list = {}
        self.scales = scales
        # self.message_passing_layers_down1 = nn.ModuleList()
        # self.message_passing_layers_down2 = nn.ModuleList()
        # self.message_passing_layers_up1 = nn.ModuleList()
        # self.message_passing_layers_up2 = nn.ModuleList()
        self.message_passing_blocks = nn.ModuleList()
        self.coarse_layers = nn.ModuleList()
        self.interpolate_layers = nn.ModuleList()

        GraphNet = MP_block(MPblock_layers, mlp_layers, hidden_channels)
        self.message_passing_blocks.append(GraphNet)
        

        for scale in scales:

            down_model = Down_NodeModel(mlp_layers, hidden_channels)
            self.coarse_layers.append(Coarse_layer(down_node_model=down_model))

            GraphNet = MP_block(MPblock_layers, mlp_layers, hidden_channels)
            self.message_passing_blocks.append(GraphNet)

        GraphNet = MP_block(MPblock_layers, mlp_layers, hidden_channels)
        self.message_passing_blocks.append(GraphNet)

        for scale in scales:

            up_model = Up_NodeModel(mlp_layers, hidden_channels)
            self.interpolate_layers.append(Interpolate_layer(up_node_model=up_model))

            GraphNet = MP_block(MPblock_layers, mlp_layers, hidden_channels)
            self.message_passing_blocks.append(GraphNet)


    def forward(self, x, edge_index, edge_attr, node_positions, pool_indices=None, pool_edges=None):
        """
        Performs the forward pass of the model.

        Args:
            x (Tensor): Input tensor representing node features.
            edge_index (LongTensor): Graph connectivity in COO format.
            edge_attr (Tensor): Edge features.
            node_positions (Tensor): Node positions.
            pool_indices (Tensor, optional): Indices of nodes to be pooled.

        Returns:
            x (Tensor): Output tensor representing node features.
            edge_attr (Tensor): Output tensor representing edge features.
        """
        x_list = {}
        edge_attr_list = {}
        edge_index_list = {}

        pool_indices_local = pool_indices
        

        # Coarsen graph and perform message passing at each scale
        # for scale, coarse_layer, mp_layer_down1, mp_layer_down2 in zip(self.scales, self.coarse_layers, 
        #                                                self.message_passing_layers_down1[:-1], self.message_passing_layers_down2[:-1]):

        for scale, coarse_layer, MP_block in zip(self.scales, self.coarse_layers, 
                                                    self.message_passing_blocks[:len(self.scales)]):
        
            if (verbose>1):
                print('x layer_down:', type(x), x.size(), x.device, '\n')   
                print('edge_index layer_down:', type(edge_index), edge_index.size(), edge_index.device, '\n')
                print('edge_attr layer_down:', type(edge_attr), edge_attr.size(), edge_attr.device, '\n')         
            # x, edge_attr = mp_layer_down1(x, edge_index, edge_attr)
            # x, edge_attr = mp_layer_down2(x, edge_index, edge_attr)
            x, edge_attr = MP_block(x, edge_index, edge_attr, node_positions)
            

            x_list[scale]=x
            edge_attr_list[scale]= edge_attr
            edge_index_list[scale] = edge_index


            if scale is not self.scales[-1]:
                num_nodes = self.fine2coarse_list[self.scales[-1]].size(0)
                num_edges = self.fine2coarse_edges_list[self.scales[-1]].size(0)
                x, edge_attr, edge_index = \
                                coarse_layer(x, self.edge_index_coarse_list[scale], edge_attr, scale,
                                self.fine2coarse_list[scale], self.fine2coarse_edges_list[scale], 
                                self.distances_list[scale], pool_indices=pool_indices_local, pool_edges=pool_edges,
                                num_nodes=num_nodes, num_edges=num_edges)


            else:
                num_nodes = None
                num_edges = None
                x, edge_attr, edge_index= \
                            coarse_layer(x, self.edge_index_coarse_list[scale], edge_attr, scale,
                            self.fine2coarse_list[scale], self.fine2coarse_edges_list[scale], 
                            self.distances_list[scale], pool_indices=pool_indices_local, pool_edges=pool_edges,
                            num_nodes=num_nodes, num_edges=num_edges)
            
            pool_indices_local = None


        # x, edge_attr = self.message_passing_layers_down1[-1](x, edge_index, edge_attr)
        # x, edge_attr = self.message_passing_layers_down2[-1](x, edge_index, edge_attr)
        x, edge_attr = self.message_passing_blocks[len(self.scales)](x, edge_index, edge_attr, node_positions)

        # x, edge_attr = self.message_passing_layers_up1[0](x, edge_index, edge_attr)
        # x, edge_attr = self.message_passing_layers_up2[0](x, edge_index, edge_attr)        
        x, edge_attr = self.message_passing_blocks[len(self.scales)+1](x, edge_index, edge_attr, node_positions)

        for scale, interpolate_layer, MP_block in zip(reversed(self.scales), self.interpolate_layers, self.message_passing_blocks[len(self.scales)+2:]):
            
            if scale == self.scales[0]:
                pool_indices_local = pool_indices

            # if removed_indices is not None:
            #     fine2coarse_index = self.fine2coarse_list[scale][~removed_indices]
            #     distances = self.distances_list[scale][:, ~removed_indices]
            #     x = interpolate_layer(x, x_list[scale], fine2coarse_index, distances, pool_indices=pool_indices_local)
            #     removed_indices = None

            # else:
            x = interpolate_layer(x, x_list[scale], self.fine2coarse_list[scale], self.distances_list[scale], pool_indices=pool_indices_local)
            
            # x, edge_attr = mp_layer_up1(x, edge_index_list[scale], edge_attr_list[scale])
            # x, edge_attr = mp_layer_up2(x, edge_index_list[scale], edge_attr_list[scale])
            x, edge_attr = MP_block(x, edge_index_list[scale], edge_attr_list[scale], node_positions)
        
        return x, edge_attr

    def assign(self, fine2coarse_list, distances_list, edge_index_coarse_list, fine2coarse_edges_list):
        self.fine2coarse_list = fine2coarse_list
        self.distances_list = distances_list
        self.edge_index_coarse_list = edge_index_coarse_list
        self.fine2coarse_edges_list = fine2coarse_edges_list

        return
    

    


class Pool(nn.Module):
    def __init__(self, k, in_dim, p):
        """
            k (int): The value of k.
            in_dim (int): The input dimension.
            p (float): The dropout probability.
        """
        super(Pool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Node features.            [batch_size, num_nodes, num_node_features]
            edge_index (torch.Tensor): Edge indices.    [2, num_edges]
            edge_attr (torch.Tensor): Edge attributes.  [batch_size, num_edges, num_edge_features]

        Returns:
            torch.Tensor: New node features.            [batch_size, num_nodes', num_node_features]
            torch.Tensor: New edge indices.             [batch_size, 2, num_edges']
            torch.Tensor: New ordered edge indices.     [batch_size, 2, num_edges']
            torch.Tensor: New edge attributes.          [batch_size, num_edges', num_edge_features]
            torch.Tensor: Pool indices.                 [batch_size, num_nodes']
            torch.Tensor: Ordered pool indices.         [batch_size, num_nodes']
        """
        # Apply dropout and project node features to get scores, then apply sigmoid
        Z = self.drop(x)
        weights = self.proj(Z).squeeze(-1)
        scores = self.sigmoid(weights)

        # Compute the threshold for top-k selection based on scores
        kth_scores, _ = torch.kthvalue(scores, int((1 - self.k) * scores.size(1)), dim=1, keepdim=True)
        mask = scores >= kth_scores

        # Use the mask to determine the indices of selected nodes
        _, pool_indices = mask.float().sort(descending=True)
        pool_indices = pool_indices[:, :int(x.size(1) * self.k)]  # Keep top k% indices

        # Select nodes based on the pool_indices
        batch_indices = torch.arange(x.size(0), device=x.device).unsqueeze(1).expand_as(pool_indices)
        new_x = x[batch_indices, pool_indices]

        # Create a mask for nodes that are selected for pooling
        mask = torch.zeros(x.size(1), dtype=torch.bool, device=x.device)
        mask[pool_indices] = True

        # Create a mask for edges that connect the nodes left in new_x
        pool_edges = mask[edge_index[0]] & mask[edge_index[1]]

        # Select edges based on the pool_edges for the first batch
        new_edge_index = edge_index[:, pool_edges]

        # Select edge attributes based on the pool_edges for the first batch
        new_edge_attr = edge_attr[:, pool_edges]

        # Repeat new_edge_attr batch_size times
        # new_edge_attr = new_edge_attr.unsqueeze(0).repeat(edge_attr.size(0), 1, 1)


        # Create a mapping from old indices to new indices
        mapping = torch.full((x.size(1),), -1, device=x.device)
        mapping[pool_indices] = torch.arange(pool_indices.size(1), device=x.device)

        # Remap the node indices in new_edge_index
        new_edge_index_ordered = mapping[edge_index[:, pool_edges]]

        # pool_indices_ordered = mapping[pool_indices]

        # # Create a mask for nodes that are selected for pooling
        # mask = torch.zeros(x.size(0), x.size(1), dtype=torch.bool, device=x.device)
        # mask[torch.arange(x.size(0))[:, None], pool_indices] = True

        # # Create a mask for edges that connect the nodes left in new_x
        # pool_edges = mask[edge_index[0]] & mask[edge_index[1]]

        # # Select edges based on the pool_edges
        # new_edge_index = edge_index[:, pool_edges]

        # # Select edge attributes based on the pool_edges
        # new_edge_attr = edge_attr[pool_edges]

        # # Create a mapping from old indices to new indices
        # mapping = torch.full((x.size(0), x.size(1)), -1, device=x.device)
        # mapping[torch.arange(x.size(0))[:, None], pool_indices] = torch.arange(pool_indices.size(1))

        # # Remap the node indices in new_edge_index
        # new_edge_index_ordered = mapping[new_edge_index]

        # # Remap the node indices in pool_indices
        # pool_indices_ordered = mapping[torch.arange(x.size(0))[:, None], pool_indices]

        # # Select nodes for pooling
        # new_x = x[torch.arange(x.size(0))[:, None], pool_indices]

        # verbose = 1

        if verbose>1:
            print('new_x:', type(new_x), new_x.size(), new_x.device, '\n')
            print('new_edge_index:', type(new_edge_index), new_edge_index.size(), new_edge_index.device, '\n')
            print('new_edge_attr:', type(new_edge_attr), new_edge_attr.size(), new_edge_attr.device, '\n')
            print('pool_indices:', type(pool_indices), pool_indices.size(), pool_indices.device, '\n')
            print('new_edge_index_ordered:', type(new_edge_index_ordered), new_edge_index_ordered.size(), new_edge_index_ordered.device, '\n')

        # return new_x, new_edge_index, new_edge_index_ordered, new_edge_attr, pool_indices, pool_indices_ordered
        return new_x, new_edge_index, new_edge_index_ordered, new_edge_attr, pool_indices, pool_edges


class Unpool(nn.Module):
    def __init__(self):
        super(Unpool, self).__init__()

    # def forward(self, x, pool_indices, n_nodes):
    #     """
    #     Unpools the features by reconstructing the original shape with unpooled features correctly placed.

    #     Args:
    #         x (torch.Tensor): The pooled features with shape [batch_size, n_nodes', hidden_channels].
    #         pool_indices (torch.Tensor): The indices of the original nodes before pooling, with shape [batch_size, n_nodes'].
    #         n_nodes (int): The number of nodes in the original data before pooling.

    #     Returns:
    #         torch.Tensor: The unpooled features with shape [batch_size, n_nodes, hidden_channels].
    #     """
    #     # Initialize tensor to hold unpooled features, filling with zeros
    #     new_x = torch.zeros(x.size(0), n_nodes, x.size(2), device=x.device, dtype=x.dtype)

    #     # Reconstruct the original shape with unpooled features correctly placed
    #     for b in range(x.size(0)):
    #         new_x[b, pool_indices[b], :] = x[b, :, :]

    #     return new_x

    def forward(self, x, unpooled_edge_index, edge_attr, pool_indices, n_nodes):
        """
        Unpools the features and edge attributes by reconstructing the original shape with unpooled features and edge attributes correctly placed.

        Args:
            x (torch.Tensor): The pooled features with shape [batch_size, n_nodes', hidden_channels].
            unpooled_edge_index (torch.Tensor): The edge indices of the bigger (unpooled) graph.
            edge_attr (torch.Tensor): The edge attributes of the pooled graph.
            pool_indices (torch.Tensor): The indices of the original nodes before pooling, with shape [batch_size, n_nodes'].
            n_nodes (int): The number of nodes in the original data before pooling.

        Returns:
            torch.Tensor: The unpooled features with shape [batch_size, n_nodes, hidden_channels].
            torch.Tensor: The new edge attributes with shape [2, n_edges, edge_attr_dim], where n_edges is the number of edges in the unpooled graph, and edge_attr_dim is the dimension of edge attributes.
        """
        # Initialize tensor to hold unpooled features, filling with zeros
        new_x = torch.zeros(x.size(0), n_nodes, x.size(2), device=x.device, dtype=x.dtype)

        # Initialize tensor to hold new edge attributes, filling with zeros
        new_edge_attr = torch.zeros(edge_attr.size(0), unpooled_edge_index.size(1), edge_attr.size(2), device=edge_attr.device, dtype=edge_attr.dtype)

        # Reconstruct the original shape with unpooled features correctly placed
        for b in range(x.size(0)):
            new_x[b, pool_indices[b], :] = x[b, :, :]

            # Get the indices of edges that correspond to the pooled graph
            mask_source = torch.isin(unpooled_edge_index[0], pool_indices[b])
            mask_target = torch.isin(unpooled_edge_index[1], pool_indices[b])
            pooled_edge_indices = mask_source & mask_target
            # Assign the attributes from edge_attr to the corresponding edges in new_edge_attr
            new_edge_attr[:, pooled_edge_indices, :] = edge_attr[b, :, :]

        return new_x, new_edge_attr


    
class GNN(torch.nn.Module):
    """Class for creating a Graph Neural Network
            Attributes
                encoder_node    (object)    A MLP object that encodes node input features.
                encoder_edge    (object)    A MLP object that encodes edge input features.
                processor       (List)      A list of MMPLayer objects of length mp_steps that propagate
                                            the messages across the mesh nodes.
                decoder         (Object)    A MLP object that decodes the output features.

    """
    def __init__(self, args):
        super(GNN, self).__init__()


        # Encoder MLPs
        self.encoder_node = MLP(args.mlp_layers,args.in_node,args.hidden_channels,args.hidden_channels)
        self.encoder_edge = MLP(args.mlp_layers,args.in_edge,args.hidden_channels,args.hidden_channels)
        # # Pooling layer
        # self.pool = Pool(k=args.pool_k, in_dim=args.hidden_channels, p=args.dropout_p)
        # # Processor MLPs
        self.processor = nn.ModuleList()
        # for _ in range(args.mp_steps):
        #     GraphNet = MultiscaleMessagePassing(args.scales, args.mlp_layers, args.MPblock_layers , args.hidden_channels)
        #     self.processor.append(GraphNet)
        # # Unpooling layer
        # self.unpool = Unpool()

        GraphNet = MultiscaleMessagePassing(args.scales, args.mlp_layers, args.MPblock_layers , args.hidden_channels)
        self.processor.append(GraphNet)
        self.pool = Pool(k=args.pool_k, in_dim=args.hidden_channels, p=args.dropout_p)
        GraphNet = MultiscaleMessagePassing(args.scales, args.mlp_layers, args.MPblock_layers , args.hidden_channels)
        self.processor.append(GraphNet)
        GraphNet = MultiscaleMessagePassing(args.scales, args.mlp_layers, args.MPblock_layers , args.hidden_channels)
        self.processor.append(GraphNet)
        self.unpool = Unpool()
        GraphNet = MultiscaleMessagePassing(args.scales, args.mlp_layers, args.MPblock_layers , args.hidden_channels)
        self.processor.append(GraphNet)
        # Decoder MLP
        self.decoder = MLP(args.mlp_layers,args.hidden_channels,args.hidden_channels,args.out_channels)

    def forward(self, x , edge_index, edge_attr, node_positions):
        '''Performs a forward pass across the GNN

        :param x:           torch.Tensor with shape [batch_size-1,n_nodes,node_features]. It is the input features
                            tensor.
        :param edge_index:  torch.Tensor with shape [2,n_edges]. It is the edge connectivity matrix
                            of the mesh, where edge_index[0] returns the source nodes and edge_index[1]
                            returns the destination nodes.
        :param edge_attr:   torch.Tensor with shape [batch_size-1,n_edges,edge_features]. It is the matrix containing
                            the edge fetaures for each edge in the mesh
        :return:            torch.Tensor with shape [batch_size-1,n_nodes,node_features]. It is the output
                            features tensor.
        '''

        if verbose:

            print('################## GNN ##################\n')

            print('x:', type(x), x.size(), '\n')

        x = self.encoder_node(x)

        if verbose:

            print('x aft encode:', type(x), x.size(), x.device, '\n')

            print('edge_attr:', type(edge_attr), edge_attr.size(), '\n')

        edge_attr = self.encoder_edge(edge_attr)

        if verbose:

            print('edge_attr aft encode:', type(edge_attr), edge_attr.size(), edge_attr.device, '\n')

            print('node_positions:', type(node_positions), node_positions.size(), node_positions.device, '\n')

            # print('node_positions[0]:', type(node_positions[0]), len(node_positions[0]), '\n')

            print('edge_index:', type(edge_index), edge_index.size(), edge_index.device, '\n')


        # x, edge_index, edge_attr, pool_indices = self.pool(x, edge_index, edge_attr)
        
        #Process
        # for GraphNet in self.processor:
        #     # x_res, edge_attr_res = GraphNet(x, edge_index, edge_attr)
        #     x, edge_attr = GraphNet(x, edge_index, edge_attr, node_positions)
        #     # x += x_res
        #     # edge_attr += edge_attr_res


        # x = self.unpool(x, pool_indices, n_nodes=node_positions.size(0))

        if stopwatch:
            start = time.time()

        if plot:
            plot_graph(node_positions, edge_index, 'original_graph', 'original_graph.png')

        if verbose:
            print('################## First MultiscaleMessagePassing ##################\n')

        # First MultiscaleMessagePassing
        x, edge_attr = self.processor[0](x, edge_index, edge_attr, node_positions)

        if stopwatch:
            end = time.time()
            print('First MultiscaleMessagePassing: ', end-start)
            start = time.time()

        if verbose:
            print('################## Pooling ##################\n')

        # Pooling
        # x, pooled_edge_index, pooled_edge_index_ordered, edge_attr, pool_indices, pool_indices_ordered = self.pool(x, edge_index, edge_attr)
        x , pooled_edge_index, pooled_edge_index_ordered, edge_attr, pool_indices, pool_edges = self.pool(x, edge_index, edge_attr)

        if verbose:
            print('x aft pool:', type(x), x.size(), x.device, '\n')
            print('pooled_edge_index:', type(pooled_edge_index_ordered), pooled_edge_index_ordered.size(), pooled_edge_index_ordered.device, '\n')
            print('edge_attr aft pool:', type(edge_attr), edge_attr.size(), edge_attr.device, '\n') 

        if stopwatch:
            end = time.time()
            print('Pooling: ', end-start)
            start = time.time()

        if plot:
            # pooled_nodes_ordered= node_positions[pool_indices_ordered[0]]
            # plot_graph_positions(pooled_nodes_ordered, 'pooled_graph_ordered', 'pooled_nodes_ordered.png')
            pooled_nodes = node_positions[pool_indices[0]] 
            plot_graph_positions(pooled_nodes, 'pooled_graph', 'pooled_nodes.png')
            # plot_graph(pooled_nodes_ordered, pooled_edge_index_ordered, 'pooled_graph', 'pooled_graph_ordered.png')
            plot_graph(pooled_nodes, pooled_edge_index_ordered, 'pooled_graph', 'pooled_graph.png')
            # plot_mapping(node_positions, pooled_nodes, pool_indices_ordered[0], 'mapping', 'pool_mapping.png')

        if verbose: 
            print('################## Second MultiscaleMessagePassing ##################\n')
        # Second MultiscaleMessagePassing
        # x, edge_attr = self.processor[1](x, pooled_edge_index_ordered, edge_attr, node_positions, pool_indices_ordered)
        x, edge_attr = self.processor[1](x, pooled_edge_index_ordered, edge_attr, node_positions, pool_indices, pool_edges)

        x, edge_attr = self.processor[2](x, pooled_edge_index_ordered, edge_attr, node_positions, pool_indices, pool_edges)
        
        if stopwatch:
            end = time.time()
            print('Second MultiscaleMessagePassing: ', end-start)
            start = time.time()

        if verbose:
            print('################## Unpooling ##################\n')

        # Unpooling
        # i should implement to return edge_index!!! and edge_attr
        x, edge_attr = self.unpool(x, edge_index, edge_attr, pool_indices, n_nodes=node_positions.size(0))


        if stopwatch:
            end = time.time()
            print('Unpooling: ', end-start)
            start = time.time()

        if verbose:
            print('x aft unpool:', type(x), x.size(), x.device, '\n')
            print('edge_index aft unpool:', type(edge_index), edge_index.size(), edge_index.device, '\n')
            print('edge_attr aft unpool:', type(edge_attr), edge_attr.size(), edge_attr.device, '\n')

        if verbose:
            print('################## Third MultiscaleMessagePassing ##################\n')

        # Third MultiscaleMessagePassing
        x, edge_attr = self.processor[3](x, edge_index, edge_attr,node_positions)
            
        if stopwatch:
            end = time.time()
            print('Third MultiscaleMessagePassing: ', end-start)
            start = time.time()


        #Decode

        x = self.decoder(x)

        return x
    
    def assign_maps_coarsening(self, fine2coarse_list, distances_list, edge_index_coarse_list, fine2coarse_edges_list):
        '''

        Copies long to compute list of coarsened graphs maps and distances into MultiScaleMessagePassing stucture
        fine2coarse_list:   list of tensors with shape [n_nodes_scale], 
        distances_list:     list of tensors with shape [1, n_nodes_scale, 1]
        '''
        for GraphNet in self.processor:
            GraphNet.assign(fine2coarse_list, distances_list, edge_index_coarse_list, fine2coarse_edges_list)
        return










class GNN_noMMP(torch.nn.Module):
    """Class for creating a Graph Neural Network with MP insteasd of Multiscale MP
            Attributes
                encoder_node    (object)    A MLP object that encodes node input features.
                encoder_edge    (object)    A MLP object that encodes edge input features.
                processor       (List)      A list of MMPLayer objects of length mp_steps that propagate
                                            the messages across the mesh nodes.
                decoder         (Object)    A MLP object that decodes the output features.

    """
    def __init__(self, args):
        super(GNN_noMMP, self).__init__()


        # Encoder MLPs
        self.encoder_node = MLP(args.mlp_layers,args.in_node,args.hidden_channels,args.hidden_channels)
        self.encoder_edge = MLP(args.mlp_layers,args.in_edge,args.hidden_channels,args.hidden_channels)
        self.processor = nn.ModuleList()
       
        GraphNet = MP_block(args.MPblock_layers, args.mlp_layers, args.hidden_channels)
        self.processor.append(GraphNet)
        self.pool = Pool(k=args.pool_k, in_dim=args.hidden_channels, p=args.dropout_p)
        GraphNet = MP_block(args.MPblock_layers, args.mlp_layers, args.hidden_channels)
        self.processor.append(GraphNet)
        GraphNet = MP_block(args.MPblock_layers, args.mlp_layers, args.hidden_channels)
        self.processor.append(GraphNet)
        self.unpool = Unpool()
        GraphNet = MP_block(args.MPblock_layers, args.mlp_layers, args.hidden_channels)
        self.processor.append(GraphNet)

        # Decoder MLP
        self.decoder = MLP(args.mlp_layers,args.hidden_channels,args.hidden_channels,args.out_channels)

    def forward(self, x , edge_index, edge_attr, node_positions):
        '''Performs a forward pass across the GNN

        :param x:           torch.Tensor with shape [batch_size-1,n_nodes,node_features]. It is the input features
                            tensor.
        :param edge_index:  torch.Tensor with shape [2,n_edges]. It is the edge connectivity matrix
                            of the mesh, where edge_index[0] returns the source nodes and edge_index[1]
                            returns the destination nodes.
        :param edge_attr:   torch.Tensor with shape [batch_size-1,n_edges,edge_features]. It is the matrix containing
                            the edge fetaures for each edge in the mesh
        :return:            torch.Tensor with shape [batch_size-1,n_nodes,node_features]. It is the output
                            features tensor.
        '''

        if verbose:

            print('################## GNN ##################\n')

            print('x:', type(x), x.size(), '\n')

        x = self.encoder_node(x)

        if verbose:

            print('x aft encode:', type(x), x.size(), x.device, '\n')

            print('edge_attr:', type(edge_attr), edge_attr.size(), '\n')

        edge_attr = self.encoder_edge(edge_attr)

        if verbose:

            print('edge_attr aft encode:', type(edge_attr), edge_attr.size(), edge_attr.device, '\n')

            print('node_positions:', type(node_positions), node_positions.size(), node_positions.device, '\n')

            # print('node_positions[0]:', type(node_positions[0]), len(node_positions[0]), '\n')

            print('edge_index:', type(edge_index), edge_index.size(), edge_index.device, '\n')


        if stopwatch:
            start = time.time()

        if plot:
            plot_graph(node_positions, edge_index, 'original_graph', 'original_graph.png')

        if verbose:
            print('################## First MessagePassing ##################\n')

        # First MessagePassing

        x, edge_attr = self.processor[0](x, edge_index, edge_attr, node_positions)

        if stopwatch:
            end = time.time()
            print('First MultiscaleMessagePassing: ', end-start)
            start = time.time()

        if verbose:
            print('################## Pooling ##################\n')

        # Pooling
        # x, pooled_edge_index, pooled_edge_index_ordered, edge_attr, pool_indices, pool_indices_ordered = self.pool(x, edge_index, edge_attr)
        x , pooled_edge_index, pooled_edge_index_ordered, edge_attr, pool_indices, pool_edges = self.pool(x, edge_index, edge_attr)

        if verbose:
            print('x aft pool:', type(x), x.size(), x.device, '\n')
            print('pooled_edge_index:', type(pooled_edge_index_ordered), pooled_edge_index_ordered.size(), pooled_edge_index_ordered.device, '\n')
            print('edge_attr aft pool:', type(edge_attr), edge_attr.size(), edge_attr.device, '\n')

        if stopwatch:
            end = time.time()
            print('Pooling: ', end-start)
            start = time.time()


        if plot:
            pooled_nodes = node_positions[pool_indices[0]] 
            plot_graph_positions(pooled_nodes, 'pooled_graph', 'pooled_nodes.png')
            plot_graph(pooled_nodes, pooled_edge_index_ordered, 'pooled_graph', 'pooled_graph.png')

        if verbose: 
            print('################## Second MessagePassing ##################\n')
        
        x, edge_attr = self.processor[1](x, pooled_edge_index_ordered, edge_attr, node_positions)

        x, edge_attr = self.processor[2](x, pooled_edge_index_ordered, edge_attr, node_positions)
        
        if stopwatch:
            end = time.time()
            print('Second MultiscaleMessagePassing: ', end-start)
            start = time.time()

        if verbose:
            print('################## Unpooling ##################\n')

        # Unpooling
        # i should implement to return edge_index!!! and edge_attr
        x, edge_attr = self.unpool(x, edge_index, edge_attr, pool_indices, n_nodes=node_positions.size(0))


        if stopwatch:
            end = time.time()
            print('Unpooling: ', end-start)
            start = time.time()
        
        if verbose:
            print('x aft unpool:', type(x), x.size(), x.device, '\n')
            print('edge_index aft unpool:', type(edge_index), edge_index.size(), edge_index.device, '\n')
            print('edge_attr aft unpool:', type(edge_attr), edge_attr.size(), edge_attr.device, '\n')

        if verbose:
            print('################## Third MessagePassing ##################\n')

        # Third MultiscaleMessagePassing
        x, edge_attr = self.processor[3](x, edge_index, edge_attr, node_positions)
            
        if stopwatch:
            end = time.time()
            print('Third MultiscaleMessagePassing: ', end-start)
            start = time.time()


        #Decode

        x = self.decoder(x)

        return x
    
    def assign_maps_coarsening(self, fine2coarse_list, distances_list, edge_index_coarse_list, fine2coarse_edges_list):
        
        return

