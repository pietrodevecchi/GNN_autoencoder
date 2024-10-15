import torch
import torch.nn as nn
from torch_scatter import scatter_mean
from torch_scatter import scatter_add
from torch.nn import Linear, ModuleList
# from coarse_interpolate_utils import *
import tracemalloc
from scipy.spatial import cKDTree
import numpy as np
import imageio
from PIL import Image


from plot_utils.plot_nodes import plot_graph_positions, plot_mapping, plot_graph

import time
import torch

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

verbose = 0
plot = 0
stopwatch = 0

plot_pooling = 0

act = nn.ELU()
residual_connection=1
layer_norm=0

encoding_latent_space=1

class MLP(nn.Module):
    '''Class for creating a Multi-Layer Perceptron
          Attributes
            layers      (List)      A list of layers transforms a tensor x into f(Wx + b), where
                                    f is act activation function, W is the weight matrix and b the bias tensor.


    '''
    def __init__(self, num_layers,in_channels,hidden_channels,out_channels):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_channels,hidden_channels))
        self.layers.append(act)
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_channels,hidden_channels))
            self.layers.append(act)
        self.layers.append(nn.Linear(hidden_channels, out_channels))
        if layer_norm:
            self.layers.append(torch.nn.LayerNorm(out_channels))

               
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
        :param x:           torch.Tensor with shape [n_nodes, hidden_channels].
        :param edge_index:  torch.Tensor with shape [2, n_edges].
        :param edge_attr:   torch.Tensor with shape [n_edges, hidden_channels].
        :return:            torch.Tensor with shape [n_nodes, hidden_channels]
        '''
        _, dest = edge_index

        out_sum = scatter_mean(edge_attr, dest, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out_sum], dim=1)
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
        :param src:         torch.Tensor with shape [n_edges, hidden_channels]. Node features
                            corresponding to source nodes of the edges.
        :param dest:        torch.Tensor with shape [n_edges, hidden_channels]. Node features
                            corresponding to destination nodes of the edges.
        :param edge_attr:   torch.Tensor with shape [n_edges, hidden_channels].
        :return:            torch.Tensor with shape [n_edges, hidden_channels].
        '''
        out = torch.cat([edge_attr, src, dest], dim=1)
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
        '''
        :param x:           torch.Tensor with shape [n_nodes, hidden_channels].
        :param edge_index:  torch.Tensor with shape [2, n_edges].
        :param edge_attr:   torch.Tensor with shape [n_edges, hidden_channels].
        :return:            (torch.Tensor with shape [n_nodes, hidden_channels],
                            torch.Tensor with shape [n_nodes, hidden_channels])
        '''

        src, dest = edge_index

        if residual_connection:
            edge_attr += self.edge_model(x[src], x[dest], edge_attr)
            x += self.node_model(x, edge_index, edge_attr)
        else:
            edge_attr = self.edge_model(x[src], x[dest], edge_attr)
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
        """
        Performs a forward pass across the GNN

        Args:
            x (torch.Tensor): The input features tensor with shape [n_nodes, node_features].
            distances (torch.Tensor): The matrix containing the distance features for each node in the mesh.

        Returns:
            torch.Tensor: The output features tensor with shape [n_nodes, node_features].
        """
        if verbose > 1:
            print('distances_down_model:', type(distances), distances.size(), distances.device, '\n')
            print('x_down_model:', type(x), x.size(), x.device, '\n')        

        # Now you can concatenate 'x' and 'distances_expanded' along dim=1
        # out = torch.cat([x, distances.expand(-1, x.size(1))], dim=1)
        out = torch.cat([x, distances], dim=1)

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
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Node features.            [num_nodes, num_node_features]
            x_scale (torch.Tensor): Node features at a different scale. [num_nodes, num_node_features]
            fine2coarse_index (torch.Tensor): Indices of nodes in the coarser graph. [num_nodes]
            distances (torch.Tensor): Distances of nodes. [num_nodes, 1]

        Returns:
            torch.Tensor: Output node features. [num_nodes, num_node_features]
        """
        x = x[fine2coarse_index, :]
        # out = torch.cat([x, x_scale, distances.expand(-1, -1)], dim = 1)
        out = torch.cat([x, x_scale, distances], dim = 1)
        out = self.node_mlp(out)

        return out
    

class Coarse_layer(nn.Module):
    def __init__(self, down_node_model=None):

        super(Coarse_layer, self).__init__()
        self.down_node_model = down_node_model

    def forward(self, x, coarse_edge_index, edge_attr, scale, fine2coarse_index, fine2coarse_edges,
                    distances, pool_indices=None, pool_edges=None, num_nodes=None, num_edges=None):
            '''Performs a coarsening operation 

            :param x:                   torch.Tensor with shape [n_nodes, hidden_channels].
            :param edge_index:          torch.Tensor with shape [2, n_edges].
            :param edge_attr:           torch.Tensor with shape [n_edges, hidden_channels].
            :param fine2coarse_index:   torch.Tensor with shape [n_edges_coarse].
            :distances:                 torch.Tensor with shape [n_edges_coarse, 1]

            :return:                    torch.Tensor with shape [n_nodes, hidden_channels]
            '''

            if pool_indices is not None:
                # Update fine2coarse_index and distances
                fine2coarse_index = fine2coarse_index[pool_indices]
                distances = distances[pool_indices]

                if verbose > 1:
                    print('pool_edges:', type(pool_edges), pool_edges.size(), pool_edges.device, '\n')

                fine2coarse_edges = fine2coarse_edges[pool_edges]

            coarse_edge_attrs = self.coarsen_edges2(coarse_edge_index, fine2coarse_index, edge_attr, fine2coarse_edges, num_edges)

            if residual_connection:
                fine_node_process = x+self.down_node_model(x, distances)
            else:
                fine_node_process = self.down_node_model(x, distances)

            if num_nodes is not None:
                # coarse_node_attrs = scatter_add(fine_node_process, fine2coarse_index, dim=0 , dim_size=num_nodes)
                # counts = scatter_add(torch.ones_like(fine_node_process), fine2coarse_index, dim=0 , dim_size=num_nodes)
                coarse_node_attrs_avg = scatter_mean(fine_node_process, fine2coarse_index, dim=0, dim_size=num_nodes)

            else:
                # coarse_node_attrs = scatter_add(fine_node_process, fine2coarse_index, dim=0 , dim_size=torch.max(fine2coarse_index).item()+1)
                # counts = scatter_add(torch.ones_like(fine_node_process), fine2coarse_index, dim=0 , dim_size=torch.max(fine2coarse_index).item()+1)
                coarse_node_attrs_avg = scatter_mean(fine_node_process, fine2coarse_index, dim=0, dim_size=torch.max(fine2coarse_index).item()+1)

            # coarse_node_attrs_avg = coarse_node_attrs / counts.clamp(min=1)

            if verbose>1:
                print('fine2coarse_index:', type(fine2coarse_index), fine2coarse_index.size(), fine2coarse_index.device, '\n')
                print('distances:', type(distances), distances.size(), distances.device, '\n')
                print('x_coarse_node', type(coarse_node_attrs_avg), coarse_node_attrs_avg.size(), coarse_node_attrs_avg.device, '\n')

            return coarse_node_attrs_avg, coarse_edge_attrs, coarse_edge_index


    def coarsen_edges2(self, coarse_edge_index, fine2coarse_index, edge_attr, fine2coarse_edges, num_edges):
        '''
        param edges:                        torch.Tensor with shape [2 ,n_edges].
        param fine2coarse_index:            torch.Tensor with shape [n_edges_coarse].
        param edge_attr:                    torch.Tensor with shape [n_edges, hidden_channels].

        '''

        num_edges_fine, _ = edge_attr.shape


        if num_edges is None:
            num_edges = coarse_edge_index.size(1)

        coarse_edge_attrs = torch.zeros(num_edges, edge_attr.size(-1), device=edge_attr.device)

        # coarse_edge_attrs.index_add_(0, fine2coarse_edges, edge_attr)

        # counts = torch.bincount(fine2coarse_edges, minlength=num_edges).to(edge_attr.device)
        # counts_reshaped = counts.view(num_edges, 1)  # Add an extra dimension for broadcasting

        # coarse_edge_attrs_avg1 = coarse_edge_attrs / counts_reshaped.clamp(min=1)

        coarse_edge_attrs_avg = scatter_mean(edge_attr, fine2coarse_edges, dim=0, dim_size=num_edges)


        return coarse_edge_attrs_avg



class Interpolate_layer(torch.nn.Module):

    def __init__(self, up_node_model=None):
        super().__init__()
        self.up_node_model = up_node_model

    def forward(self, x, x_scale, fine2coarse_index, distances, pool_indices=None):
        '''
        :param x:                   torch.Tensor with shape [n_nodes, hidden_channels].
        :param x_scale:             torch.Tensor with shape [n_nodes, hidden_channels].
        :param fine2coarse_index:   torch.Tensor with shape [n_nodes].
        :param distances:           torch.Tensor with shape [n_nodes, 1].
        :return:                    torch.Tensor with shape [n_nodes, hidden_channels]
        '''
        if pool_indices is not None:
            fine2coarse_index = fine2coarse_index[pool_indices]
            distances = distances[pool_indices]

        if x_scale.size(0) != distances.size(0):
            print("beware distances shrinked form {} to {} \n".format(distances.size(0), x_scale.size(0)))
            distances = distances[:x_scale.size(0), :]
            fine2coarse_index = fine2coarse_index[:x_scale.size(0)]

        if residual_connection:
            interp_node_attrs = x_scale+self.up_node_model(x, x_scale, fine2coarse_index, distances)
        else:
            interp_node_attrs = self.up_node_model(x, x_scale, fine2coarse_index, distances)

        return interp_node_attrs





class MultiscaleMessagePassing(nn.Module):
    def __init__(self, scales, mlp_layers, MPblock_layers, hidden_channels):
        super(MultiscaleMessagePassing, self).__init__()
        self.fine2coarse_list = {}
        self.fine2coarse_edges_list = {}
        self.edge_index_coarse_list = {}
        self.distances_list = {}
        self.node_coarse_list = {}
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


    # def forward(self, x, edge_index, edge_attr, node_positions, i=0, pool_indices=None, pool_edges=None):
    def forward(self, data, i=0, pool_indices=None, pool_edges=None):
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
        x, edge_index, edge_attr, node_positions = data.x, data.edge_index, data.edge_attr, data.pos

        x_list = {}
        edge_attr_list = {}
        edge_index_list = {}

        pool_indices_local = pool_indices
        

        # Coarsen graph and perform message passing at each scale
        # for scale, coarse_layer, mp_layer_down1, mp_layer_down2 in zip(self.scales, self.coarse_layers, 
        #                                                self.message_passing_layers_down1[:-1], self.message_passing_layers_down2[:-1]):

        for scale, coarse_layer, MP_block in zip(self.scales, self.coarse_layers, 
                                                    self.message_passing_blocks[:len(self.scales)]):
        
            
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
            
                

            if i!=0:
                if pool_indices_local is not None:
                    map = self.fine2coarse_list[scale][pool_indices_local]
                    device = self.node_coarse_list[scale].device
                    # map_edges = self.fine2coarse_edges_list[scale][pool_edges]
                    # mapping_nodes_edge_index = torch.full((self.fine2coarse_list[scale].size(0),), -1, device=x.device)
                    # mapping_nodes_edge_index[pool_indices_local] = torch.arange(pool_indices_local.size(0), device=x.device)
                    # mapping_nodes_edge_index = mapping_nodes_edge_index.to(device)

                    plot_graph_positions(self.node_coarse_list[scale][map.to(device)], "Coarsen Nodes", f'../plot_{i}.png')
                else:
                    plot_graph_positions(self.node_coarse_list[scale], "Coarsen Nodes", f'../plot_{i}.png')
                i+=1
                # if pool_indices_local is not None:
                #     # plot_graph(self.node_coarse_list[scale][map.to(device)], mapping_nodes_edge_index[(self.edge_index_coarse_list[scale].to(device))[:, map_edges.to(device)]], "Coarsen Graph", f'../plot_{i}.png')
                #     plot_graph(self.node_coarse_list[scale][map.to(device)], edge_index, "Coarsen Graph", f'../plot_{i}.png')
                # else:
                plot_graph(self.node_coarse_list[scale], edge_index, "Coarsen Graph", f'../plot_{i}.png')
                i+=1

            pool_indices_local = None


        
        x, edge_attr = self.message_passing_blocks[len(self.scales)](x, edge_index, edge_attr, node_positions)

              
        x, edge_attr = self.message_passing_blocks[len(self.scales)+1](x, edge_index, edge_attr, node_positions)

        for scale, interpolate_layer, MP_block in zip(reversed(self.scales), self.interpolate_layers, self.message_passing_blocks[len(self.scales)+2:]):
            
            if scale == self.scales[0]:
                pool_indices_local = pool_indices

          
            x = interpolate_layer(x, x_list[scale], self.fine2coarse_list[scale], self.distances_list[scale], pool_indices=pool_indices_local)
            
            x, edge_attr = MP_block(x, edge_index_list[scale], edge_attr_list[scale], node_positions)

        data.x, data.edge_attr = x, edge_attr
        
        # return x, edge_attr, i
        return data, i

    def assign(self, fine2coarse_list, distances_list, edge_index_coarse_list, fine2coarse_edges_list, node_coarse_list):
        self.fine2coarse_list = fine2coarse_list
        self.distances_list = distances_list
        self.edge_index_coarse_list = edge_index_coarse_list
        self.fine2coarse_edges_list = fine2coarse_edges_list
        self.node_coarse_list = node_coarse_list

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

    # def forward(self, x, edge_index, edge_attr, node_positions):
    def forward(self, data):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Node features.            [num_nodes, num_node_features]
            edge_index (torch.Tensor): Edge indices.    [2, num_edges]
            edge_attr (torch.Tensor): Edge attributes.  [num_edges, num_edge_features]

        Returns:
            torch.Tensor: New node features.            [num_nodes', num_node_features]
            torch.Tensor: New edge indices.             [2, num_edges']
            torch.Tensor: New edge attributes.          [num_edges', num_edge_features]
            torch.Tensor: Pool indices.                 [num_nodes']
        """

        x, edge_index, edge_attr, node_positions = data.x, data.edge_index, data.edge_attr, data.pos

        # Apply dropout and project node features to get scores, then apply sigmoid
        Z = self.drop(x)
        weights = self.proj(Z).squeeze(-1)
        scores = self.sigmoid(weights)

        # Compute the threshold for top-k selection based on scores
        kth_scores, _ = torch.kthvalue(scores, int((1 - self.k) * scores.size(0)), keepdim=True)
        mask = scores >= kth_scores

        # Use the mask to determine the indices of selected nodes
        _, pool_indices = mask.float().sort(descending=True)
        pool_indices = pool_indices[:int(x.size(0) * self.k)]  # Keep top k% indices

        # Select nodes based on the pool_indices
        new_x = x[pool_indices]

        pooled_node_positions = node_positions[pool_indices]

        # Create a mask for nodes that are selected for pooling
        mask = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
        mask[pool_indices] = True
       
        # Create a mask for edges that connect the nodes left in new_x
        pool_edges_mask = mask[edge_index[0]] & mask[edge_index[1]]
        pool_edges = torch.nonzero(pool_edges_mask, as_tuple=True)[0]

        # Select edges based on the pool_edges
        new_edge_index = edge_index[:, pool_edges]

        # Select edge attributes based on the pool_edges
        new_edge_attr = edge_attr[pool_edges]

        # Create a mapping from old indices to new indices
        mapping = torch.full((x.size(0),), -1, device=x.device)
        mapping[pool_indices] = torch.arange(pool_indices.size(0), device=x.device)

        # Remap the node indices in new_edge_index
        new_edge_index_ordered = mapping[new_edge_index]

        data.x, data.edge_index, data.edge_attr, data.pos = new_x, new_edge_index_ordered, new_edge_attr, pooled_node_positions

        # return new_x, new_edge_index, new_edge_index_ordered, new_edge_attr, pooled_node_positions, pool_indices, pool_edges
        return data, pool_indices, pool_edges


class Unpool(nn.Module):
    def __init__(self):
        super(Unpool, self).__init__()


    # def forward(self, x, unpooled_edge_index, edge_attr, node_positions, pool_indices, pool_edges, n_nodes):
    def forward(self, data, i, pool_indices, pool_edges, unpooled_edge_index, n_nodes):
        """
        Unpools the features and edge attributes by reconstructing the original shape with unpooled features and edge attributes correctly placed.

        Args:
            x (torch.Tensor): The pooled features with shape [n_nodes', hidden_channels].
            unpooled_edge_index (torch.Tensor): The edge indices of the bigger (unpooled) graph.
            edge_attr (torch.Tensor): The edge attributes of the pooled graph.
            pool_indices (torch.Tensor): The indices of the original nodes before pooling, with shape [n_nodes'].

        Returns:
            torch.Tensor: The unpooled features with shape [n_nodes, hidden_channels].
            torch.Tensor: The new edge attributes with shape [2, n_edges, edge_attr_dim], where n_edges is the number of edges in the unpooled graph, and edge_attr_dim is the dimension of edge attributes.

        """
        x, edge_attr = data.x, data.edge_attr

        # Initialize tensor to hold unpooled features, filling with zeros
        new_x = torch.zeros(n_nodes, x.size(1), device=x.device, dtype=x.dtype)

        # Initialize tensor to hold new edge attributes, filling with zeros
        new_edge_attr = torch.zeros(unpooled_edge_index.size(1), edge_attr.size(1), device=edge_attr.device, dtype=edge_attr.dtype)

        # Reconstruct the original shape with unpooled features correctly placed
        new_x[pool_indices] = x

        # # Get the indices of edges that correspond to the pooled graph
        # mask_source = torch.isin(unpooled_edge_index[0], pool_indices)
        # mask_target = torch.isin(unpooled_edge_index[1], pool_indices)
        # pooled_edge_indices = mask_source & mask_target

        # # Assign the attributes from edge_attr to the corresponding edges in new_edge_attr
        # new_edge_attr[pooled_edge_indices] = edge_attr
        new_edge_attr[pool_edges] = edge_attr

        data.x, data.edge_attr, data.edge_index = new_x, new_edge_attr, unpooled_edge_index

        # return new_x, new_edge_attr
        return data, i

    
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

        self.plot = args.plot_nodes


        # Encoder MLPs
        self.encoder_node = MLP(args.mlp_layers,args.in_node,args.hidden_channels,args.hidden_channels)
        self.encoder_node_ROM = MLP(args.mlp_layers,args.in_node_ROM,args.hidden_channels,args.hidden_channels)
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
        self.final_decoder = MLP(args.mlp_layers,args.hidden_channels,args.hidden_channels,args.out_channels)

        if encoding_latent_space:
            # adding a decoder and a decoder layers in the latent space in order to massively reduce dimensionality
            self.latent_decoder = MLP(args.mlp_layers,args.hidden_channels,args.hidden_channels,args.out_channels)
            self.latent_encoder = MLP(args.mlp_layers,args.out_channels,args.hidden_channels,args.hidden_channels)


    def forward(self, data, dt=None):
        """
        Performs a forward pass across the GNN

        Args:
            x (torch.Tensor): The input features tensor with shape [n_nodes, node_features].
            edge_index (torch.Tensor): The edge connectivity matrix of the mesh, where edge_index[0] returns the source nodes and edge_index[1] returns the destination nodes.
            edge_attr (torch.Tensor): The matrix containing the edge features for each edge in the mesh.
            node_positions (torch.Tensor): The positions of the nodes in the graph.

        Returns:
            torch.Tensor: The output features tensor with shape [n_nodes, node_features].
        """

        data, pool_indices, pool_edges, edge_index, n_nodes, i = self.encoder(data, dt)

        data.x = self.decoder(data, pool_indices, pool_edges, edge_index, n_nodes, i)

        return data.x
    
    def assign_maps_coarsening(self, fine2coarse_list, distances_list, edge_index_coarse_list, fine2coarse_edges_list, node_coarse_list):
        '''

        Copies long to compute list of coarsened graphs maps and distances into MultiScaleMessagePassing stucture
        fine2coarse_list:   list of tensors with shape [n_nodes_scale], 
        distances_list:     list of tensors with shape [1, n_nodes_scale, 1]
        '''
        for GraphNet in self.processor:
            GraphNet.assign(fine2coarse_list, distances_list, edge_index_coarse_list, fine2coarse_edges_list, node_coarse_list)
        return


    def encoder(self, data, dt=None):

        data.x = self.encoder_node(data.x)
        data.edge_attr = self.encoder_edge(data.edge_attr)
        i=0

        if plot_pooling and dt==0:
            plot_graph(data.pos, data.edge_index, 'Graph', f'paper/plot_original.png')

        if self.plot: 
            plot_graph_positions(data.pos, 'Initial Graph', f'../plot_{i}.png')
            i+=1
            plot_graph(data.pos, data.edge_index, 'Graph', f'../plot_{i}.png')
            i+=1

        # First MultiscaleMessagePassing
        # x, edge_attr, i = self.processor[0](x, edge_index, edge_attr, node_positions, i)
        data, i = self.processor[0](data, i)

        # for unpooling
        edge_index = data.edge_index
        n_nodes=data.pos.size(0)

        # Pooling
        # x, pooled_edge_index, pooled_edge_index_ordered, edge_attr, pooled_node_positions, pool_indices, pool_edges = self.pool(x, edge_index, edge_attr, node_positions)
        data, pool_indices, pool_edges = self.pool(data)

        if plot_pooling and (dt==0 or dt==35 or dt==70):
            plot_graph(data.pos, data.edge_index, 'Graph', f'paper/plot_graph_pooled_{dt}.png')

        if self.plot:
            # plot_graph_positions(pooled_node_positions, 'Pooled Graph', f'../plot_{i}.png')
            plot_graph_positions(data.pos, 'Pooled Graph', f'../plot_{i}.png')
            i+=1
            # plot_graph(pooled_node_positions, pooled_edge_index_ordered, 'Pooled Graph', f'../plot_{i}.png')
            plot_graph(data.pos, data.edge_index, 'Pooled Graph', f'../plot_{i}.png')
            i+=1

        # if dt is not None:
        #     plot_graph(data.pos, data.edge_index, 'Pooled Graph', f'../plot_pooled_{dt}.png')

        # Second MultiscaleMessagePassing
        # x, edge_attr, i = self.processor[1](x, pooled_edge_index_ordered, edge_attr, pooled_node_positions, i, pool_indices, pool_edges)
        data, i = self.processor[1](data, i, pool_indices, pool_edges)

        if encoding_latent_space:
            data.x = self.latent_decoder(data.x)

        return data, pool_indices, pool_edges, edge_index, n_nodes, i
    
    def encoder_ROM(self, data, dt=None):

        data.x = self.encoder_node_ROM(data.x)
        data.edge_attr = self.encoder_edge(data.edge_attr)
        i=0

        if plot_pooling and dt==0:
            plot_graph(data.pos, data.edge_index, 'Graph', f'paper/plot_original.png')
        if self.plot: 
            plot_graph_positions(data.pos, 'Initial Graph', f'../plot_{i}.png')
            i+=1
            plot_graph(data.pos, data.edge_index, 'Graph', f'../plot_{i}.png')
            i+=1

        # First MultiscaleMessagePassing
        # x, edge_attr, i = self.processor[0](x, edge_index, edge_attr, node_positions, i)
        data, i = self.processor[0](data, i)

        # for unpooling
        edge_index = data.edge_index
        n_nodes=data.pos.size(0)

        # Pooling
        # x, pooled_edge_index, pooled_edge_index_ordered, edge_attr, pooled_node_positions, pool_indices, pool_edges = self.pool(x, edge_index, edge_attr, node_positions)
        data, pool_indices, pool_edges = self.pool(data)

        if plot_pooling:
            if dt==0 or dt==50 or dt==100:
                plot_graph(data.pos, data.edge_index, 'Graph', f'paper/plot_graph_pooled_{dt}.png')

        if self.plot:
            # plot_graph_positions(pooled_node_positions, 'Pooled Graph', f'../plot_{i}.png')
            plot_graph_positions(data.pos, 'Pooled Graph', f'../plot_{i}.png')
            i+=1
            # plot_graph(pooled_node_positions, pooled_edge_index_ordered, 'Pooled Graph', f'../plot_{i}.png')
            plot_graph(data.pos, data.edge_index, 'Pooled Graph', f'../plot_{i}.png')
            i+=1

        # if dt is not None:
        #     plot_graph(data.pos, data.edge_index, 'Pooled Graph', f'../plot_pooled_{dt}.png')

        # Second MultiscaleMessagePassing
        # x, edge_attr, i = self.processor[1](x, pooled_edge_index_ordered, edge_attr, pooled_node_positions, i, pool_indices, pool_edges)
        data, i = self.processor[1](data, i, pool_indices, pool_edges)

        if encoding_latent_space:
            data.x = self.latent_decoder(data.x)

        return data, pool_indices, pool_edges, edge_index, n_nodes, i
    
    
    def decoder(self, data, pool_indices, pool_edges, edge_index, n_nodes, i):
        if encoding_latent_space:
            data.x = self.latent_encoder(data.x)
        

        # x, edge_attr, i = self.processor[2](x, pooled_edge_index_ordered, edge_attr, pooled_node_positions, i, pool_indices, pool_edges)
        data, i = self.processor[2](data, i, pool_indices, pool_edges)

        # Unpooling
        # x, edge_attr = self.unpool(x, edge_index, edge_attr, pooled_node_positions, pool_indices, pool_edges, n_nodes=node_positions.size(0))
        data, i = self.unpool(data, i, pool_indices, pool_edges, edge_index, n_nodes)


        # if self.plot:
        #     # plot_graph_positions(node_positions, 'Unpooled Graph', f'../plot_{i}.png')
        #     plot_graph_positions(data.pos, 'Unpooled Graph', f'../plot_{i}.png')
        #     i+=1
        #     # plot_graph(node_positions, edge_index, 'Unpooled Graph', f'../plot_{i}.png')
        #     plot_graph(data.pos, data.edge_index, 'Unpooled Graph', f'../plot_{i}.png')
        #     i+=1

        # Third MultiscaleMessagePassing
        # x, edge_attr, i = self.processor[3](x, edge_index, edge_attr, node_positions, i)
        data, i = self.processor[3](data, i)


        if self.plot:
            images_pil = []
            for j in range(i):
                img = Image.open(f'plot_{j}.png')
                images_pil.append(img)
                # os.remove(f'plot_{j}.png')

            # Save as GIF with each frame lasting 40 seconds (40000 milliseconds)
            images_pil[0].save('plot_nodes/graph_transformation.gif', save_all=True, append_images=images_pil[1:], duration=1000, loop=0)    

        
        # Decode
        data.x = self.final_decoder(data.x)

        return data.x








# class GNN_noMMP(torch.nn.Module):
#     """Class for creating a Graph Neural Network with MP insteasd of Multiscale MP
#             Attributes
#                 encoder_node    (object)    A MLP object that encodes node input features.
#                 encoder_edge    (object)    A MLP object that encodes edge input features.
#                 processor       (List)      A list of MMPLayer objects of length mp_steps that propagate
#                                             the messages across the mesh nodes.
#                 decoder         (Object)    A MLP object that decodes the output features.

#     """
#     def __init__(self, args):
#         super(GNN_noMMP, self).__init__()


#         # Encoder MLPs
#         self.encoder_node = MLP(args.mlp_layers,args.in_node,args.hidden_channels,args.hidden_channels)
#         self.encoder_edge = MLP(args.mlp_layers,args.in_edge,args.hidden_channels,args.hidden_channels)
#         self.processor = nn.ModuleList()
       
#         GraphNet = MP_block(args.MPblock_layers, args.mlp_layers, args.hidden_channels)
#         self.processor.append(GraphNet)
#         self.pool = Pool(k=args.pool_k, in_dim=args.hidden_channels, p=args.dropout_p)
#         GraphNet = MP_block(args.MPblock_layers, args.mlp_layers, args.hidden_channels)
#         self.processor.append(GraphNet)
#         GraphNet = MP_block(args.MPblock_layers, args.mlp_layers, args.hidden_channels)
#         self.processor.append(GraphNet)
#         self.unpool = Unpool()
#         GraphNet = MP_block(args.MPblock_layers, args.mlp_layers, args.hidden_channels)
#         self.processor.append(GraphNet)

#         # Decoder MLP
#         self.decoder = MLP(args.mlp_layers,args.hidden_channels,args.hidden_channels,args.out_channels)

#     def forward(self, x, edge_index, edge_attr, node_positions):
#         """
#         Performs a forward pass across the GNN

#         Args:
#             x (torch.Tensor): The input features tensor with shape [n_nodes, node_features].
#             edge_index (torch.Tensor): The edge connectivity matrix of the mesh, where edge_index[0] returns the source nodes and edge_index[1] returns the destination nodes.
#             edge_attr (torch.Tensor): The matrix containing the edge features for each edge in the mesh.
#             node_positions (torch.Tensor): The positions of the nodes in the graph.

#         Returns:
#             torch.Tensor: The output features tensor with shape [n_nodes, node_features].
#         """
#         x = self.encoder_node(x)
#         edge_attr = self.encoder_edge(edge_attr)

#         # First MultiscaleMessagePassing
#         x, edge_attr = self.processor[0](x, edge_index, edge_attr, node_positions)

#         # Pooling
#         x, pooled_edge_index, pooled_edge_index_ordered, edge_attr, pool_indices, pool_edges = self.pool(x, edge_index, edge_attr)

#         # Second MultiscaleMessagePassing
#         x, edge_attr = self.processor[1](x, pooled_edge_index_ordered, edge_attr, node_positions)
#         x, edge_attr = self.processor[2](x, pooled_edge_index_ordered, edge_attr, node_positions)

#         # Unpooling
#         x, edge_attr = self.unpool(x, edge_index, edge_attr, pool_indices, n_nodes=x.size(0))

#         # Third MultiscaleMessagePassing
#         x, edge_attr = self.processor[3](x, edge_index, edge_attr, node_positions)

#         # Decode
#         x = self.decoder(x)

#         return x
    
#     def assign_maps_coarsening(self, fine2coarse_list, distances_list, edge_index_coarse_list, fine2coarse_edges_list):
        
#         return
