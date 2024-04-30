import torch
import torch.nn as nn
from torch_geometric.nn.aggr import Aggregation
from torch_scatter import scatter_mean
from torch_scatter import scatter_add
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.utils import scatter
from torch_geometric.nn import MetaLayer
from geom_AE import Graph_AE, Graph_AE_noMMP
from torch_geometric.nn.pool.topk_pool import TopKPooling

from torch_geometric.nn.models import GraphUNet


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
stopwatch = 0




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

        self.plot = args.plot_nodes
        # self.plot =  0


        # Encoder MLPs
        self.encoder_node = MLP(args.mlp_layers,args.in_node,args.hidden_channels,args.hidden_channels)
        self.encoder_edge = MLP(args.mlp_layers,args.in_edge,args.hidden_channels,args.hidden_channels)
        
        self.processor = Graph_AE_noMMP(in_channels= args.hidden_channels, hidden_channels= args.hidden_channels, out_channels= args.hidden_channels, 
                                        mlp_layers= args.mlp_layers,
                                        MP_block_layers= args.MPblock_layers, 
                                        depth= args.pooling_depth,
                                        pool_ratios = args.pool_k , sum_res= True, 
                                        act = 'relu', plot=self.plot, edge_augment_pooling= args.edge_augment_pooling)
        
        # self.processor = GraphUNet(in_channels= args.hidden_channels, hidden_channels= args.hidden_channels, out_channels= args.hidden_channels, 
        #                             depth= 2, pool_ratios = args.pool_k , sum_res= True, 
        #                             act = 'relu')


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

        if self.plot>1:
            plot_graph(node_positions, edge_index, 'original_graph', 'original_graph.png')

        if verbose:
            print('################## Graph_AE##################\n')


        x = self.processor(x[0], edge_index, edge_attr[0], node_pos=node_positions).unsqueeze(0)

        # x = self.processor(x[0], edge_index).unsqueeze(0)

        #Decode

        x = self.decoder(x)

        return x
    
    def assign_maps_coarsening(self, fine2coarse_list, distances_list, edge_index_coarse_list, fine2coarse_edges_list):
        
        return
    


class GNN(torch.nn.Module):
    """Class for creating a Graph Neural Network with MP insteasd of Multiscale MP
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
        self.encoder_edge = MLP(args.mlp_layers,args.in_edge,args.hidden_channels,args.hidden_channels)
        
        self.processor = Graph_AE(in_channels= args.hidden_channels, hidden_channels= args.hidden_channels, out_channels= args.hidden_channels, 
                                    mlp_layers= args.mlp_layers,
                                    MP_block_layers= args.MPblock_layers,
                                    depth= args.pooling_depth, multiscales= args.scales,
                                    pool_ratios = args.pool_k , sum_res= True, 
                                    act = 'relu', plot=self.plot, edge_augment_pooling= args.edge_augment_pooling)
        
        # self.processor = GraphUNet(in_channels= args.hidden_channels, hidden_channels= args.hidden_channels, out_channels= args.hidden_channels, 
        #                             depth= 2, pool_ratios = args.pool_k , sum_res= True, 
        #                             act = 'relu')


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

        if self.plot>1:
            plot_graph(node_positions, edge_index, 'original_graph', 'original_graph.png')

        if verbose:
            print('################## Graph_AE##################\n')


        x = self.processor(x[0], edge_index, edge_attr[0], node_pos=node_positions).unsqueeze(0)

        # x = self.processor(x[0], edge_index).unsqueeze(0)

        #Decode

        x = self.decoder(x)

        return x
    
    def assign_maps_coarsening(self, fine2coarse_list, distances_list, edge_index_coarse_list, fine2coarse_edges_list):
        
        return






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
        self.layers.append(nn.ReLU())
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_channels,hidden_channels))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_channels, out_channels))

               
    def forward(self, x):
        '''
        :param x: torch.Tensor
        :return: torch.Tensor
        '''
        for layer in self.layers:
            x = layer(x)
        return x
    




