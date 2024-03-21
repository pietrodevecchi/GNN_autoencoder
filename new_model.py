import torch
import torch.nn as nn
from torch_scatter import scatter_add
from torch.nn import Linear, ModuleList
# from coarse_interpolate_utils import *
import tracemalloc
from scipy.spatial import cKDTree
import numpy as np




verbose = 0

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
    


class EdgeModel(torch.nn.Module):
    '''Class for creating a model for the edges
        Attributes
            edge_mlp      (object)    A MLP object that tranforms edge features

    '''
    def __init__(self, mlp_layers, hidden_channels):
        super(EdgeModel, self).__init__()
        self.edge_mlp = MLP(mlp_layers,3*hidden_channels,hidden_channels,hidden_channels)

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
        out = scatter_add(edge_attr, dest, dim=1, dim_size=x.size(1))

        out = torch.cat([x, out], dim=2)
        out = self.node_mlp(out)
        return out



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

        if verbose:
            print('distances_down_model:', type(distances), distances.size(), '\n')
            print('x_down_model:', type(x), x.size(), '\n')

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

        if(verbose):
            print('up fine2coarse_index: ', type(fine2coarse_index), fine2coarse_index.size(), '\n')
            print('up x: ', type(x), x.size(), '\n')
            print('up x_scale: ', type(x_scale), x_scale.size(), '\n')
            print('up distances: ', type(distances), distances.size(), '\n')

        out = torch.cat([x, x_scale, distances.expand(x.size(0), -1, -1)], dim = 2)
        
        out = self.node_mlp(out)

        return out




class Interpolate_layer(torch.nn.Module):

    def __init__(self, up_node_model=None):
        super().__init__()
        self.up_node_model = up_node_model

    def forward(self, x, x_scale, fine2coarse_index, distances):

        '''Performs an interpolation 

        :param x:                   torch.Tensor with shape [batch_size-1,n_nodes,hidden_channels].
        :param x_scale:             torch.Tensor with shape [batch_size-1,n_nodes,hidden_channels].
        :param fine2coarse_index:   torch.Tensor with shape [n_nodes].
        :param distances:           torch.Tensor with shape [1,n_nodes,1].
        :return:                    torch.Tensor with shape [batch_size-1,n_nodes,hidden_channels]
        '''

        interp_node_attrs = self.up_node_model(x, x_scale, fine2coarse_index, distances)

        return interp_node_attrs




# class Coarse_layer(torch.nn.Module):
#     # '''Class for creating a single message passing layer
#     # Attributes
#     #         edge_model      (object)    A edge_model object that transforms the current edge_features
#     #         node_model      (object)    A node_model object that combines node_features and edge_features and
#     #                                     transforms them.
#     # '''

#     def __init__(self, down_node_model=None):
#         super().__init__()
#         self.down_node_model = down_node_model

#     def forward(self, x, node_positions, edge_index, edge_attr, scale):
#         # '''Performs a message passing forward pass

#         # :param x:           torch.Tensor with shape [batch_size-1,n_nodes,hidden_channels].
#         # :param edge_index:  torch.Tensor with shape [2,n_edges].
#         # :param edge_attr:   torch.Tensor with shape [batch_size-1,n_edges,hidden_channels]
#         # :return:            (torch.Tensor with shape [batch_size-1,n_nodes,hidden_channels],
#         #                     torch.Tensor with shape [batch_size-1,n_nodes,hidden_channels])
#         # '''



#         coarse_nodes_pos, coarse_edges, fine2coarse_index, coarse_edge_attrs, distances = coarsen_graph(node_positions, edge_index, scale, edge_attr)

#         fine_node_process = self.down_node_model(x, distances)

#         if verbose:
#             print('coarse_nodes_pos: ', type(coarse_nodes_pos), coarse_nodes_pos.size(), '\n')
#             print('coarse_edges: ', type(coarse_edges), coarse_edges.size(), '\n')
#             print('fine2coarse_index: ', type(fine2coarse_index), fine2coarse_index.size(), '\n')
#             print('coarse_edge_attrs: ', type(coarse_edge_attrs),coarse_edge_attrs.size() , '\n')
#             print('distances: ', type(distances), distances.size(), '\n')

#             print('fine_node_process: ', type(fine_node_process), fine_node_process.size(), '\n')

        

#         dest = fine2coarse_index
        

#         coarse_node_attrs = scatter_add(fine_node_process, dest, dim=1, dim_size=coarse_nodes_pos.size(0))

#         ones = torch.ones_like(fine_node_process)
#         counts = scatter_add(ones, dest, dim=1, dim_size=coarse_nodes_pos.size(0))

#         # Divide 'coarse_node_attrs' by 'counts' to average the attributes
#         coarse_node_attrs_avg = coarse_node_attrs / counts.clamp(min=1)  

#         if verbose:
#             print('coarse_node_attrs_avg: ', type(coarse_node_attrs_avg), coarse_node_attrs_avg.size(), '\n')

#         return coarse_node_attrs_avg, coarse_edge_attrs, coarse_nodes_pos, coarse_edges, fine2coarse_index, distances


# scales = [0.1, 0.5, 1.0]



class Coarse_layer(nn.Module):
    def __init__(self, down_node_model=None):

        super(Coarse_layer, self).__init__()
        self.down_node_model = down_node_model

    def forward(self, x, edge_index, edge_attr, scale, closest_centroid_indices,
                distances):
        '''Performs a coarsening operation 

        :param x:                   torch.Tensor with shape [batch_size-1,n_nodes,hidden_channels].
        :param node_positions:      torch.Tensor with shape [n_nodes,2].
        :param edge_index:          torch.Tensor with shape [2, n_edges].
        :param edge_attr:           torch.Tensor with shape [batch_size-1,n_edges,hidden_channels].
        :return:                    torch.Tensor with shape [batch_size-1,n_nodes,hidden_channels]
        '''

        # voxel_centroids, closest_centroid_indices, distances = self.create_voxel_grid_with_distances(node_positions, scale)
        coarse_edges, fine2coarse_index, coarse_edge_attrs = self.coarsen_graph(edge_index, 
                                                                                                  closest_centroid_indices, 
                                                                                                  edge_attr)

        fine_node_process = self.down_node_model(x, distances)
        # coarse_node_attrs = scatter_add(fine_node_process, fine2coarse_index, dim=1, dim_size=coarse_nodes_pos.size(0))
        # counts = scatter_add(torch.ones_like(fine_node_process), fine2coarse_index, dim=1, dim_size=coarse_nodes_pos.size(0))
        # print('controlla dimensione giusta \n')
        coarse_node_attrs = scatter_add(fine_node_process, fine2coarse_index, dim=1) #, dim_size=distances.size(1))
        counts = scatter_add(torch.ones_like(fine_node_process), fine2coarse_index, dim=1) #, dim_size=distances.size(1))
        coarse_node_attrs_avg = coarse_node_attrs / counts.clamp(min=1)

        # return coarse_node_attrs_avg, coarse_edge_attrs, coarse_nodes_pos, coarse_edges, fine2coarse_index, distances
        return coarse_node_attrs_avg, coarse_edge_attrs, coarse_edges, fine2coarse_index, distances


    # def create_voxel_grid_with_distances(self, nodes, length_scale):
    #     min_x, min_y = torch.min(nodes, dim=0).values
    #     max_x, max_y = torch.max(nodes, dim=0).values
    #     grid_x = torch.arange(min_x + length_scale / 2, max_x, length_scale)
    #     grid_y = torch.arange(min_y + length_scale / 2, max_y, length_scale)
    #     voxel_centroids = torch.stack([grid_x.repeat_interleave(len(grid_y)), grid_y.repeat(len(grid_x))], dim=1)

    #     centroid_tree = cKDTree(voxel_centroids.numpy())
    #     distances, closest_centroid_indices = centroid_tree.query(nodes.numpy())

    #     unique_assigned_centroids = torch.tensor(np.unique(closest_centroid_indices), dtype=torch.long)
    #     final_voxel_centroids = voxel_centroids[unique_assigned_centroids]
        
    #     fin_centroid_tree = cKDTree(final_voxel_centroids.numpy())
    #     fin_distances, fin_closest_centroid_indices = fin_centroid_tree.query(nodes.numpy())
        
    #     fin_closest_centroid_indices = torch.tensor(fin_closest_centroid_indices, dtype=torch.long)
    #     fin_distances = torch.tensor(fin_distances, dtype=torch.float).unsqueeze(0).unsqueeze(-1)

    #     return final_voxel_centroids, fin_closest_centroid_indices, fin_distances


    # def coarsen_graph(self, nodes, edges, voxel_centroids, closest_centroid_indices, edge_attr):
    def coarsen_graph(self, edges, closest_centroid_indices, edge_attr):
        batch_size = edge_attr.size(0)
        start_voxels_list = []
        end_voxels_list = []
        coarse_edge_attrs_list = []

        edge_index_map = {}
        for i in range(edges.size(1)):
            start_node, end_node = edges[:, i]
            start_voxel, end_voxel = closest_centroid_indices[start_node].item(), closest_centroid_indices[end_node].item()

            if start_voxel != end_voxel:
                key = tuple(sorted([start_voxel, end_voxel]))
                edge_index_map.setdefault(key, []).append(i)

        for batch_idx in range(batch_size):
            batch_edge_attrs_list = []
            for (start_voxel, end_voxel), indices in edge_index_map.items():
                indices_tensor = torch.tensor(indices, dtype=torch.long)
                selected_attrs = edge_attr[batch_idx, indices_tensor].mean(dim=0)
                batch_edge_attrs_list.append(selected_attrs.unsqueeze(0))
                if batch_idx == 0:
                    start_voxels_list.append(start_voxel)
                    end_voxels_list.append(end_voxel)

            if batch_edge_attrs_list:
                batch_coarse_edge_attrs = torch.cat(batch_edge_attrs_list, dim=0)
                coarse_edge_attrs_list.append(batch_coarse_edge_attrs.unsqueeze(0))

        start_voxels = torch.tensor(start_voxels_list, dtype=torch.long)
        end_voxels = torch.tensor(end_voxels_list, dtype=torch.long)
        coarse_edges = torch.stack([start_voxels, end_voxels], dim=0)

        if coarse_edge_attrs_list:
            coarse_edge_attrs = torch.cat(coarse_edge_attrs_list, dim=0)
        else:
            coarse_edge_attrs = torch.empty((0, edge_attr.size(-1)), dtype=edge_attr.dtype)

        # coarse_nodes = voxel_centroids
        # return coarse_nodes, coarse_edges, closest_centroid_indices, coarse_edge_attrs
        return coarse_edges, closest_centroid_indices, coarse_edge_attrs



class MultiscaleMessagePassing(nn.Module):
    def __init__(self, scales, mlp_layers, hidden_channels):
        super(MultiscaleMessagePassing, self).__init__()
        self.fine2coarse_list = {}
        self.distances_list = {}
        self.scales = scales
        self.message_passing_layers_down = nn.ModuleList()
        self.message_passing_layers_up = nn.ModuleList()
        self.coarse_layers = nn.ModuleList()
        self.interpolate_layers = nn.ModuleList()

        node_model = NodeModel(mlp_layers, hidden_channels)
        edge_model = EdgeModel(mlp_layers, hidden_channels)
        self.message_passing_layers_down.append(MPLayer(node_model=node_model, edge_model=edge_model))

        for scale in scales:

            down_model = Down_NodeModel(mlp_layers, hidden_channels)
            self.coarse_layers.append(Coarse_layer(down_node_model=down_model))

            node_model = NodeModel(mlp_layers, hidden_channels)
            edge_model = EdgeModel(mlp_layers, hidden_channels)
            self.message_passing_layers_down.append(MPLayer(node_model=node_model, edge_model=edge_model))

        for scale in scales:

            up_model = Up_NodeModel(mlp_layers, hidden_channels)
            self.interpolate_layers.append(Interpolate_layer(up_node_model=up_model))

            node_model = NodeModel(mlp_layers, hidden_channels)
            edge_model = EdgeModel(mlp_layers, hidden_channels)
            self.message_passing_layers_up.append(MPLayer(node_model=node_model, edge_model=edge_model))


    def forward(self, x, edge_index, edge_attr, node_positions):
        x_list = {}
        edge_attr_list = {}
        edge_index_list = {}

        # x, edge_attr = self.message_passing_layers_down[0](x, edge_index, edge_attr)

        # scale = scale[0]
        # x_list[scale] = x
        # edge_attr_list[scale]= edge_attr
        

        # Coarsen graph and perform message passing at each scale
        for scale, coarse_layer, mp_layer_down in zip(self.scales, self.coarse_layers, self.message_passing_layers_down[:-1]):
            
            x, edge_attr = mp_layer_down(x, edge_index, edge_attr)
            x_list[scale]=x
            edge_attr_list[scale]= edge_attr
            edge_index_list[scale] = edge_index

            x, edge_attr, edge_index, fine2coarse_index, dist = \
                coarse_layer(x, edge_index, edge_attr, scale,
                             self.fine2coarse_list[scale],self.distances_list[scale])


            # fine2coarse_list[scale]= fine2coarse_index
            # distances[scale]= dist

        x, edge_attr = self.message_passing_layers_down[-1](x, edge_index, edge_attr)

        for scale, interpolate_layer, mp_layer_up in zip(reversed(self.scales), self.interpolate_layers, self.message_passing_layers_up):

            x = interpolate_layer(x, x_list[scale], self.fine2coarse_list[scale], self.distances_list[scale])
            x, edge_attr = mp_layer_up(x, edge_index_list[scale], edge_attr_list[scale])
        
        return x, edge_attr

    def assign(self, fine2coarse_list, distances_list):
        self.fine2coarse_list = fine2coarse_list
        self.distances_list = distances_list

        return
    






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
    


class GNN(torch.nn.Module):
    """Class for creating a Graph Neural Network
            Attributes
                encoder_node    (object)    A MLP object that encodes node input features.
                encoder_edge    (object)    A MLP object that encodes edge input features.
                processor       (List)      A list of MPLayer objects of length mp_steps that propagate
                                            the messages across the mesh nodes.
                decoder         (Object)    A MLP object that decodes the output features.

    """
    def __init__(self, args):
        super(GNN, self).__init__()


        # Encoder MLPs
        self.encoder_node = MLP(args.mlp_layers,args.in_node,args.hidden_channels,args.hidden_channels)
        self.encoder_edge = MLP(args.mlp_layers,args.in_edge,args.hidden_channels,args.hidden_channels)
        # Processor MLPs
        self.processor = nn.ModuleList()
        for _ in range(args.mp_steps):
            GraphNet = MultiscaleMessagePassing(args.scales, args.mlp_layers, args.hidden_channels)
            self.processor.append(GraphNet)
        # Decoder MLP
        self.decoder = MLP(args.mlp_layers,args.hidden_channels,args.hidden_channels,args.out_channels)

    def forward(self, x , edge_index, edge_attr, node_positions):
        '''Performs a forward pass across the GNN

        :param x:           torch.Tensor with shape [batch_size-1,n_nodes,in_node]. It is the input features
                            tensor.
        :param edge_index:  torch.Tensor with shape [2,n_edges]. It is the edge connectivity matrix
                            of the mesh, where edge_index[0] returns the source nodes and edge_index[1]
                            returns the destination nodes.
        :param edge_attr:   torch.Tensor with shape [batch_size-1,n_edges,in_edge]. It is the matrix containing
                            the edge fetaures for each edge in the mesh
        :return:            torch.Tensor with shape [batch_size-1,n_nodes,in_node]. It is the output
                            features tensor.
        '''
        #Encode
        # tracemalloc.start()


        if verbose:

            print('x:', type(x), x.size(), '\n')

        x = self.encoder_node(x)

        if verbose:

            print('x aft encode:', type(x), x.size(), '\n')

            print('edge_attr:', type(edge_attr), edge_attr.size(), '\n')

        edge_attr = self.encoder_edge(edge_attr)

        if verbose:

            print('edge_attr aft encode:', type(edge_attr), edge_attr.size(), '\n')

            print('node_positions:', type(node_positions), node_positions.size(), '\n')

            # print('node_positions[0]:', type(node_positions[0]), len(node_positions[0]), '\n')

            print('edge_index:', type(edge_index), edge_index.size(), '\n')


        #Process
        for GraphNet in self.processor:
            # x_res, edge_attr_res = GraphNet(x, edge_index, edge_attr)
            x, edge_attr = GraphNet(x, edge_index, edge_attr, node_positions)
            # x += x_res
            # edge_attr += edge_attr_res

        #Decode

        x = self.decoder(x)

        return x
    
    def assign_maps_coarsening(self, fine2coarse_list, distances_list):
        '''

        Copies long to compute list of coarsened graphs maps and distances into MultiScaleMessagePassing stucture
        fine2coarse_list:   list of tensors with shape [n_nodes_scale], 
        distances_list:     list of tensors with shape [1, n_nodes_scale, 1]
        '''
        for GraphNet in self.processor:
            GraphNet.assign(fine2coarse_list, distances_list)
        return
    

