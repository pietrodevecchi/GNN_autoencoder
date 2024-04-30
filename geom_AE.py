

from typing import Callable, List, Union, Optional, Tuple

import torch
from torch import Tensor


# from torch_geometric.nn import GCNConv, 
from geom_pooling import TopKPooling
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import OptTensor, PairTensor
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    to_torch_csr_tensor,
)
from torch_geometric.utils.repeat import repeat


from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.utils import scatter
from torch_geometric.nn.models.meta import MetaLayer

from geom_MPBlock import MUS_layer
from geom_base_models import EdgeModel, NodeModel, MP_block

from plot_nodes import plot_graph_positions, plot_mapping, plot_graph

# plot = 0


class Graph_AE(torch.nn.Module):
    r"""The Graph U-Net model from the `"Graph U-Nets"
    <https://arxiv.org/abs/1905.05178>`_ paper which implements a U-Net like
    architecture with graph pooling and unpooling operations.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        depth (int): The depth of the U-Net architecture.
        pool_ratios (float or [float], optional): Graph pooling ratio for each
            depth. (default: :obj:`0.5`)
        sum_res (bool, optional): If set to :obj:`False`, will use
            concatenation for integration of skip connections instead
            summation. (default: :obj:`True`)
        act (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.nn.functional.relu`)
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        mlp_layers: int,
        depth: int,
        pool_ratios: Union[float, List[float]] = 0.5,
        MP_block_layers: int = 2,
        multiscales: Union[float, List[float]] = [0.1, 0.2],
        sum_res: bool = True,
        act: Union[str, Callable] = 'relu',
        plot: int = 0,
        edge_augment_pooling: bool = False,
    ):
        super().__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.mlp_layers = mlp_layers
        self.MP_block_layers = MP_block_layers
        self.depth = depth
        self.pool_ratios = repeat(pool_ratios, depth)
        self.act = activation_resolver(act)
        self.sum_res = sum_res
        self.depth_mus = len(multiscales)
        self.multiscales = multiscales
        self.adj_augment = edge_augment_pooling
        self.plot = plot

        channels = hidden_channels

        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        # self.down_convs.append(GCNConv(in_channels, channels, improved=True))
        # self.down_convs.append(MP_block(channels, 2))
        self.down_convs.append(MUS_layer(in_channels, channels, channels, self.mlp_layers, self.MP_block_layers, 
                                         self.depth_mus, self.multiscales, True, 'relu', self.plot))
        # x, edge_attr= op(x, edge_index, edge_attr, batch)     
        for i in range(depth):
            self.pools.append(TopKPooling(channels, self.pool_ratios[i], nonlinearity='relu'))
            # self.down_convs.append(GCNConv(channels, channels, improved=True))
            # self.down_convs.append(MP_block(channels, 2))
            self.down_convs.append(MUS_layer(channels, channels, channels, self.mlp_layers, self.MP_block_layers,
                                             self.depth_mus, self.multiscales, True, 'relu', self.plot))

        in_channels = channels if sum_res else 2 * channels

        self.up_convs = torch.nn.ModuleList()
        for i in range(depth):
            # self.up_convs.append(GCNConv(in_channels, channels, improved=True))
            # self.up_convs.append(MetaLayer(EdgeModel(channels), NodeModel(channels)))
            self.up_convs.append(MUS_layer(channels, channels, channels, self.mlp_layers,
                                           self.MP_block_layers,
                                           self.depth_mus, self.multiscales, True, 'relu', self.plot))
        # self.up_convs.append(GCNConv(channels, out_channels, improved=True))

        self.up_convs.append(MUS_layer(channels, channels, channels, self.mlp_layers,
                                       self.MP_block_layers,
                                       self.depth_mus, self.multiscales, True, 'relu', self.plot))

        # self.up_convs.append(MetaLayer(EdgeModel(channels), NodeModel(channels)))

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for conv in self.up_convs:
            conv.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: OptTensor = None,
                batch: OptTensor = None, node_pos: OptTensor = None) -> Tensor:
        """"""  # noqa: D419
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        if edge_weight is None:
            edge_weight = x.new_ones(edge_index.size(1))

        # x = self.down_convs[0](x, edge_index, edge_weight)
        # x = self.act(x)
        # x, edge_weight, _= self.down_convs[0](x, edge_index, edge_attr=edge_weight)
        x, edge_weight = self.down_convs[0](x, edge_index, edge_attr=edge_weight, node_pos=node_pos)

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        node_poss = [node_pos]
        perms = []
        perm_edgess = []

        for i in range(1, self.depth + 1):
            if self.adj_augment:
                # qui tolgo perché non gira e non so cosa faccia
                (edge_index, edge_weight), edge_index_mapping = self.augment_adj(edge_index, edge_weight,
                                                            x.size(0))
                
            x, edge_index, edge_weight, batch, perm, perm_edges = self.pools[i - 1](
                x, edge_index, edge_weight)
            
            if self.adj_augment:
                perm_edges = edge_index_mapping[perm_edges]
            
            if self.plot:
                plot_graph_positions(node_pos, title="Graph", save_path='geom_graph.png')
                plot_graph_positions(node_pos[perm], title="Graph", save_path='geom_pooled_graph_pos.png')

                plot_graph(node_pos[perm], edge_index, "Pooled graph", "geom_pooled_graph_AE.png")

            node_pos = node_pos[perm]
            # x = self.down_convs[i](x, edge_index, edge_weight)
            # x = self.act(x)
            # x, edge_weight, _ = self.down_convs[i](x, edge_index, edge_attr=edge_weight) 
            x, edge_weight = self.down_convs[i](x, edge_index, edge_attr=edge_weight, node_pos=node_pos)

            if i < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
                node_poss += [node_pos]
    
            perms += [perm]
            perm_edgess += [perm_edges]

        
        x, edge_weight = self.up_convs[0](x, edge_index, edge_attr=edge_weight, node_pos=node_pos)

        for i in range(self.depth):
            j = self.depth - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            # edge_weight = edge_weights[j]
            perm = perms[j]
            perm_edges = perm_edgess[j]

            up = torch.zeros_like(res)
            up[perm] = x

            # questo sarebbe barare usi info del grafo che hai encoded
            # x = res + up if self.sum_res else torch.cat((res, up), dim=-1)
            x = up

            # unpooling edge_weight

            # up_edge_weight = torch.ones_like(edge_weights[j])
            # edge_weight = up_edge_weight

            up_edge_weight = torch.zeros_like(edge_weights[j])

            # perm_edges = torch.isin(edge_index[0], perm) & torch.isin(edge_index[1], perm)

            up_edge_weight[perm_edges] = edge_weight

            edge_weight = up_edge_weight


            # x = self.up_convs[i](x, edge_index, edge_weight)
            # x = self.act(x) if i < self.depth - 1 else x
            # if i < self.depth - 1:
            #     x, edge_weight, _ = self.up_convs[i](x, edge_index, edge_attr=edge_weight)
            # else:
            x, edge_weight = self.up_convs[i+1](x, edge_index, edge_attr=edge_weight, node_pos=node_poss[j])


        return x


    def augment_adj(self, edge_index: Tensor, edge_weight: Tensor,
                    num_nodes: int) -> Tuple[PairTensor, Tensor]:
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        # Save the original edge_index.
        original_edge_index = edge_index

        # Split the edge_weight tensor along the last dimension.
        edge_weights = torch.split(edge_weight, 1, dim=-1)

        adjacencies = []
        for edge_weight in edge_weights:
            # Remove the extra dimension from edge_weight.
            edge_weight = torch.squeeze(edge_weight, -1)

            # Use the original edge_index in each iteration.
            edge_index, edge_weight = add_self_loops(original_edge_index, edge_weight,
                                                     num_nodes=num_nodes)
            adj = to_torch_csr_tensor(edge_index, edge_weight,
                                      size=(num_nodes, num_nodes))
            adj = (adj @ adj).to_sparse_coo()
            edge_index, edge_weight = adj.indices(), adj.values()
            edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

            # Add the extra dimension back to edge_weight.
            edge_weight = torch.unsqueeze(edge_weight, -1)

            adjacencies.append(edge_weight)

        # Stack the adjacency matrices back together.
        edge_weight = torch.cat(adjacencies, dim=-1)

        original_edge_indices = torch.arange(original_edge_index.size(1), device=edge_index.device, dtype=torch.long)

        # Create a mapping from original edge indices to augmented edge indices.

        # new_edge_indices = torch.arange(edge_index.size(1), device=edge_index.device)

        edge_index_mapping = torch.zeros(edge_index.size(1), device=edge_index.device, dtype=torch.long)
        edge_index_mapping[original_edge_indices] = original_edge_indices

        return (edge_index, edge_weight), edge_index_mapping

    # def augment_adj(self, edge_index: Tensor, edge_weight: Tensor,
    #                 num_nodes: int) -> PairTensor:
    #     edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    #     edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
    #                                              num_nodes=num_nodes)
    #     adj = to_torch_csr_tensor(edge_index, edge_weight,
    #                               size=(num_nodes, num_nodes))
    #     adj = (adj @ adj).to_sparse_coo()
    #     edge_index, edge_weight = adj.indices(), adj.values()
    #     edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    #     return edge_index, edge_weight

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.hidden_channels}, {self.out_channels}, '
                f'depth={self.depth}, pool_ratios={self.pool_ratios})')
    



############################################################################################################
############################################################################################################    
############################################################################################################



class Graph_AE_noMMP(torch.nn.Module):
    r"""The Graph U-Net model from the `"Graph U-Nets"
    <https://arxiv.org/abs/1905.05178>`_ paper which implements a U-Net like
    architecture with graph pooling and unpooling operations.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        depth (int): The depth of the U-Net architecture.
        pool_ratios (float or [float], optional): Graph pooling ratio for each
            depth. (default: :obj:`0.5`)
        sum_res (bool, optional): If set to :obj:`False`, will use
            concatenation for integration of skip connections instead
            summation. (default: :obj:`True`)
        act (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.nn.functional.relu`)
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        mlp_layers: int,
        MP_block_layers: int,
        depth: int,
        pool_ratios: Union[float, List[float]] = 0.5,
        sum_res: bool = True,
        act: Union[str, Callable] = 'relu',
        plot: int = 0,
        edge_augment_pooling: bool = False,
    ):
        super().__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.mlp_layers = mlp_layers
        self.MP_block_layers = MP_block_layers
        self.depth = depth
        self.pool_ratios = repeat(pool_ratios, depth)
        self.act = activation_resolver(act)
        self.sum_res = sum_res
        self.adj_augment = edge_augment_pooling
        self.plot = plot

        channels = hidden_channels

        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        # self.down_convs.append(GCNConv(in_channels, channels, improved=True))
        self.down_convs.append(MP_block(channels, self.MP_block_layers, self.mlp_layers))
        # self.down_convs.append(MUS_layer(in_channels, channels, channels, self.depth_mus, self.multiscales, True, 'relu'))
        # x, edge_attr= op(x, edge_index, edge_attr, batch)     
        for i in range(depth):
            self.pools.append(TopKPooling(channels, self.pool_ratios[i], nonlinearity='relu'))
            # self.down_convs.append(GCNConv(channels, channels, improved=True))
            self.down_convs.append(MP_block(channels, self.MP_block_layers, self.mlp_layers))
            # self.down_convs.append(MUS_layer(channels, channels, channels, self.depth_mus, self.multiscales, True, 'relu'))

        in_channels = channels if sum_res else 2 * channels

        self.up_convs = torch.nn.ModuleList()
        for i in range(depth):
            # self.up_convs.append(GCNConv(in_channels, channels, improved=True))
            # self.up_convs.append(MetaLayer(EdgeModel(channels), NodeModel(channels)))
            # self.up_convs.append(MUS_layer(channels, channels, channels, self.depth_mus, self.multiscales, True, 'relu'))
            self.up_convs.append(MP_block(channels, self.MP_block_layers, self.mlp_layers))
        # self.up_convs.append(GCNConv(channels, out_channels, improved=True))

        # self.up_convs.append(MUS_layer(channels, channels, channels, self.depth_mus, self.multiscales, True, 'relu'))
        self.up_convs.append(MP_block(channels, self.MP_block_layers, self.mlp_layers))

        # self.up_convs.append(MetaLayer(EdgeModel(channels), NodeModel(channels)))

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for conv in self.up_convs:
            conv.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: OptTensor = None,
                batch: OptTensor = None, node_pos: OptTensor = None) -> Tensor:
        """"""  # noqa: D419
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        if edge_weight is None:
            edge_weight = x.new_ones(edge_index.size(1))

        # x = self.down_convs[0](x, edge_index, edge_weight)
        # x = self.act(x)
        x, edge_weight, _= self.down_convs[0](x, edge_index, edge_attr=edge_weight)
        # x, edge_weight = self.down_convs[0](x, edge_index, edge_attr=edge_weight, node_pos=node_pos)

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        node_poss = [node_pos]
        perms = []
        perm_edgess = []

        for i in range(1, self.depth + 1):
            if self.adj_augment:
                # qui tolgo perché non gira e non so cosa faccia
                (edge_index, edge_weight), edge_index_mapping = self.augment_adj(edge_index, edge_weight,
                                                            x.size(0))
                
            x, edge_index, edge_weight, batch, perm, perm_edges = self.pools[i - 1](
                x, edge_index, edge_weight)
            
            if self.adj_augment:
                perm_edges = edge_index_mapping[perm_edges]
            
            if self.plot:
                plot_graph_positions(node_pos, title="Graph", save_path='geom_graph.png')
                plot_graph_positions(node_pos[perm], title="Graph", save_path='geom_pooled_graph_pos.png')

                plot_graph(node_pos[perm], edge_index, "Pooled graph", "geom_pooled_graph_AE.png")

            node_pos = node_pos[perm]
            # x = self.down_convs[i](x, edge_index, edge_weight)
            # x = self.act(x)
            x, edge_weight, _ = self.down_convs[i](x, edge_index, edge_attr=edge_weight) 
            # x, edge_weight = self.down_convs[i](x, edge_index, edge_attr=edge_weight, node_pos=node_pos)

            if i < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
                node_poss += [node_pos]
    
            perms += [perm]
            perm_edgess += [perm_edges]

        
        # x, edge_weight = self.up_convs[0](x, edge_index, edge_attr=edge_weight, node_pos=node_pos)
        x, edge_weight, _ = self.up_convs[0](x, edge_index, edge_attr=edge_weight)

        for i in range(self.depth):
            j = self.depth - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            # edge_weight = edge_weights[j]
            perm = perms[j]
            perm_edges = perm_edgess[j]

            up = torch.zeros_like(res)
            up[perm] = x

            # questo sarebbe barare usi info del grafo che hai encoded
            # x = res + up if self.sum_res else torch.cat((res, up), dim=-1)
            x = up

            # unpooling edge_weight

            # up_edge_weight = torch.ones_like(edge_weights[j])
            # edge_weight = up_edge_weight

            up_edge_weight = torch.zeros_like(edge_weights[j])

            # perm_edges = torch.isin(edge_index[0], perm) & torch.isin(edge_index[1], perm)

            up_edge_weight[perm_edges] = edge_weight

            edge_weight = up_edge_weight


            # x = self.up_convs[i](x, edge_index, edge_weight)
            # x = self.act(x) if i < self.depth - 1 else x
            # if i < self.depth - 1:
            #     x, edge_weight, _ = self.up_convs[i](x, edge_index, edge_attr=edge_weight)
            # else:
            # x, edge_weight = self.up_convs[i+1](x, edge_index, edge_attr=edge_weight, node_pos=node_poss[j])
            x, edge_weight, _ = self.up_convs[i+1](x, edge_index, edge_attr=edge_weight)


        return x


    def augment_adj(self, edge_index: Tensor, edge_weight: Tensor,
                    num_nodes: int) -> Tuple[PairTensor, Tensor]:
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        # Save the original edge_index.
        original_edge_index = edge_index

        # Split the edge_weight tensor along the last dimension.
        edge_weights = torch.split(edge_weight, 1, dim=-1)

        adjacencies = []
        for edge_weight in edge_weights:
            # Remove the extra dimension from edge_weight.
            edge_weight = torch.squeeze(edge_weight, -1)

            # Use the original edge_index in each iteration.
            edge_index, edge_weight = add_self_loops(original_edge_index, edge_weight,
                                                     num_nodes=num_nodes)
            adj = to_torch_csr_tensor(edge_index, edge_weight,
                                      size=(num_nodes, num_nodes))
            adj = (adj @ adj).to_sparse_coo()
            edge_index, edge_weight = adj.indices(), adj.values()
            edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

            # Add the extra dimension back to edge_weight.
            edge_weight = torch.unsqueeze(edge_weight, -1)

            adjacencies.append(edge_weight)

        # Stack the adjacency matrices back together.
        edge_weight = torch.cat(adjacencies, dim=-1)

        original_edge_indices = torch.arange(original_edge_index.size(1), device=edge_index.device, dtype=torch.long)

        # Create a mapping from original edge indices to augmented edge indices.

        # new_edge_indices = torch.arange(edge_index.size(1), device=edge_index.device)

        edge_index_mapping = torch.zeros(edge_index.size(1), device=edge_index.device, dtype=torch.long)
        edge_index_mapping[original_edge_indices] = original_edge_indices

        return (edge_index, edge_weight), edge_index_mapping

    # def augment_adj(self, edge_index: Tensor, edge_weight: Tensor,
    #                 num_nodes: int) -> PairTensor:
    #     edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    #     edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
    #                                              num_nodes=num_nodes)
    #     adj = to_torch_csr_tensor(edge_index, edge_weight,
    #                               size=(num_nodes, num_nodes))
    #     adj = (adj @ adj).to_sparse_coo()
    #     edge_index, edge_weight = adj.indices(), adj.values()
    #     edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    #     return edge_index, edge_weight

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.hidden_channels}, {self.out_channels}, '
                f'depth={self.depth}, pool_ratios={self.pool_ratios})')


def print_tensor_sizes(tensor, prefix=''):
    print(f'{prefix}size: {tensor.size()}')
    if tensor.dim() > 0:
        for i, sub_tensor in enumerate(tensor):
            print_tensor_sizes(sub_tensor, prefix=f'{prefix}[{i}]')