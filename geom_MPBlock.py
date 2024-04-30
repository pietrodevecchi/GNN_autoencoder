from typing import Callable, List, Union

from plot_nodes import plot_graph_positions, plot_mapping, plot_graph

import torch
from torch import Tensor

from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import OptTensor, PairTensor
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    to_torch_csr_tensor,
)
from torch_geometric.utils.repeat import repeat

from torch_geometric.nn.models.meta import MetaLayer

from geom_base_models import EdgeModel, NodeModel, MP_block

from torch_geometric.nn.pool import voxel_grid
from torch_geometric.nn import avg_pool 

from torch_geometric.data import Data


# plot = 0

class MUS_layer(torch.nn.Module):
    r"""The Graph U-Net model from the `"Graph U-Nets"
    <https://arxiv.org/abs/1905.05178>`_ paper which implements a U-Net like
    architecture with graph pooling and unpooling operations.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        depth (int): The depth of the U-Net architecture.
        scales (float or [float], optional): Graph pooling ratio for each
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
        scales: Union[float, List[float]] = 0.5,
        sum_res: bool = True,
        act: Union[str, Callable] = 'relu',
        plot: int = 0
    ):
        super().__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.mlp_layers = mlp_layers
        self.MP_block_layers = MP_block_layers
        self.depth = depth
        self.scales = scales if isinstance(scales, list) else [scales] * depth
        self.act = activation_resolver(act)
        self.sum_res = sum_res
        self.plot = plot



        channels = hidden_channels

        self.down_convs = torch.nn.ModuleList()

        # self.pools = torch.nn.ModuleList()
        
        self.down_convs.append(MP_block(channels, self.MP_block_layers, self.mlp_layers))

        for i in range(depth):
            # self.pools.append(TopKPooling(channels, self.scales[i], nonlinearity='sigmoid'))
           
            self.down_convs.append(MP_block(channels, self.MP_block_layers, self.mlp_layers))

        in_channels = channels if sum_res else 2 * channels

        self.up_convs = torch.nn.ModuleList()
        for i in range(depth - 1):

            self.up_convs.append(MP_block(channels, self.MP_block_layers, self.mlp_layers))

        self.up_convs.append(MP_block(channels, self.MP_block_layers, self.mlp_layers))


        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.down_convs:
            conv.reset_parameters()
        # for pool in self.pools:
        #     pool.reset_parameters()
        for conv in self.up_convs:
            conv.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: OptTensor = None, node_pos: OptTensor = None,
                batch: OptTensor = None) -> Tensor:
        """"""  # noqa: D419
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        if edge_attr is None:
            print("Edge_attr is None")
            edge_attr = x.new_ones(edge_index.size(1))

        x, edge_attr, _ = self.down_convs[0](x, edge_index, edge_attr)
        # x = self.act(x)

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_attr]
        perms = []

        

        for i in range(1, self.depth + 1):
            # edge_index, edge_attr = self.augment_adj(edge_index, edge_attr,
            #                                            x.size(0))
            perm_not_ordered = voxel_grid(node_pos, size=[self.scales[i-1], self.scales[i-1]], start=[0,0], end=[1, 1]) #, device=x.device)

            unique_perm, inverse_indices = torch.unique(perm_not_ordered, return_inverse=True)
            perm = torch.arange(unique_perm.size(0), device=x.device)[inverse_indices]

            data_old = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch, pos=node_pos)

            if self.plot:
                plot_graph(data_old.pos, data_old.edge_index, "Original graph", "geom_original_graph.png")
                

            data = avg_pool(perm, data_old)
            
            if self.plot:
                plot_graph(data.pos, data.edge_index, "Pooled graph", "geom_pooled_graph.png")
                plot_mapping(data_old.pos, data.pos, perm, "Mappping", "geom_mapping.png")

            # se avg_pool fa anche una media delle posizioni poi posso reiterare,
            # il nuovo nodo non sarà il centroid del voxel ma del cluster

            x = data.x
            edge_index = data.edge_index
            edge_attr = data.edge_attr
            batch = data.batch
            node_pos = data.pos
            # x, edge_index, edge_attr, batch, perm, _ = self.pools[i - 1](
            #     x, edge_index, edge_attr, batch)

            x, edge_attr, _ = self.down_convs[i](x, edge_index, edge_attr)
            # x = self.act(x)

            if i < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_attr]
            perms += [perm]

        for i in range(self.depth):
            j = self.depth - 1 - i

            edge_index_down = edge_index
            res = xs[j]
            edge_index = edge_indices[j]
            res_edge_attr = edge_weights[j]
            perm = perms[j]

            # up = torch.zeros_like(res)
            
            # up[perm] = x
            up = x[perm]

            x = res + up if self.sum_res else torch.cat((res, up), dim=-1)

            unpool_edges=1

            if unpool_edges:
                perm_edges = unpool_edge(perm, perm_not_ordered, inverse_indices, edge_index_down, edge_index)
                up_edge = torch.zeros_like(res_edge_attr)
                up_edge[perm_edges] = edge_attr
                edge_attr = res_edge_attr + up_edge if self.sum_res else torch.cat((res_edge_attr, up_edge), dim=-1)
            
            else:
                edge_attr = res_edge_attr
            
            x, edge_attr, _ = self.up_convs[i](x, edge_index, edge_attr)
            # x = self.act(x) if i < self.depth - 1 else x

        return x, edge_attr

    def augment_adj(self, edge_index: Tensor, edge_attr: Tensor,
                    num_nodes: int) -> PairTensor:
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr,
                                                 num_nodes=num_nodes)
        adj = to_torch_csr_tensor(edge_index, edge_attr,
                                  size=(num_nodes, num_nodes))
        # adj = adj.to_dense()  # Convert to dense tensor
        adj = adj @ adj
        adj = adj.to_sparse_coo()  # Convert back to sparse tensor
        # adj = (adj @ adj).to_sparse_coo()
        edge_index, edge_attr = adj.indices(), adj.values()
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        return edge_index, edge_attr

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.hidden_channels}, {self.out_channels}, '
                f'depth={self.depth}, scales={self.scales})')

def unpool_edge(perm: Tensor, perm_not_ordered: Tensor, inverse_indices: Tensor, pooled_edge_index: Tensor, edge_index: Tensor) -> Tensor:
    # Create a mapping from original nodes to pooled nodes
    orig_to_pooled = perm.new_full((perm_not_ordered.max().item() + 1,), -1)
    orig_to_pooled[perm_not_ordered] = torch.arange(perm.size(0), device=perm.device)

    # Use inverse_indices to map pooled edges back to original edges
    mapped_edge_index = inverse_indices[edge_index]

    # Rest of your unpooling code...

    # Map the edges in the original graph to the pooled graph
    # mapped_edge_index = orig_to_pooled[edge_index]

    # Create a tensor to hold the mapping from pooled edges to original edges
    pooled_to_orig = torch.full((pooled_edge_index.size(1),), -1, dtype=torch.long, device=pooled_edge_index.device)

    # For each edge in the pooled graph, find the corresponding edge in the original graph
    for i in range(pooled_edge_index.size(1)):
        # Get the source and target nodes of the edge in the pooled graph
        source, target = pooled_edge_index[:, i]

        # Find the corresponding edge in the original graph
        orig_edge = torch.where((mapped_edge_index[0] == source) & (mapped_edge_index[1] == target))[0]

        # If the edge is found in the original graph, store its index in the mapping
        if orig_edge.numel() > 0:
            pooled_to_orig[i] = orig_edge[0]

    return pooled_to_orig