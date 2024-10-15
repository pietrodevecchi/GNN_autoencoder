import torch

from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.utils import scatter
from torch_geometric.nn.models.meta import MetaLayer


class MP_block(torch.nn.Module):
    def __init__(self, hidden_channels, MP_layers, mlp_layers):
        super().__init__()
        self.MP_layers = MP_layers
        self.hidden_channels = hidden_channels
        self.MP_layers = torch.nn.ModuleList()
        for i in range(MP_layers):
            self.MP_layers.append(MetaLayer(EdgeModel(hidden_channels, mlp_layers), NodeModel(hidden_channels, mlp_layers)))
    
    def forward(self, x, edge_index, edge_attr, batch=None):
        for i in range(len(self.MP_layers)):
            x, edge_attr, u = self.MP_layers[i](x, edge_index, edge_attr, batch)
        return x, edge_attr, u
    
    def reset_parameters(self):
        for layer in self.MP_layers:
            layer.reset_parameters()


class EdgeModel(torch.nn.Module):
    def __init__(self,hidden_channels, mlp_layers):
        super().__init__()
        # self.edge_mlp = Seq(Lin(3*hidden_channels, hidden_channels), ReLU(), Lin(hidden_channels, hidden_channels))
        self.edge_mlp = MLP(mlp_layers , 3*hidden_channels, hidden_channels, hidden_channels)

    def forward(self, src, dst, edge_attr, u, batch):
        # src, dst: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        # out = torch.cat([src, dst, edge_attr, u[batch]], 1)
        # return self.edge_mlp(out)
        out = torch.cat([src, dst, edge_attr], 1)
        return self.edge_mlp(out)
    

class NodeModel(torch.nn.Module):
    def __init__(self, hidden_channels, mlp_layers):
        super().__init__()
        # self.node_mlp_1 = Seq(Lin(2*hidden_channels, hidden_channels), ReLU(), Lin(hidden_channels, hidden_channels))
        # self.node_mlp_2 = Seq(Lin(2*hidden_channels, hidden_channels), ReLU(), Lin(hidden_channels, hidden_channels))
        self.node_mlp_1 = MLP(mlp_layers, 2*hidden_channels, hidden_channels, hidden_channels)
        self.node_mlp_2 = MLP(mlp_layers, 2*hidden_channels, hidden_channels, hidden_channels)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter(out, col, dim=0, dim_size=x.size(0),
                      reduce='mean')
        out = torch.cat([x, out], dim=1)
        return self.node_mlp_2(out)
    

class MLP(torch.nn.Module):
    '''Class for creating a Multi-Layer Perceptron
          Attributes
            layers      (List)      A list of layers transforms a tensor x into f(Wx + b), where
                                    f is SiLU activation function, W is the weight matrix and b the bias tensor.


    '''
    def __init__(self, num_layers,in_channels,hidden_channels,out_channels):
        super(MLP, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(in_channels,hidden_channels))
        self.layers.append(torch.nn.ReLU())
        for _ in range(num_layers - 2):
            self.layers.append(torch.nn.Linear(hidden_channels,hidden_channels))
            self.layers.append(torch.nn.ReLU())
        self.layers.append(torch.nn.Linear(hidden_channels, out_channels))

               
    def forward(self, x):
        '''
        :param x: torch.Tensor
        :return: torch.Tensor
        '''
        for layer in self.layers:
            x = layer(x)
        return x        




# class Mlp(torch.nn.Module):
#     def __init__(self, input_size : int, widths : list[int], layernorm=True):
#         super().__init__()
#         widths = [input_size] + widths
#         modules = []
#         for i in range(len(widths) - 1):
#             if i < len(widths) - 2:
#                 modules.append(torch.nn.Sequential(
#                     torch.nn.Linear(widths[i], widths[i + 1]), torch.nn.ReLU()))
#             else:
#                 modules.append(torch.nn.Sequential(
#                     torch.nn.Linear(widths[i], widths[i + 1])))

#         if layernorm: modules.append(torch.nn.LayerNorm(widths[-1]))
#         self.model = torch.nn.Sequential(*modules)

#     def forward(self, x): return self.model(x)