import torch
from torch_scatter import scatter
from torch.nn.functional import gumbel_softmax, softmax
from torch_geometric.utils import degree
import numpy as np


class GCNConv(torch.nn.Module):

    def __init__(self, input_num, out_num, dropout, node_nums, ab: bool = True):
        super().__init__()

        self.dropout = torch.nn.Dropout(dropout)
        self.W = torch.nn.Linear(in_features=input_num, out_features=out_num)
        self.ab = ab
        if ab:
            self.act = torch.nn.GELU()
            self.bn = torch.nn.BatchNorm1d(num_features=node_nums)

    def forward(self, x, edge_index, norm):
        x = self.dropout(x)
        x = self.W(x)
        if self.ab:
            x = self.act(self.bn(x))
        x = scatter(x[:, edge_index[1], :] * norm, index=edge_index[0], dim=1, dim_size=x.size(1))
        return x


class GCN(torch.nn.Module):

    def __init__(self, input_num, hidden_num, out_num, layer, dropout, edge_index, node_num):

        super().__init__()
        row, col = edge_index
        deg = degree(col, node_num)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        self.register_buffer("norm", norm.unsqueeze(0).unsqueeze(-1))
        self.register_buffer("edge_index", edge_index)

        if layer == 1:
            self.GCNConvs = torch.nn.ModuleList(
                [GCNConv(input_num=input_num, out_num=out_num, dropout=dropout, node_nums=node_num)])
        else:
            self.GCNConvs = torch.nn.ModuleList(
                [GCNConv(input_num=input_num, out_num=hidden_num, ab=True, dropout=dropout, node_nums=node_num)] +
                [GCNConv(input_num=hidden_num, out_num=hidden_num, ab=True, dropout=dropout, node_nums=node_num) for _
                 in range(layer - 2)] +
                [GCNConv(input_num=hidden_num, out_num=out_num, dropout=dropout, node_nums=node_num)])

    def forward(self, x):
        for gnn in self.GCNConvs:
            x = gnn(x, self.edge_index, self.norm)
        return x
