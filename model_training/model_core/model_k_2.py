import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import logging
from typing import Optional, Union
from torch_geometric.typing import Adj

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import trange

import math
import os

import torch
from torch import Tensor, nn
import torch.nn.functional as F

import torch_geometric as pyg
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn import Linear, BatchNorm, DeepGraphInfomax
from torch_geometric.utils import (
    remove_self_loops,
    add_self_loops,
    softmax,
    to_undirected,
)
class GraphAttention_layer(MessagePassing):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 attention_type: str = 'CUSTOM',
                 edge_dim: int = None,
                 flow: str = 'source_to_target',
                 heads: int = 1,
                 concat: bool = True,
                 dropout: float = 0.0,
                 add_self_loops: bool = False,
                 to_undirected: bool = False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        kwargs.setdefault('flow', flow)
        super(GraphAttention_layer, self).__init__(node_dim=0, **kwargs)

        assert attention_type == 'CUSTOM', "This version only supports the new CUSTOM attention"

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.to_undirected = to_undirected
        self.edge_dim = edge_dim

        self.lin_v = nn.Linear(input_dim, heads * output_dim, bias=False)
        self.lin_e = nn.Linear(edge_dim, heads * output_dim, bias=False)

        self.att = nn.Parameter(torch.Tensor(1, heads, output_dim))

        if concat:
            self.bias = nn.Parameter(torch.Tensor(heads * output_dim))
            self.weight_concat = nn.Parameter(torch.Tensor(heads * output_dim, output_dim))
        else:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
            self.register_parameter('weight_concat', None)

        self._alpha = None
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_v.weight)
        glorot(self.lin_e.weight)
        glorot(self.att)
        zeros(self.bias)
        glorot(self.weight_concat)

    def forward(self, x, edge_index, edge_attr=None, return_attention_weights: Optional[bool] = None):
        if self.to_undirected:
            edge_index = to_undirected(edge_index)

        if self.add_self_loops:
            edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
            edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value=1.0, num_nodes=x.size(0))
        else:
            edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)

        x_proj = self.lin_v(x).view(-1, self.heads, self.output_dim)
        edge_attr_proj = self.lin_e(edge_attr).view(-1, self.heads, self.output_dim)

        out = self.propagate(edge_index, x=(x_proj, x_proj), edge_attr=edge_attr_proj, size=None)


        alpha = self._alpha
        self._alpha = None
        unnorm_alpha = self._unnormalized_alpha
        if self.concat:
            out = out.view(-1, self.heads * self.output_dim)
            out += self.bias
            out = torch.matmul(out, self.weight_concat)
        else:
            out = out.mean(dim=1)
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            return out, edge_index, alpha, self._unnormalized_alpha
        else:
            return out

    def message(self, x_i, x_j, edge_attr, edge_index_i,size_i):
        Tau = 1.0
        alpha_input = x_i + x_j + edge_attr
        alpha = torch.tanh((alpha_input * self.att).sum(dim=-1))
        self._unnormalized_alpha = alpha.clone().detach()
        alpha = softmax(alpha / Tau, edge_index_i, num_nodes=size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.view(-1, self.heads, 1)

    def __repr__(self):
        return '{}({}, {}, heads={}, type={})'.format(self.__class__.__name__,
                                                      self.input_dim,
                                                      self.output_dim,
                                                      self.heads,
                                                      self.attention_type)
class Encoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 edge_dim: int,
                 heads_num: int = 1,
                 dropout: float = 0.0):
        super(Encoder, self).__init__()

        self.att_weights_first = None
        self.att_weights_second = None
        self.x_embs = None


        self.c = nn.Parameter(torch.Tensor([1.6]))
        self.d = nn.Parameter(torch.log(torch.tensor([19.0])))

        self.act = nn.GELU()
        self.layers = nn.ModuleList([])

        dims = [hidden_dim, hidden_dim]
        self.layers.append(nn.ModuleList([
            nn.BatchNorm1d(input_dim),
            GraphAttention_layer(input_dim, hidden_dim,
                                 heads=heads_num,
                                 concat=True,
                                 dropout=dropout,
                                 attention_type='CUSTOM',
                                 edge_dim=edge_dim),
        ]))


        self.layers.append(nn.ModuleList([
            nn.BatchNorm1d(hidden_dim),
            GraphAttention_layer(hidden_dim, output_dim,
                                 heads=heads_num,
                                 concat=True,
                                 dropout=dropout,
                                 attention_type='CUSTOM',
                                 edge_dim=edge_dim),
        ]))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_attr = data.edge_attr
        att_weights_in = []
        unnorm_att_weights_in = []

        for norm, attn_in in self.layers:
            x = norm(x)
            x, edge_index,att_weight, unnorm_weight = attn_in(x, edge_index, edge_attr, return_attention_weights=True)
            att_weights_in.append(att_weight)
            unnorm_att_weights_in.append(unnorm_weight)

        self.x_embs = x
        self.att_weights_first = att_weights_in[0]
        self.att_weights_second = att_weights_in[1]
        self.unnormalized_att_weights_first = unnorm_att_weights_in[0]
        self.unnormalized_att_weights_second = unnorm_att_weights_in[1]
        self.unnormalized_att_weights_mean = (
        self.unnormalized_att_weights_first + self.unnormalized_att_weights_second
    ) / 2
        return self.x_embs

class GRACE(nn.Module):
    def __init__(self, encoder: nn.Module, num_hidden: int, num_proj_hidden: int, tau: float = 0.5):
        super(GRACE, self).__init__()
        self.encoder = encoder
        self.tau = tau

        self.fc1 = nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, data):
        return self.encoder(data)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        return -torch.log(
            between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag())
        )

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True, batch_size: int = None) -> torch.Tensor:
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size is None:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            raise NotImplementedError("Batched loss not implemented in this version.")

        ret = (l1 + l2) * 0.5
        return ret.mean() if mean else ret.sum()
def get_network(
    keep_self_loops=False,
    edge_threshold_avgDegree=None,
    edge_threshold_zscore=None,
    chrom=None,
    edge_index=None,
    _att_coefs=None,
    idx2region=None,
    epoch=None
) -> nx.DiGraph:




    edge_index_with_selfloop, unnorm_att_coefs_mean = _att_coefs
    unnorm_att_coef_i = unnorm_att_coefs_mean.t()

    """
    # 去除 self-loop
    if not keep_self_loops:
        edge_index, unnorm_att_coef_i = remove_self_loops(edge_index_with_selfloop, unnorm_att_coef_i)
    else:
        edge_index = edge_index_with_selfloop
    """

    unnorm_attention_mean = unnorm_att_coef_i.mean(dim=0).cpu().numpy()


    scaled_att_coefs_all = pd.DataFrame({
        'from': edge_index[0].cpu().numpy().astype(int),
        'to': edge_index[1].cpu().numpy().astype(int),
        'unnorm_weights_mean': unnorm_attention_mean
    })


    att_weights = scaled_att_coefs_all['unnorm_weights_mean']
    if edge_threshold_avgDegree is not None:
        num_nodes = edge_index.max().item() + 1
        topk = num_nodes * edge_threshold_avgDegree
        filtered_edge_idx = np.argsort(-att_weights)[:topk]
    elif edge_threshold_zscore is not None:
        m, s = att_weights.mean(), att_weights.std()
        threshold = m + (edge_threshold_zscore * s)
        filtered_edge_idx = np.where(att_weights > threshold)[0]
    else:
        filtered_edge_idx = list(range(len(att_weights)))

    scaled_att_coefs_filtered = scaled_att_coefs_all.iloc[filtered_edge_idx, :].copy()


    G_nx = nx.from_pandas_edgelist(
        scaled_att_coefs_filtered,
        source='from',
        target='to',
        edge_attr=True,
        create_using=nx.DiGraph
    )

    if idx2region is not None:
        mappings = {idx: idx2region[idx] for idx in G_nx.nodes if idx in idx2region}
        G_nx = nx.relabel_nodes(G_nx, mappings)

    output_dir = f"./weights_combined_{epoch}/"
    file_name = f"{chrom}_attention_score.csv"
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    nx.write_edgelist(
        G_nx,
        output_dir + file_name,
        delimiter='\t',
        data=['unnorm_weights_mean']
    )
    return G_nx
