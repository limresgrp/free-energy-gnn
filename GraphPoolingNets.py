import torch
# import torch_geometric
import torch.nn as nn
from torch.nn import AdaptiveAvgPool1d
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ELU, AdaptiveMaxPool1d
from torch_geometric.nn.conv import NNConv, CGConv, GatedGraphConv, GraphConv
from torch_geometric.nn.pool import TopKPooling
from torch_geometric.nn import global_sort_pool, global_add_pool, global_mean_pool, TopKPooling, avg_pool_x
# from torch_geometric.data import Data
# from torch_geometric.utils import to_networkx
# from dgl import DGLGraph
# from dgl.nn.pytorch.glob import SortPooling
from drawing import save_graph
import random
import numpy as np


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        # batch_size = x.size(0)
        return x.view(-1)


# It works good with SortPooling and 3 nodes
class TopKPoolingNet(nn.Module):
    def __init__(self, sample=None, pooling_layers=1, topk_ratio=0.6, final_pooling="avg_pool_x", dense_output = False, channels_optuna=2):
        super(TopKPoolingNet, self).__init__()
        out_channels = 4
        augmented_channels_multiplier = 4
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.empty_edges = torch.tensor([[], []], dtype=torch.long, device=self.device)
        self.channels = out_channels * augmented_channels_multiplier * channels_optuna
        self.augmented_channels_multiplier = augmented_channels_multiplier
        self.dense_input = GraphConv(sample.num_node_features, augmented_channels_multiplier)
        self.input = GraphConv(augmented_channels_multiplier, self.channels)
        self.topkpool1 = TopKPooling(self.channels, ratio=topk_ratio+0.1)
        self.conv1 = GraphConv(self.channels, 2 * self.channels)
        self.topkpool2 = TopKPooling(2*self.channels, ratio=topk_ratio)
        self.conv2 = GraphConv(2 * self.channels, 4 * self.channels)

        self.final_pooling = final_pooling
        self.pooling_layers = pooling_layers
        self.final_nodes = 3
        self.input_nodes_output_layer = self.final_nodes * self.channels * (2*pooling_layers) if pooling_layers > 0 else self.final_nodes * self.channels * 4
        if dense_output:
            self.output = nn.Sequential(
                nn.Linear(self.input_nodes_output_layer, 2 * self.channels),
                nn.GELU(),
                nn.Linear(2 * self.channels, 1)
            )
        else:
            self.output = nn.Linear(self.input_nodes_output_layer, 1)

    def forward(self, sample):
        x, edge_index = sample.x, sample.edge_index

        # Dropout layer
        # edge_index = self.dropout_edges(edge_index, dropout=0.2)

        x = self.dense_input(x, self.empty_edges)
        x = F.gelu(x)

        x = self.input(x, edge_index)
        x = F.gelu(x)
        if self.pooling_layers > 0:
            pooled = self.topkpool1(x, edge_index)
            x, edge_index = pooled[0], pooled[1]
            x = self.conv1(x, edge_index)
            x = F.gelu(x)
            if self.pooling_layers > 1:
                pooled = self.topkpool2(x, edge_index)
                x, edge_index = pooled[0], pooled[1]
                x = self.conv2(x, edge_index)
                x = F.gelu(x)

        # x = self.conv3(x, edge_index)
        # x = F.gelu(x)
        batch = torch.tensor([0 for _ in x], dtype=torch.long, device=self.device)
        # With sort_pool it works but we have the same problem: the output layer learns the order of the pooled nodes
        # using k = 3, let's see what happens by shuffling the nodes
        if self.final_pooling == "avg_pool_x":
            cluster = torch.as_tensor([i % self.final_nodes for i in range(len(x))], device=self.device)
            (x, cluster) = avg_pool_x(cluster, x, batch)
        elif self.final_pooling == "sort_pooling":
            x = global_sort_pool(x, batch, self.final_nodes)

        return self.output(x.view(-1))

    def dropout_edges(self, edge_index, dropout):
        # Do not drop anything in validation/test
        if not self.training:
            i = 1
            return edge_index

        indexes = set([i for i in range(len(edge_index[0]))])

        for i, edge in enumerate(edge_index[0]):
            to_drop = random.random()
            if to_drop < dropout:
                indexes.remove(i)

        return edge_index[:, list(indexes)]



