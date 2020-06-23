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


class WeightedGraphNet(nn.Module):
    def __init__(self, data_sample, out_channels=24, iterations=4, hidden_output_nodes=16, output_nodes=1):
        super(WeightedGraphNet, self).__init__()
        self.n_edge_features = data_sample.edge_attr.size()[1]
        self.n_node_features = data_sample.num_node_features
        self.iterations = iterations
        self.output_nodes = output_nodes
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # maps edge features edge_attr of shape [-1, num_edge_features] to shape [-1, in_channels * out_channels]
        self.first_edge_nn = Sequential(
            Linear(self.n_edge_features, self.n_node_features * out_channels)
        ).to(device)
        self.edge_nn = Sequential(
            Linear(self.n_edge_features, out_channels * out_channels)
        ).to(device)
        # in_channels is the number of node_features
        # out_channels is the number of OUTPUT node_features
        # NNConv
        self.nnconv_input = NNConv(data_sample.num_node_features, out_channels, self.first_edge_nn, aggr="add").to(device)
        self.pooling = TopKPooling(out_channels).to(device)
        self.nnconvs = [NNConv(out_channels, out_channels, self.edge_nn, aggr="add").to(device) for _ in range(iterations)]
        # GCNConv
        # self.nnconv_input = CGConv(data_sample.num_node_features, dim=self.n_edge_features).to(device)
        # self.pooling = TopKPooling(out_channels).to(device)
        # self.nnconvs = [CGConv(data_sample.num_node_features, dim=self.n_edge_features).to(device) for _ in range(iterations)]
        self.flatten = Flatten().to(device)
        self.output = Sequential(
            Linear(out_channels * data_sample.x.size()[0], hidden_output_nodes).to(device),
            ELU(),
            Linear(hidden_output_nodes, output_nodes).to(device)
        ).to(device)

    def forward(self, sample):
        # Expand convolution space from input to out_channels
        x, edge_index, edge_attr = sample.x, sample.edge_index, sample.edge_attr
        x = self.nnconv_input(x, edge_index, edge_attr)
        x = F.relu(x)

        # x = torch.tanh(x)

        # Do some convolutions
        for i in range(self.iterations):
            x = self.nnconvs[i](x, edge_index, edge_attr)
            x = F.relu(x)
            # x = torch.tanh(x)

        # print(x.size()[0])
        # x, edge_index, edge_attr, _, _, _ = self.pooling(x, edge_index, edge_attr)
        # x = torch.relu(x)
        x = self.flatten(x)
        # print(x.size())
        # Return output as a single value
        x = self.output(x)
        return torch.tanh(x)


class MultiGraphNet(nn.Module):
    def __init__(self, graph_nets: list):
        super(MultiGraphNet, self).__init__()
        self.nets = graph_nets
        in_nodes = sum([n.output_nodes for n in graph_nets])
        self.out = Sequential(
            Linear(in_nodes, 1)
        )
    
    def forward(self, samples):
        xs = [self.nets[i](sample) for i, sample in enumerate(samples)]
        return self.out(torch.cat(xs))


class UnweightedDebruijnGraphNet(nn.Module):
    def __init__(self, sample=None, out_channels=4):
        super(UnweightedDebruijnGraphNet, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input = GraphConv(1, out_channels=out_channels)
        self.conv1 = GraphConv(out_channels, 2*out_channels)
        self.conv2 = GraphConv(2*out_channels, 4*out_channels)
        self.conv3 = GraphConv(4*out_channels, 8*out_channels)
        # self.conv4 = GraphConv(8*out_channels, 16*out_channels)
        self.final_nodes = 41
        # self.pool = SortPooling(self.final_nodes)
        self.output = nn.Sequential(
            AdaptiveAvgPool1d(self.final_nodes),
            Flatten(),
            Flatten(),
            nn.Linear(self.final_nodes * out_channels * 8, 1)
        )

    def forward(self, sample):
        x = sample.x
        # Dropout layer
        # x = self.dropout(sample.x, dropout=0.3)
        x = self.input(x, sample.edge_index)
        x = F.gelu(x)
        x = self.conv1(x, sample.edge_index)
        x = F.gelu(x)
        x = self.conv2(x, sample.edge_index)
        x = F.gelu(x)
        x = self.conv3(x, sample.edge_index)
        x = F.gelu(x)
        return self.output(x.reshape(x.size()[1], x.size()[0]).unsqueeze(dim=1))

# It works good, converge faster (sin, cos even better) than the flat deep neural network
class UnweightedSimplifiedDebruijnGraphNet(nn.Module):
    def __init__(self, sample=None, out_channels=4, augmented_channels_multiplier=5, convlayers=4):
        super(UnweightedSimplifiedDebruijnGraphNet, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.empty_edges = torch.tensor([[], []], dtype=torch.long, device=self.device)
        self.channels = out_channels * augmented_channels_multiplier
        self.augmented_channels_multiplier = augmented_channels_multiplier
        self.dense_input = GraphConv(sample.num_node_features, augmented_channels_multiplier)
        self.input = GraphConv(augmented_channels_multiplier, self.channels)
        self.conv1 = GraphConv(self.channels, 2 * self.channels)
        self.conv2 = GraphConv(2 * self.channels, 4 * self.channels)
        self.conv3 = GraphConv(4 * self.channels, 8 * self.channels)
        # self.conv4 = GraphConv(8*out_channels, 16*out_channels)
        self.final_nodes = len(sample.x)
        self.convlayers = convlayers
        self.hidden_channels = 2**(convlayers-1)
        # self.pool = SortPooling(self.final_nodes)
        self.output = nn.Sequential(
            AdaptiveAvgPool1d(self.final_nodes),
            Flatten(),
            Flatten(),
            nn.Linear(self.final_nodes * self.channels * self.hidden_channels, 1)
            # nn.Linear(self.final_nodes*8*out_channels, 1)
        )

    def forward(self, sample):
        x, edge_index = sample.x, sample.edge_index
        x = self.dense_input(x, self.empty_edges)
        x = F.gelu(x)

        x = self.input(x, edge_index)
        x = F.gelu(x)
        if self.convlayers > 1:
            x = self.conv1(x, edge_index)
            x = F.gelu(x)
        if self.convlayers > 2:
            x = self.conv2(x, edge_index)
            x = F.gelu(x)
        if self.convlayers > 3:
            x = self.conv3(x, edge_index)
            x = F.gelu(x)
        # save_graph(x, sample.edge_index, "after_four.gv")
        # x = self.conv4(x, sample.edge_index)
        # x = F.relu(x)

        return self.output(x.reshape(x.size()[1], x.size()[0]).unsqueeze(dim=1))


