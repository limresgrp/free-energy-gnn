import torch
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ELU
from torch_geometric.nn.conv import NNConv
from torch_geometric.nn.pool import TopKPooling


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        # batch_size = x.size(0)
        return x.view(-1)


"""
To be used with normal weighted graphs.
TODO: To be tweaked
"""
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
        self.nnconv_input = NNConv(data_sample.num_node_features, out_channels, self.first_edge_nn, aggr="max").to(device)
        self.pooling = TopKPooling(out_channels).to(device)
        self.nnconvs = [NNConv(out_channels, out_channels, self.edge_nn, aggr="max").to(device) for i in range(iterations)]
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
        x = F.relu6(x)

        # x = torch.tanh(x)

        # Do some convolutions
        for i in range(self.iterations):
            x = self.nnconvs[i](x, edge_index, edge_attr)
            x = F.relu6(x)
            # x = torch.tanh(x)

        # print(x.size()[0])
        # x, edge_index, edge_attr, _, _, _ = self.pooling(x, edge_index, edge_attr)
        # x = torch.relu(x)
        x = self.flatten(x)
        # print(x.size())
        # Return output as a single value
        x = self.output(x)
        return F.elu(x)


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

        x = self.out(torch.cat(xs))
        return F.relu(x)

