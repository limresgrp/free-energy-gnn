import torch
# import torch_geometric
import torch.nn as nn
from torch.nn import AdaptiveAvgPool1d
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ELU, AdaptiveMaxPool1d
from torch_geometric.nn.conv import NNConv, CGConv, GatedGraphConv, GraphConv
from torch_geometric.nn.pool import TopKPooling
# from torch_geometric.nn import global_sort_pool, global_add_pool
# from torch_geometric.data import Data
# from torch_geometric.utils import to_networkx
# from dgl import DGLGraph
# from dgl.nn.pytorch.glob import SortPooling
from drawing import save_graph
import random


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        # batch_size = x.size(0)
        return x.view(-1)


class SimplifiedLinearNet(nn.Module):
    def __init__(self, sample):
        super(SimplifiedLinearNet, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nodes = len(sample.x)
        self.input = nn.Linear(7, 7*6*2)
        self.middle = nn.Linear(7*6*2, self.nodes * 6 * 8)
        self.middle2 = nn.Linear(self.nodes * 6 * 8, self.nodes * 6 * 4 * 8)
        self.output = nn.Linear(self.nodes * 6 * 4 * 8, 1)
        
    def forward(self, sample):
        x = sample.x.view(-1)
        x = self.input(x)
        x = F.gelu(x)
        x = self.middle(x)
        x = F.gelu(x)
        x = self.middle2(x)
        x = F.gelu(x)
        return self.output(x)


class LinearNet(nn.Module):
    def __init__(self, sample):
        super(LinearNet, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nodes = len(sample.x)
        self.input = nn.Linear(self.nodes, 256)
        self.middle = nn.Linear(256, 1024)
        self.middle2 = nn.Linear(1024, 128)
        self.middle3 = nn.Linear(128, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, sample):
        x = sample.x.view(-1)
        x = self.input(x)
        x = F.gelu(x)
        x = self.middle(x)
        x = F.gelu(x)
        x = self.middle2(x)
        x = F.gelu(x)
        x = self.middle3(x)
        x = F.gelu(x)
        return self.output(x)

