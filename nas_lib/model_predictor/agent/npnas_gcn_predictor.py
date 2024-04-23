import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn import global_mean_pool

class DirectedGCNConv(nn.Module):
    """
    "To make information flow both ways, we always use the average of two GCN layers:
    one where we use A to propagate information in the forward directions
    and another where we use A^T to reverse the direction"

    """

    def __init__(self, in_channels, out_channels, directed_gcn=True):
        super(DirectedGCNConv, self).__init__()
        self.forward_layer = GCNConv(in_channels=in_channels, out_channels=out_channels)
        self.backward_layer = GCNConv(in_channels=in_channels, out_channels=out_channels)
        self.directed_gcn = directed_gcn

    def forward(self, x, edge_index):
        output1 = F.relu(self.forward_layer(x, edge_index))
        if self.directed_gcn:
            reverse_idx = edge_index[[1, 0]]  # swap the order
        else:
            reverse_idx = edge_index[[0, 1]]
        output2 = F.relu(self.backward_layer(x, reverse_idx))
        output = (output1 + output2) / 2

        return output




class NPNASGcnPredictorAgent(nn.Module):
    """
    Graph Convolutional Network Predictor

    Implemented based on paper:

    Wen Wei, Hanxiao Liu, Hai Li, Yiran Chen, Gabriel Bender, and Pieter-Jan Kindermans.
    “Neural Predictor for Neural Architecture Search.” arXiv, December 2, 2019. http://arxiv.org/abs/1912.00848.
    refer to https://github.com/ultmaster/neuralpredictor.pytorch
    """

    def __init__(self, input_dim=5, gcn_hidden=144, gcn_layers=3, linear_hidden=128, directed_gcn=True):
        super(NPNASGcnPredictorAgent, self).__init__()


        self.gcn = [DirectedGCNConv(in_channels=input_dim if i == 0 else gcn_hidden, out_channels=gcn_hidden, directed_gcn=directed_gcn)
                        for i in range(gcn_layers)]


        self.gcn = nn.ModuleList(self.gcn)

        self.dropout = nn.Dropout(0.1)

        self.fc1 = nn.Linear(gcn_hidden, linear_hidden)
        self.fc2 = nn.Linear(linear_hidden, 1)

        # not mentioned on original papers
        # nn.init.kaiming_uniform_(self.fc1.weight, a=1)
        # nn.init.constant_(self.fc1.bias, 0)
        # nn.init.kaiming_uniform_(self.fc2.weight, a=1)
        # nn.init.constant_(self.fc2.bias, 0)

    def forward(self, graph_data):
        x = graph_data.x
        edge_index = graph_data.edge_index
        batch = graph_data.batch

        for layer in self.gcn:
            x = layer(x, edge_index)
        out = global_mean_pool(x, batch)

        # print(out)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out).view(-1)

        return out
