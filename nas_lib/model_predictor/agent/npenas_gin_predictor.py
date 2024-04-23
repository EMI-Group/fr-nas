import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv
from torch_geometric.nn import global_mean_pool



class NPENASGinPredictorAgent(nn.Module):
    '''
    Graph Isomorphism Network Predictor

    Wei, Chen, Chuang Niu, Yiping Tang, Yue Wang, Haihong Hu, and Jimin Liang.
    “NPENAS: Neural Predictor Guided Evolution for Neural Architecture Search.”
    IEEE Transactions on Neural Networks and Learning Systems, 2022, 1–15.
    https://doi.org/10.1109/TNNLS.2022.3151160.
    '''

    def __init__(self, input_dim=6):
        super(NPENASGinPredictorAgent, self).__init__()
        layers = []
        dim = 32
        dim2 = 16
        nn1 = Sequential(Linear(input_dim, dim, bias=True), ReLU(), Linear(dim, dim, bias=True))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim, bias=True), ReLU(), Linear(dim, dim, bias=True))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)
        #
        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        self.linear_before = torch.nn.Linear(dim, dim2, bias=True)

        self.linear_mean = Linear(dim2, 1)
        layers.append(self.linear_mean)
        layers.append(self.linear_before)
        self.out_layer = torch.nn.Sigmoid()

        # cause worse performance on NB101 and NB201, better performance on NB301
        # for layer in layers:
        #     if isinstance(layer, nn.Linear):
        #         nn.init.kaiming_uniform_(layer.weight, a=1)
        #         nn.init.constant_(layer.bias, 0)

    def forward(self, graph_data):
        x = graph_data.x
        edge_index = graph_data.edge_index
        batch = graph_data.batch
        return self.forward_batch(x, edge_index, batch)

    def forward_batch(self, x, edge_index, batch):
        x1 = F.relu(self.conv1(x, edge_index))
        x1 = self.bn1(x1)

        x2 = F.relu(self.conv2(x1, edge_index))
        x2 = self.bn2(x2)

        x3 = F.relu(self.conv3(x2, edge_index))
        x3 = self.bn3(x3)

        x_embedding = global_mean_pool(x3, batch)
        x_embedding_mean = F.relu(self.linear_before(x_embedding))
        x_embedding_drop = F.dropout(x_embedding_mean, p=0.1, training=self.training)
        mean = self.linear_mean(x_embedding_drop)
        mean = self.out_layer(mean)
        return mean
