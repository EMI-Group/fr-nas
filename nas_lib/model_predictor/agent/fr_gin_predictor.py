import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv
from torch_geometric.nn import global_mean_pool



class DirectedGINConv(nn.Module):
    """
    "To make information flow both ways, we always use the average of two GCN layers:
    one where we use A to propagate information in the forward directions
    and another where we use A^T to reverse the direction"
    Reimplement in GIN way, using NPNAS idea using directed convolution
    """

    def __init__(self, in_dim, out_dim):
        super(DirectedGINConv, self).__init__()
        nn1 = Sequential(Linear(in_dim, out_dim, bias=True), ReLU(), Linear(out_dim, out_dim, bias=True))
        nn2 = Sequential(Linear(in_dim, out_dim, bias=True), ReLU(), Linear(out_dim, out_dim, bias=True))
        self.forward_layer = GINConv(nn1)
        self.backward_layer = GINConv(nn2)
        self.batch_norm = nn.BatchNorm1d(out_dim)

    def forward(self, x, edge_index):
        output1 = F.relu(self.forward_layer(x, edge_index))
        reverse_idx = edge_index[[1, 0]]  # swap the order
        output2 = F.relu(self.backward_layer(x, reverse_idx))
        output = (output1 + output2) / 2
        output = self.batch_norm(output)
        return output

class DirectedGinPredictorAgent(nn.Module):
    '''
    Reimplement the method
    '''

    def __init__(self, input_dim=6, encoder_dim=32, decoder_dim=16):
        super(DirectedGinPredictorAgent, self).__init__()

        self.directed_gin_1 = DirectedGINConv(in_dim=input_dim, out_dim=encoder_dim)
        self.directed_gin_2 = DirectedGINConv(in_dim=encoder_dim, out_dim=encoder_dim)
        self.directed_gin_3 = DirectedGINConv(in_dim=encoder_dim, out_dim=encoder_dim)

        self.linear_before = torch.nn.Linear(encoder_dim, decoder_dim, bias=True)

        self.linear_mean = Linear(decoder_dim, 1)
        self.out_layer = torch.nn.Sigmoid()

        self.features = []


    def forward(self, graph_data):
        x = graph_data.x
        edge_index = graph_data.edge_index
        batch = graph_data.batch
        x = self.forward_batch(x, edge_index, batch)
        return x

    def forward_batch(self, x, edge_index, batch):
        x = self.directed_gin_1(x, edge_index)
        x = self.directed_gin_2(x, edge_index)
        x = self.directed_gin_3(x, edge_index)

        x_embedding = global_mean_pool(x, batch)

        x_embedding_mean = F.relu(self.linear_before(x_embedding))
        x_embedding_drop = F.dropout(x_embedding_mean, p=0.1, training=self.training)
        mean = self.linear_mean(x_embedding_drop)
        mean = self.out_layer(mean)
        return mean

# using normal GIN predictor


class GIN_Encoder(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=32):
        super(GIN_Encoder, self).__init__()

        dim = hidden_dim

        self.layers = nn.ModuleList()

        nn1 = Sequential(Linear(input_dim, dim, bias=True), ReLU(), Linear(dim, dim, bias=True))
        self.layers.append(GINConv(nn1))
        self.layers.append(torch.nn.BatchNorm1d(dim))

        nn2 = Sequential(Linear(dim, dim, bias=True), ReLU(), Linear(dim, dim, bias=True))
        self.layers.append(GINConv(nn2))
        self.layers.append(torch.nn.BatchNorm1d(dim))
        #
        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.layers.append(GINConv(nn3))
        self.layers.append(torch.nn.BatchNorm1d(dim))

        self.num_layers = len(self.layers)//2

    def forward(self, x, edge_index, batch):
        for i in range(self.num_layers):
            x = self.layers[i*2](x, edge_index)
            x = self.layers[i*2+1](x)

        return x


class FRGinPredictorAgentAfter(nn.Module):
    def __init__(self, encoder_forward, encoder_reverse, encoder_dim=32, decoder_dim=16):
        super(FRGinPredictorAgentAfter, self).__init__()

        self.gin_encoder_forward = encoder_forward
        self.gin_encoder_reverse = encoder_reverse

        for name, param in self.named_parameters():
            param.requires_grad = False

        self.linear_before = torch.nn.Linear(encoder_dim * 2, decoder_dim * 2, bias=True)
        self.linear_middle = torch.nn.Linear(decoder_dim * 2, decoder_dim, bias=True)
        self.linear_mean = Linear(decoder_dim, 1)
        self.out_layer = torch.nn.Sigmoid()

    def forward(self, graph_data):

        x = graph_data.x
        edge_index = graph_data.edge_index
        batch = graph_data.batch
        x = self.forward_batch(x, edge_index, batch)
        return x

    def forward_batch(self, x, edge_index, batch):
        reverse_edge_idx = edge_index[[1, 0]]  # swap the order
        x_forward = self.gin_encoder_forward(x, edge_index, batch)
        x_embedding_forward = global_mean_pool(x_forward, batch)

        x_reverse = self.gin_encoder_reverse(x, reverse_edge_idx, batch)
        x_embedding_reverse = global_mean_pool(x_reverse, batch)

        x_embedding = torch.cat((x_embedding_forward, x_embedding_reverse), 1)

        x_embedding_mean = F.relu(self.linear_before(x_embedding))
        x_embedding_mean = F.relu(self.linear_middle(x_embedding_mean))
        x_embedding_drop = F.dropout(x_embedding_mean, p=0.1, training=self.training)
        mean = self.linear_mean(x_embedding_drop)
        mean = self.out_layer(mean)
        return mean

class FRGinPredictorAgent(nn.Module):
    '''
    Reimplement the method
    '''

    def __init__(self, input_dim=6, encoder_dim=32, decoder_dim=16):
        super(FRGinPredictorAgent, self).__init__()

        self.gin_encoder = GIN_Encoder(input_dim=input_dim, hidden_dim=encoder_dim)

        self.linear_before = torch.nn.Linear(encoder_dim, decoder_dim, bias=True)

        self.linear_mean = Linear(decoder_dim, 1)
        self.out_layer = torch.nn.Sigmoid()

        self.features = []


    def forward(self, graph_data):
        self.features = []

        x = graph_data.x
        edge_index = graph_data.edge_index
        batch = graph_data.batch
        x = self.forward_batch(x, edge_index, batch)
        return x

    def forward_batch(self, x, edge_index, batch):
        x = self.gin_encoder(x, edge_index, batch)

        x_embedding = global_mean_pool(x, batch)

        self.features.append(x_embedding)

        x_embedding_mean = F.relu(self.linear_before(x_embedding))
        x_embedding_drop = F.dropout(x_embedding_mean, p=0.1, training=self.training)
        mean = self.linear_mean(x_embedding_drop)
        mean = self.out_layer(mean)
        return mean


