import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv
from torch_geometric.nn import global_mean_pool
import numpy


class MultiTransformGinPredictorAgent(nn.Module):
    def __init__(self, input_dim=6, num_transforms=2):
        super(MultiTransformGinPredictorAgent, self).__init__()

        self.num_transforms = num_transforms
        self.input_dim = input_dim
        self.dim = 36
        self.linear_dim = 54
        self.out_dim = 18

        self.build_gnn_block()

        self.mlp_layers = nn.ModuleList()
        for i in range(num_transforms):
            self.mlp_layers.append(Linear(self.dim, self.linear_dim, bias=True))

        self.mlp_layers.append(Linear(self.dim * num_transforms, self.linear_dim, bias=True))
        self.mlp_layers.append(Linear(self.linear_dim, self.out_dim))
        self.mlp_layers.append(Linear(self.out_dim, 1))
        self.out_layer = nn.Sigmoid()
        self.init_mlp()

    def init_mlp(self):
        for layer in self.mlp_layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=1)
                nn.init.constant_(layer.bias, 0)

    def build_gnn_block(self):
        input_dim = self.input_dim
        dim = self.dim
        num_transforms = self.num_transforms

        self.gnn_list = nn.ModuleList()

        self.gnn_weight_mask = [True for _ in range(num_transforms)]
        self.gnn_forward_switch = -1

        for _ in range(num_transforms):
            nn1 = Sequential(Linear(input_dim, dim, bias=True), ReLU(), Linear(dim, dim, bias=True))
            conv1 = GINConv(nn1)
            bn1 = torch.nn.BatchNorm1d(dim)

            nn2 = Sequential(Linear(dim, dim, bias=True), ReLU(), Linear(dim, dim, bias=True))
            conv2 = GINConv(nn2)
            bn2 = torch.nn.BatchNorm1d(dim)

            nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            conv3 = GINConv(nn3)
            bn3 = torch.nn.BatchNorm1d(dim)

            self.gnn_list.append(nn.ModuleList([conv1, bn1, conv2, bn2, conv3, bn3]))

    def forward(self, graph_data_list):
        x_embedding_list = []
        # print("-"*30)
        gnn_index = 0
        for graph_data in graph_data_list:
            if self.gnn_forward_switch < 0 or self.gnn_forward_switch == gnn_index:
                x = graph_data.x
                edge_index = graph_data.edge_index
                batch = graph_data.batch
                # print(x)
                # print(edge_index)
                # print(batch)
                x_embedding = self.forward_batch(x, edge_index, batch, gnn_index)
                x_embedding_list.append(x_embedding)
            gnn_index += 1

        # print(x_embedding_list)
        x_embedding = torch.cat(x_embedding_list, dim=1)  # note the dim 0 are for batches
        if self.gnn_forward_switch < 0:
            x_embedding_all = F.relu(self.mlp_layers[-3](x_embedding))
        else:
            x_embedding_all = F.relu(self.mlp_layers[self.gnn_forward_switch](x_embedding))

        x_embedding = F.relu(self.mlp_layers[-2](x_embedding_all))
        x_embedding_drop = F.dropout(x_embedding, p=0.1, training=self.training)
        mean = self.mlp_layers[-1](x_embedding_drop)
        mean = self.out_layer(mean)
        return mean

    def forward_batch(self, x, edge_index, batch, gnn_index):
        x1 = x
        index = 0
        for layer in self.gnn_list[gnn_index]:
            if index % 2 == 0:
                x1 = F.relu(layer(x1, edge_index))
            else:
                x1 = layer(x1)
            index += 1
        x_embedding = global_mean_pool(x1, batch)
        return x_embedding

    def set_gnn_weight_masking(self, gnn_weight_mask):
        """
        mask: one dim list, [True,False] indicating first gnn is activated
        weight controls gradient calculation
        forward controls calculation
        """
        self.gnn_weight_mask = gnn_weight_mask

        # freezing weight
        index = 0
        for gnn in self.gnn_list:
            if not gnn_weight_mask[index]:
                for params in gnn.parameters():
                    params.requires_grad = False
            else:
                for params in gnn.parameters():
                    params.requires_grad = True
            index += 1

    def switch_gnn_forward(self, gnn_forward_switch):
        """

        :param gnn_forward_switch:
        :return:
        """
        self.gnn_forward_switch = gnn_forward_switch
        # freezing weight
        gnn_weight_mask = [False for _ in range(self.num_transforms)]
        if gnn_forward_switch == -1:
            gnn_weight_mask = [True for _ in range(self.num_transforms)]
        else:
            gnn_weight_mask[gnn_forward_switch] = True
        self.set_gnn_weight_masking(gnn_weight_mask=gnn_weight_mask)


def multi_transform(matrix):
    transform_matrix_list = []
    transform_matrix_list.append(no_transform(matrix))
    transform_matrix_list.append(reverse_transform(matrix))
    return transform_matrix_list


def no_transform(matrix):
    matrix = numpy.array(matrix)
    assert len(matrix.shape) == 2 and matrix.shape[0] == matrix.shape[1]
    new_matrix = []

    for i in range(matrix.shape[0]):
        row = []
        for j in range(matrix.shape[1]):
            row.append(matrix[i][j])
        new_matrix.append(row)

    return matrix


def reverse_transform(matrix):
    matrix = numpy.array(matrix)
    assert len(matrix.shape) == 2 and matrix.shape[0] == matrix.shape[1]
    matrix = matrix.transpose()
    new_matrix = []
    for i in range(matrix.shape[0]):
        row = []
        for j in range(matrix.shape[1]):
            row.append(matrix[i][j])
        new_matrix.append(row)

    return matrix


def undirected_transform(matrix):
    """
    :param matrix: square matrix containing only 0 and 1
    """
    matrix = numpy.array(matrix)
    assert len(matrix.shape) == 2 and matrix.shape[0] == matrix.shape[1]
    transpose_matrix = matrix.transpose()
    new_matrix = []
    for i in range(matrix.shape[0]):
        row = []
        for j in range(matrix.shape[1]):
            row.append(min(matrix[i][j] + transpose_matrix[i][j], 1))
        new_matrix.append(row)
    return new_matrix
