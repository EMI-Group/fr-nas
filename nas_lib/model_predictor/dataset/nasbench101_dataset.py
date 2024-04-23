import random

import numpy
import torch
from torch_geometric.data import Dataset, Data

from nas_lib.model_predictor.search_space.nasbench101_search_space import OPS


def operation_to_one_hot(ops):
    op_dims = len(OPS)
    output_ops = []

    for op in ops:
        if op in OPS:
            index = OPS[op]
            feature = [0 for _ in range(op_dims)]
            feature[index] = 1
            output_ops.append(feature)
        else:
            raise Exception()

    return output_ops


def matrix_to_edge_index(matrix):
    matrix = numpy.array(matrix)
    shape = matrix.shape
    assert len(shape) == 2 and shape[0] == shape[1]  # must be square
    total_nodes = matrix.shape[0]

    node_out = []
    node_in = []

    for i in range(total_nodes):
        for j in range(total_nodes):
            if matrix[i][j] != 0:
                node_out.append(i)
                node_in.append(j)

    return [node_out, node_in]


class NasBench101Dataset(Dataset):

    input_dim = 6

    def __init__(self, arch_data_list, use_valid_error=True, key_matrix = "matrix", key_ops = "ops", device=None):
        """
        :param arch_data_list:
        :param use_valid_data:
        """

        super().__init__(root=None, transform=None, pre_transform=None, pre_filter=None)

        self.data_list = []

        for data in arch_data_list:
            ops = operation_to_one_hot(data[key_ops])
            edge_index = matrix_to_edge_index(data[key_matrix])
            if use_valid_error:
                label = data["valid_error"]
            else:
                label = data["test_error"]

            ops = torch.from_numpy(numpy.array(ops)).type(torch.float)
            edge_index = torch.from_numpy(numpy.array(edge_index)).type(torch.long)
            label = torch.from_numpy(numpy.array(label)).type(torch.float)

            self.data_list.append(Data(x=ops, edge_index=edge_index, y=label))

        if device:
            self.to_device(device)


    def to_device(self, device):
        for data in self.data_list:
            data.to(device)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


class NasBench101MultiTransformDataset(Dataset):

    input_dim = 6

    def __init__(self, arch_data_list, use_valid_error=True, device=None):
        """
        :param arch_data_list:
        :param use_valid_data:
        """
        super().__init__(root=None, transform=None, pre_transform=None, pre_filter=None )

        self.data_list = []

        for data in arch_data_list:

            matrix_list = data["transform_matrix_list"]
            graph_data_list = []
            for matrix in matrix_list:
                ops = operation_to_one_hot(data["ops"])
                edge_index = matrix_to_edge_index(matrix)

                if use_valid_error:
                    label = data["valid_error"]
                else:
                    label = data["test_error"]

                ops = torch.from_numpy(numpy.array(ops)).type(torch.float)
                edge_index = torch.from_numpy(numpy.array(edge_index)).type(torch.long)
                label = torch.from_numpy(numpy.array(label)).type(torch.float)

                graph_data_list.append(Data(x=ops, edge_index=edge_index, y=label))
            self.data_list.append(graph_data_list)

        if device:
            self.to_device(device)

    def to_device(self, device):
        for data in self.data_list:
            for graph_data in data:
                graph_data.to(device)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]
