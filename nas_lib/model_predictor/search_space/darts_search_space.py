import random

import numpy as np

from nas_lib.model_predictor.search_space.search_space import DatabaseSearchSpace
from evoxbench.benchmarks.darts import NASBench301Result

from collections import namedtuple
import json

OPS = {
    'skip_connect': 0,
    'avg_pool_3x3': 1,
    'max_pool_3x3': 2,
    'sep_conv_3x3': 3,
    'sep_conv_5x5': 4,
    'dil_conv_3x3': 5,
    'dil_conv_5x5': 6,
    'normal_input_0': 7,
    'normal_input_1': 8,
    'reduce_input_0': 9,
    'reduce_input_1': 10,
    'add': 11,
    'concat': 12,
}

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
Genotype_cell = namedtuple('CellGenotype', 'cell concat')


class DartsSearchSpace(DatabaseSearchSpace):
    """
    DARTS search space, using prediction result from EvoxBench and NasBench301 database
    Datasets from nasbench301 training data
    """

    def __init__(self, determistic=True, graph_encoding="operation_on_node"):
        super(DartsSearchSpace, self).__init__()
        self.determistic = determistic
        self.graph_encoding = graph_encoding

    def query(self, archs):
        """
        query archs
        :param archs: lists of arch: {"matrix":[square adjacent matrix], "ops": [original operations]}
        """
        raise NotImplementedError()

    def sample_data(self, sample_size, from_database=True):
        if from_database:
            return self.sample_data_nb301_database(sample_size=sample_size)
        else:
            raise NotImplementedError()

    def sample_data_nb301_database(self, sample_size):
        """
        Uniformly sample from nasbench-301 database,
        it samples and evaluates darts model using different sampling methods
        """
        print("sampling data in nasbench-301: {}".format(sample_size))
        query_set = NASBench301Result.objects.order_by('?')[:sample_size]
        # NOTE: this will return a list QuerySet, every time we get the value from same QuerySet will get
        # different result because of random shuffle , must assign value to get exact result.
        query_result = []
        for query in query_set:
            query_result.append(query)

        result = self._extract_data(data_list=query_result)

        return result

    def sample_data_uniform(self, sample_size):
        return self.sample_data_nb301_database(sample_size)

    def _extract_data(self, data_list):
        output_data = []
        for data in data_list:
            index = data.id
            genotype = data.genotype  # the value is a str type,
            genotype = genotype.replace('\'', '\"')  # note eval() method is not allowed
            genotype = json.loads(genotype)
            normal = [tuple(x) for x in genotype[0]]
            normal_concat = genotype[1]
            reduce = [tuple(x) for x in genotype[2]]
            reduce_concat = genotype[3]

            # print(genotype)
            # print(normal)
            # print(normal_concat)
            # print(reduce)
            # print(reduce_concat)

            genotype = Genotype(normal=normal, normal_concat=normal_concat, reduce=reduce, reduce_concat=reduce_concat)

            matrix, ops = self._genotype_to_nodes_edges(genotype)

            valid_acc = data.result["val_accuracy"]
            test_acc = data.result["test_accuracy"]

            valid_error = 100 - valid_acc
            test_error = 100 - test_acc

            output_data.append(
                {
                    "index": index,
                    "matrix": matrix,
                    "ops": ops,
                    "valid_error": valid_error,
                    "test_error": test_error
                }
            )
        return output_data

    def _cell_to_graph(self, cell, cell_concat, cell_type="normal"):
        """
        given cell and cell concat, return the nodes and edge index
        """

        assert cell_type in ["normal", "reduce"]
        state_index_list = [0, 1]
        ops_index_list = []
        ops = ['{}_input_0'.format(cell_type), '{}_input_1'.format(cell_type)]
        edges = []  # list of edges (u,v) : u -> v
        node_num = 2
        op_num = 0
        state_num = 2
        for op, idx in cell:
            ops_index_list.append(node_num)
            ops.append(op)
            edges.append((state_index_list[idx], node_num))
            node_num += 1
            op_num += 1

            # every output of two operator forms a new state
            if op_num % 2 == 0:
                state_index_list.append(node_num)
                ops.append('add')
                edges.append((ops_index_list[-1], node_num))
                edges.append((ops_index_list[-2], node_num))
                node_num += 1
                state_num += 1

        ops.append('concat')
        for idx in cell_concat:
            edges.append((state_index_list[idx], node_num))
        node_num += 1

        return ops, edges

    def _genotype_to_nodes_edges(self, genotype_data):
        # normal cell
        normal_ops, normal_edges = self._cell_to_graph(cell=genotype_data.normal,
                                                       cell_concat=genotype_data.normal_concat,
                                                       cell_type="normal")
        # reduce cell
        reduce_ops, reduce_edges = self._cell_to_graph(cell=genotype_data.reduce,
                                                       cell_concat=genotype_data.reduce_concat,
                                                       cell_type="reduce")
        # merge two cells together
        num_nodes = len(normal_ops)
        new_reduce_edges = []
        for u, v in reduce_edges:
            new_reduce_edges.append((u + num_nodes, v + num_nodes))
        ops = normal_ops + reduce_ops
        edges = normal_edges + new_reduce_edges
        num_nodes = len(ops)
        matrix = np.zeros((num_nodes, num_nodes), dtype=np.int8)
        for u,v in edges:
            matrix[u,v] = 1

        matrix = matrix.tolist()

        return matrix, ops


def _test_database():
    print("total archs: ", len(NASBench301Result.objects.order_by('?')[:]))

    query_set = NASBench301Result.objects.order_by('?')[:2]
    query_result = []
    for query in query_set:
        query_result.append(query)

    for query in query_result:
        print("------")
        print(query)
        print(query.id)
        print(query.phenotype)
        print(query.genotype)
        print(query.normal)
        print(query.normal_concat)
        print(query.reduce)
        print(query.reduce_concat)
        print(query.result)
        print(query.result["val_accuracy"])
        print(query.result["test_accuracy"])
    for query in query_result:
        print("------")
        print(query)
        print(query.id)
        print(query.phenotype)
        print(query.genotype)
        print(query.normal)
        print(query.normal_concat)
        print(query.reduce)
        print(query.reduce_concat)
        print(query.result)
        print(query.result["val_accuracy"])
        print(query.result["test_accuracy"])


def _test():
    darts_search_space = DartsSearchSpace()
    sample = darts_search_space.sample_data(1)
    print(sample)


if __name__ == '__main__':
    _test_database()
    _test()
