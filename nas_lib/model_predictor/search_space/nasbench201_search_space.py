import random
from nas_lib.model_predictor.search_space.search_space import DatabaseSearchSpace

from evoxbench.benchmarks.nb201 import NASBench201SearchSpace
from evoxbench.benchmarks.nb201 import NASBench201Evaluator  # the evaluator uses database

import numpy as np

OPS = {
    'input': 0,
    'skip_connect': 1,
    'nor_conv_1x1': 2,
    'nor_conv_3x3': 3,
    'avg_pool_3x3': 4,
    'output': 5,
    'isolate': 6,
}

OP_LIST = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']


NODE_NUM = 8


class NasBench201SearchSpace(DatabaseSearchSpace):
    def __init__(self,
                 determistic=True,
                 dataset='cifar10',  # ['cifar10', 'cifar100', 'ImageNet16-120']
                 ):
        super(NasBench201SearchSpace, self).__init__()

        self.determistic = determistic
        self.search_space = NASBench201SearchSpace()
        self.evaluator = NASBench201Evaluator(objs="err", dataset=dataset)

    def query(self, archs):
        """
        query archs
        :param archs: lists of arch dict with index {'index': [0, 3, 1, 2, 4, 4], ...}
        """
        # query_result = []
        # for i, arch in enumerate(archs):
        #     # TODO: implement this
        #
        #     query_result.append(result)
        # query_result = self._extract_data(data_list=query_result)
        # return query_result
        raise NotImplementedError()

    def sample_data(self, sample_size):
        """
        sample from nasbench-201
        """
        return self.sample_data_uniform(sample_size=sample_size)

    def sample_data_uniform(self, sample_size):
        """
        Uniformly sample from nasbench-201, i.e. choose operations randomly
        Using interface from evoxbench
        """
        print("sampling data in nasbench-201: {}".format(sample_size))
        sample = self.search_space.sample(n_samples=sample_size)
        # sample : ['|none~0|+|skip_connect~0|nor_conv_3x3~1|+|skip_connect~0|skip_connect~1|none~2|', ]

        performance_valid = self.evaluator.evaluate(archs=sample, true_valid=True)
        performance_test = self.evaluator.evaluate(archs=sample, true_eval=True)
        # print(sample)
        # print(performance_valid)
        # print(performance_test)
        data_list = []
        for i in range(len(sample)):
            data_list.append({
                "code": sample[i],
                "index": self.search_space.encode([sample[i]])[0].tolist(),
                "valid_error": performance_valid[i]["err"],
                "test_error": performance_test[i]["err"],
            })

        result = self._extract_data(data_list=data_list)
        return result

    def _extract_data(self, data_list):
        """
        data_list: sample ['|none~0|+|skip_connect~0|nor_conv_3x3~1|+|skip_connect~0|skip_connect~1|none~2|', ]
        """
        output_data = []
        for data in data_list:

            genotype = self._str_to_genotypes(data["code"])

            matrix, ops = self._genotype_to_nodes_edges(genotype)

            output_data.append(
                {
                    "index": data["index"],
                    "matrix": matrix,
                    "ops": ops,
                    "valid_error": data["valid_error"],  # No test error?
                    "test_error": data["test_error"]
                }
            )
        return output_data

    def _genotype_to_nodes_edges(self, genotype_data):
        """
        codes from NPENAS
        """
        ops = ['input']
        data_list = []
        for k in genotype_data:
            data_list.append(k)
        ops.append(data_list[0][0][0])  # 0--->1
        ops.append(data_list[1][0][0])  # 0--->2
        ops.append(data_list[2][0][0])  # 0--->3
        ops.append(data_list[1][1][0])  # 1--->4
        ops.append(data_list[2][1][0])  # 1--->5
        ops.append(data_list[2][2][0])  # 2--->6
        ops.append('output')

        adjacency_matrix = np.zeros((8, 8))
        adjacency_matrix[0, 1] = 1
        adjacency_matrix[0, 2] = 1
        adjacency_matrix[0, 3] = 1
        adjacency_matrix[1, 4] = 1
        adjacency_matrix[1, 5] = 1
        adjacency_matrix[2, 6] = 1
        adjacency_matrix[4, 6] = 1
        adjacency_matrix[3, 7] = 1
        adjacency_matrix[5, 7] = 1
        adjacency_matrix[6, 7] = 1

        del_idxs = [id for id, op in enumerate(ops) if op == 'none']
        ops = [op for op in ops if op != 'none']

        counter = 0
        for id in del_idxs:
            temp_id = id - counter
            adjacency_matrix = np.delete(adjacency_matrix, temp_id, axis=0)
            adjacency_matrix = np.delete(adjacency_matrix, temp_id, axis=1)
            counter += 1
        return adjacency_matrix, ops

    def _str_to_genotypes(self, xstr):
        assert isinstance(xstr, str), 'must take string (not {:}) as input'.format(type(xstr))
        nodestrs = xstr.split('+')
        genotypes = []
        for i, node_str in enumerate(nodestrs):
            inputs = list(filter(lambda x: x != '', node_str.split('|')))
            for xinput in inputs: assert len(xinput.split('~')) == 2, 'invalid input length : {:}'.format(xinput)
            inputs = (xi.split('~') for xi in inputs)
            input_infos = tuple((op, int(IDX)) for (op, IDX) in inputs)
            genotypes.append(input_infos)
        return genotypes


if __name__ == '__main__':
    # query_result = NASBench201Result.objects.order_by('?')[:2]
    serach_space = NasBench201SearchSpace()
    query_result = serach_space.sample_data(5)
    print(query_result)
