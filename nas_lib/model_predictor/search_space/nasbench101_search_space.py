import random
from nas_lib.model_predictor.search_space.search_space import DatabaseSearchSpace
from evoxbench.benchmarks.nb101 import NASBench101Result
from evoxbench.benchmarks.nb101 import NASBench101Graph

OPS = {
    'input': 0,
    'conv3x3-bn-relu': 1,
    'conv1x1-bn-relu': 2,
    'maxpool3x3': 3,
    'output': 4,
    'isolate': 5
}

NODE_NUM = 7


class NasBench101SearchSpace(DatabaseSearchSpace):
    def __init__(self, determistic=True):
        super(NasBench101SearchSpace, self).__init__()
        self.determistic = determistic

    def _extract_data(self, data_list):
        """
        given data list from database
        """
        output_data = []
        for data in data_list:
            index = data["index"]
            matrix = data["phenotype"]["module_adjacency"]
            ops = data["phenotype"]["module_operations"]
            valid_acc_list = data["final_validation_accuracy"]["epoch108"]
            test_acc_list = data["final_test_accuracy"]["epoch108"]

            if self.determistic:
                valid_error = (1 - sum(valid_acc_list) / len(valid_acc_list)) * 100
                test_error = (1 - sum(test_acc_list) / len(test_acc_list)) * 100
            else:
                valid_error = (1 - random.choice(valid_acc_list)) * 100
                test_error = (1 - random.choice(test_acc_list)) * 100

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

    def query(self, archs):
        """
        query archs
        :param archs: lists of arch: {"matrix":[square adjacent matrix], "ops": [original operations]}
        """
        query_result = []
        for i, arch in enumerate(archs):
            model_spec = NASBench101Graph(matrix=arch['matrix'], ops=arch['ops'])
            index = model_spec.hash_spec(['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3'])  # hash value of one arch
            result = NASBench101Result.objects.values("index", "phenotype", "final_test_accuracy",
                                                      "final_validation_accuracy").get(index=index)
            query_result.append(result)
        query_result = self._extract_data(data_list=query_result)
        return query_result

    def sample_data(self, sample_size):
        return self.sample_data_uniform(sample_size=sample_size)

    def sample_data_uniform(self, sample_size):
        """
        Uniformly sample from nasbench-101
        """
        print("sampling data in nasbench-101: {}".format(sample_size))
        query_set = NASBench101Result.objects.order_by('?') \
                           .values("index", "phenotype", "final_test_accuracy", "final_validation_accuracy")[
                       :sample_size]
        # NOTE: this will return a list QuerySet, every time we get the value from same QuerySet will get
        # different result because of random shuffle , must assign value to get exact result.
        query_result = []
        for query in query_set:
            query_result.append(query)


        result = self._extract_data(data_list=query_result)
        return result


if __name__ == '__main__':
    nb101data = NasBench101SearchSpace()
    sample = nb101data.sample_data_uniform(10)
    query_archs = []
    for s in sample:
        print(s)
        query_archs.append({"matrix": s["matrix"], "ops": s["ops"]})
    print("-" * 20)

    result = nb101data.query(query_archs)
    for s in result:
        print(s)
        query_archs.append({"matrix": s["matrix"], "ops": s["ops"]})
    print("-" * 20)
