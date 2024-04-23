import json
import argparse
import numpy as np
import os

from datetime import datetime

# utils
from experiments.utils import set_random_seed

from experiments.utils import evaluate_stats

# search space
from nas_lib.model_predictor.search_space.nasbench101_search_space import NasBench101SearchSpace
from nas_lib.model_predictor.search_space.nasbench201_search_space import NasBench201SearchSpace
from nas_lib.model_predictor.search_space.darts_search_space import DartsSearchSpace

# algorithms
# from experiments.algorithms.NPENAS_BKD import NPENAS_BKD
from experiments.algorithms.FRNAS import FRNAS

WeightFactorList = [
    0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
]

AvailableSearchSpace = {
    "nasbench-101": NasBench101SearchSpace,
    "nasbench-201": NasBench201SearchSpace,
    "darts": DartsSearchSpace,
}

DEVICE = "cpu"  # the bottleneck of this type of predictor is cpu and memory


def main(args):
    sample_size, test_size = args.sample_size, args.test_size
    data_dict = {}

    # search space

    search_space = AvailableSearchSpace[args.search_space]()
    sampled_archs = search_space.sample_data_uniform(sample_size + test_size)
    train_archs = sampled_archs[:sample_size]
    test_archs = sampled_archs[sample_size:]

    for weight_factor in WeightFactorList:
        for loss in ["MSE",  "IRG"]:
            algorithm = "FRNAS_{}_wf{}".format(loss, weight_factor)
            print("Running algorithm: {}".format(algorithm))
            if loss == "IRG":
                algo_result = FRNAS(search_space=args.search_space,
                                         train_archs=train_archs,
                                         test_archs=test_archs,
                                         learning_rate=0.005, batch_size=16, epochs=200, scaling_factor=20,
                                         weight_decay=1e-4,
                                         weight_factor=weight_factor,
                                         device=DEVICE,
                                         loss=loss)
            elif loss == "MSE":
                algo_result = FRNAS(search_space=args.search_space,
                                         train_archs=train_archs,
                                         test_archs=test_archs,
                                         learning_rate=0.005, batch_size=16, epochs=200, scaling_factor=20,
                                         weight_decay=1e-4,
                                         weight_factor=weight_factor,
                                         device=DEVICE,
                                         loss=loss)

            data_dict[algorithm] = algo_result

    return data_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args for algorithms compare.')
    parser.add_argument('--search_space', type=str, default="darts", choices=["nasbench-101", "nasbench-201", "darts"],
                        help='search space')
    parser.add_argument('--trials', type=int, default=100, help='Number of trials')
    parser.add_argument('--seed', type=int, default=15234, help='random seed')
    parser.add_argument('--save_path', type=str, default='./result/', help='path to save')
    parser.add_argument('--save_prefix', type=str, default='weight_factor_new', help='prefix of the file')

    parser.add_argument('--sample_size', type=int, default=200, help='Number of sampled architectures')

    parser.add_argument('--test_size', type=int, default=5000, help='Number of test architectures')
    parser.add_argument('--save_interval', type=int, default=30, help='intervals to save')
    args = parser.parse_args()
    # print(args)
    set_random_seed(args.seed)

    init_time = datetime.now()
    init_time = init_time.strftime("%y%m%d") + "-" + init_time.strftime("%H%M%S")


    save_path = args.save_path + args.save_prefix + "_" + init_time + "_" + args.search_space
    os.mkdir(save_path)

    config = {
        "seed": args.seed,
        "sample_size": args.sample_size,
        "trials": args.trials,
        "test_size": args.test_size,
        "search_space": args.search_space,
        "epochs": 200,
    }

    with open(save_path + "/config.json", "w") as f:
        json.dump(config, f, indent=2)

    trials_result_dict = {}
    for trial in range(args.trials):
        print("trial {}".format(trial))
        result_dict = main(args)
        trials_result_dict[trial] = result_dict
        if (trial + 1) % args.save_interval == 0:
            with open(save_path + "/ckpt-{}.json".format(trial+1), "w") as f:
                json.dump(trials_result_dict, f, indent=2)


    with open(save_path + "/results.json", "w") as f:
        json.dump(trials_result_dict, f, indent=2)
