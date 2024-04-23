import json
import argparse

import numpy
import numpy as np
import torch

# utils
from experiments.utils import set_random_seed
from experiments.utils import evaluate_stats

# datas
from nas_lib.model_predictor.search_space.nasbench101_search_space import NasBench101SearchSpace
from nas_lib.model_predictor.search_space.nasbench201_search_space import NasBench201SearchSpace
from nas_lib.model_predictor.search_space.darts_search_space import DartsSearchSpace
from nas_lib.model_predictor.dataset.nasbench101_dataset import  NasBench101MultiTransformDataset, NasBench101Dataset
from nas_lib.model_predictor.dataset.nasbench201_dataset import  NasBench201MultiTransformDataset, NasBench201Dataset
from nas_lib.model_predictor.dataset.darts_dataset import  DartsMultiTransformDataset, DartsDataset

# trainer
from nas_lib.model_predictor.trainer.fr_gin_trainer import FRGinPredictorTrainer
from nas_lib.model_predictor.trainer.gin_trainer import GinPredictorTrainer

# algorithms

from nas_lib.model_predictor.agent.multi_transform_gin_predictor import multi_transform

from nas_lib.model_predictor.agent.fr_gin_predictor import FRGinPredictorAgent
from nas_lib.model_predictor.agent.fr_gin_predictor import FRGinPredictorAgentAfter

AvailableMultiTransformDataset = {
    "nasbench-101": NasBench101MultiTransformDataset,
    "nasbench-201": NasBench201MultiTransformDataset,
    "darts": DartsMultiTransformDataset,
}

AvailableDataset = {
    "nasbench-101": NasBench101Dataset,
    "nasbench-201": NasBench201Dataset,
    "darts": DartsDataset,
}

AvailableSearchSpace = {

    "nasbench-101": NasBench101SearchSpace,
    "nasbench-201": NasBench201SearchSpace,
    "darts": DartsSearchSpace,
}
DEVICE = "cpu"


def get_IRG( features):
    dist_matrix_list = []
    for feature in features:
        dist_matrix = torch.norm(feature[:, None] - feature, dim=2, p=2)  # broadcast, calculate A
        dist_matrix_list.append(dist_matrix)
    return dist_matrix_list


def plot(matrix, file_name, vmin=0, vmax=4, cmap="rocket_r", center=None):
    from matplotlib import pyplot as plt
    import seaborn as sns
    from matplotlib.backends.backend_pdf import PdfPages

    plt.rcParams["font.family"] = "serif"

    fig, axs = plt.subplots(1, 1)
    fig.set_figheight(5)
    fig.set_figwidth(6)

    sns.heatmap(data=matrix, vmin=vmin, vmax=vmax, cmap=cmap, center=center)

    # axs.grid(linestyle='--', which='both', linewidth='0.5', color=(0.9, 0.9, 0.9, 0.5))
    axs.set_xlabel('Instance', fontsize=14)
    axs.set_ylabel('Instance', fontsize=14)

    # lg3 = plt.legend(loc='lower right', fontsize=14, numpoints=1)
    # lg3.get_frame().set_linewidth(0.75)
    # lg3.get_frame().set_edgecolor("k")
    # plt.gca().add_artist(lg3)

    with PdfPages('./{}.pdf'.format(file_name)) as pdf:
        # plt.plot(....)
        pdf.savefig(bbox_inches = 'tight')





def FRNAS(search_space, train_archs, test_archs, draw_archs, learning_rate, batch_size, epochs, scaling_factor, device, weight_factor, weight_decay, loss="IRG"):
    for arch in train_archs:
        arch["transform_matrix_list"] = multi_transform(arch["matrix"])

    for arch in test_archs:
        arch["transform_matrix_list"] = multi_transform(arch["matrix"])

    Dataset = AvailableMultiTransformDataset[search_space]

    train_dataset = Dataset(arch_data_list=train_archs, device=device)
    test_dataset = Dataset(arch_data_list=test_archs, device=device)
    draw_dataset = Dataset(arch_data_list=draw_archs, device=device)

    agent_1 = FRGinPredictorAgent(input_dim=Dataset.input_dim)
    agent_2 = FRGinPredictorAgent(input_dim=Dataset.input_dim)

    trainer = FRGinPredictorTrainer(agent_1, agent_2, lr=learning_rate, epochs=epochs, batch_size=batch_size, scaling_factor=scaling_factor,
                                     weight_factor=weight_factor, weight_decay=weight_decay,
                                     device=device, loss=loss)

    trainer.fit(train_dataset)



    # test dataset evaluation

    pred_result, label_result = trainer.test(test_dataset)
    pred_result = pred_result.cpu().numpy()
    label_result = label_result.cpu().numpy()
    # print(pred_result)
    # print(label_result)
    stats = evaluate_stats(pred_result, label_result)

    # draw dataset
    pred_result, label_result = trainer.test(draw_dataset)
    pred_result = pred_result.cpu().numpy()
    label_result = label_result.cpu().numpy()
    error_distribution = numpy.abs(pred_result-label_result)
    stats["error"] = error_distribution
    stats["target"] = label_result
    stats["pred"] = pred_result
    pred_result, label_result = trainer.test(draw_dataset)
    forward_feats = agent_1.features
    reverse_feats = agent_2.features
    forward_feats = get_IRG(forward_feats)
    reverse_feats = get_IRG(reverse_feats)
    forward_feats = forward_feats[0].cpu().numpy()
    reverse_feats = reverse_feats[0].cpu().numpy()

    # print(reverse_feats - forward_feats)

    # forward_reverse
    pred_result_1, pred_result_2, label_result = trainer.test_feature_difference(test_dataset)
    pred_result_1 = pred_result_1.cpu().numpy()
    pred_result_2 = pred_result_2.cpu().numpy()
    label_result = label_result.cpu().numpy()
    stats["forward"] = evaluate_stats(pred_result_1, label_result)
    stats["reverse"] = evaluate_stats(pred_result_2, label_result)
    return forward_feats, reverse_feats, stats

def run(train_archs, test_archs, draw_archs, weight_factor, name):
    stats = FRNAS(search_space=args.search_space,
                     train_archs=train_archs,
                     test_archs=test_archs,
                     draw_archs=draw_archs,
                     learning_rate=0.005, batch_size=16, epochs=300, scaling_factor=20,
                     weight_decay=1e-4,
                     weight_factor=weight_factor,
                     device=DEVICE,
                     loss="IRG")
    plot(stats[0], name+"_f")
    plot(stats[1], name+"_r")
    plot(numpy.abs(stats[0]-stats[1]), name+"_diff")
    print(name, stats[2])


def main(learning_rate, sampled_archs, sample_size, sample_size_2, sample_size_3, test_size):
    data_dict = {}

    # the dataset is fixed, we only do retrains
    train_archs = sampled_archs[:sample_size]
    train_archs_2 = sampled_archs[:sample_size_2]
    train_archs_3 = sampled_archs[:sample_size_3]
    test_archs = sampled_archs[sample_size_3:]
    draw_archs = sampled_archs[:32]

    print("Running")

    # run(train_archs_3, test_archs, draw_archs, 1, "test")

    run(train_archs, test_archs, draw_archs, 0.8, "IRG_feat_50_0.8")
    run(train_archs_2, test_archs, draw_archs, 0.8, "IRG_feat_200_0.8")
    run(train_archs_3, test_archs, draw_archs, 0.8, "IRG_feat_400_0.8")
    run(train_archs, test_archs, draw_archs, 0, "IRG_feat_50_0")
    run(train_archs_2, test_archs, draw_archs, 0, "IRG_feat_200_0")
    run(train_archs_3, test_archs, draw_archs, 0, "IRG_feat_400_0")


    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args for algorithms compare.')
    parser.add_argument('--search_space', type=str, default="nasbench-201", choices=["nasbench-101", "nasbench-201", "darts"], help='search space')
    parser.add_argument('--trials', type=int, default=1, help='Number of trials')
    parser.add_argument('--seed', type=int, default=24657, help='random seed')
    parser.add_argument('--lr', type=float, default=0.005, help='Loss used to train architecture.')
    parser.add_argument('--sample_size', type=int, default=50, help='Number of sampled architectures')
    parser.add_argument('--sample_size_2', type=int, default=200, help='Number of sampled architectures')
    parser.add_argument('--sample_size_3', type=int, default=400, help='Number of sampled architectures')
    parser.add_argument('--test_size', type=int, default=5000, help='Number of test architectures')
    args = parser.parse_args()
    # print(args)
    set_random_seed(args.seed)

    trials_result_dict = {}

    search_space = AvailableSearchSpace[args.search_space]()
    sampled_archs = search_space.sample_data_uniform(args.sample_size_3 + args.test_size)
    for trial in range(args.trials):
        print("trial {}".format(trial))
        result_dict = main(args.lr, sampled_archs, args.sample_size, args.sample_size_2, args.sample_size_3, args.test_size)
        trials_result_dict[trial] = result_dict

