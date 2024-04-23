
# utils
from experiments.utils import evaluate_stats

# trainer
from nas_lib.model_predictor.trainer.gin_trainer import GinPredictorTrainer

# algorithms
from nas_lib.model_predictor.agent.npnas_gcn_predictor import NPNASGcnPredictorAgent

from nas_lib.model_predictor.agent.multi_transform_gin_predictor import multi_transform, undirected_transform, \
    no_transform, reverse_transform


from nas_lib.model_predictor.search_space.nasbench101_search_space import NasBench101SearchSpace
from nas_lib.model_predictor.dataset.nasbench101_dataset import NasBench101Dataset
from nas_lib.model_predictor.dataset.nasbench201_dataset import NasBench201Dataset
from nas_lib.model_predictor.dataset.darts_dataset import DartsDataset

AvailableDataset = {
    "nasbench-101": NasBench101Dataset,
    "nasbench-201": NasBench201Dataset,
    "darts": DartsDataset,
}




def NPNAS(search_space, train_archs, test_archs, learning_rate, batch_size, epochs, scaling_factor, weight_decay, bias_multiply, device):

    Dataset = AvailableDataset[search_space]

    train_dataset = Dataset(arch_data_list=train_archs)
    test_dataset = Dataset(arch_data_list=test_archs)

    agent = NPNASGcnPredictorAgent(input_dim=Dataset.input_dim, directed_gcn=True)
    trainer = GinPredictorTrainer(agent, lr=learning_rate, epochs=epochs, batch_size=batch_size, scaling_factor=scaling_factor, device=device,
                                  bias_multiply=bias_multiply, weight_decay = weight_decay
                                  )
    trainer.fit(train_dataset)

    # train dataset evaluation
    pred_result, label_result = trainer.test(train_dataset)
    pred_result = pred_result.cpu().numpy()
    label_result = label_result.cpu().numpy()

    stats = evaluate_stats(pred_result, label_result)
    print(stats)

    # test dataset evaluation
    pred_result, label_result = trainer.test(test_dataset)

    pred_result = pred_result.cpu().numpy()
    label_result = label_result.cpu().numpy()
    # print(pred_result)
    # print(label_result)
    stats = evaluate_stats(pred_result, label_result)
    print(stats)
    return stats
