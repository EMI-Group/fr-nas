
# utils
from experiments.utils import evaluate_stats

# trainer
from nas_lib.model_predictor.trainer.gin_trainer import GinPredictorTrainer

# algorithms
from nas_lib.model_predictor.agent.npenas_gin_predictor import NPENASGinPredictorAgent
from nas_lib.model_predictor.agent.fr_gin_predictor import DirectedGinPredictorAgent

from nas_lib.model_predictor.agent.multi_transform_gin_predictor import MultiTransformGinPredictorAgent
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


def NPENAS(search_space, train_archs, test_archs, learning_rate, batch_size, epochs, scaling_factor, weight_decay, device):

    Dataset = AvailableDataset[search_space]

    train_dataset = Dataset(arch_data_list=train_archs)
    test_dataset = Dataset(arch_data_list=test_archs)

    agent = NPENASGinPredictorAgent(input_dim=Dataset.input_dim)
    trainer = GinPredictorTrainer(agent, lr=learning_rate, epochs=epochs, batch_size=batch_size, scaling_factor=scaling_factor, weight_decay=weight_decay, device=device)
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
    stats = evaluate_stats(pred_result, label_result)
    print(stats)
    return stats



def NPENAS_ensemble_transform(search_space, train_archs, test_archs, learning_rate, batch_size, epochs, scaling_factor, weight_decay, device, reverse=True):
    for arch in train_archs:
        arch["matrix_forward"] = no_transform(arch["matrix"])
        if reverse:
            arch["matrix_backward"] = reverse_transform(arch["matrix"])
        else:
            arch["matrix_backward"] = no_transform(arch["matrix"])

    for arch in test_archs:
        arch["matrix_forward"] = no_transform(arch["matrix"])
        if reverse:
            arch["matrix_backward"] = reverse_transform(arch["matrix"])
        else:
            arch["matrix_backward"] = no_transform(arch["matrix"])

    Dataset = AvailableDataset[search_space]

    train_dataset_forward = Dataset(arch_data_list=train_archs, key_matrix="matrix_forward")
    test_dataset_forward = Dataset(arch_data_list=test_archs, key_matrix="matrix_forward")

    train_dataset_backward = Dataset(arch_data_list=train_archs, key_matrix="matrix_backward")
    test_dataset_backward = Dataset(arch_data_list=test_archs, key_matrix="matrix_backward")

    agent_forward = NPENASGinPredictorAgent(input_dim=Dataset.input_dim)
    trainer_forward = GinPredictorTrainer(agent_forward, lr=learning_rate, epochs=epochs, batch_size=batch_size, scaling_factor=scaling_factor, weight_decay=weight_decay, device=device)
    trainer_forward.fit(train_dataset_forward)

    agent_backward = NPENASGinPredictorAgent(input_dim=Dataset.input_dim)
    trainer_backward = GinPredictorTrainer(agent_backward, lr=learning_rate, epochs=epochs, batch_size=batch_size, scaling_factor=scaling_factor, weight_decay=weight_decay, device=device)
    trainer_backward.fit(train_dataset_backward)

    pred_result_forward, label_result_forward = trainer_forward.test(test_dataset_forward)
    pred_result_backward, label_result_backward = trainer_backward.test(test_dataset_backward)

    if not label_result_forward.equal(label_result_backward):
        raise Exception("disorder!")

    pred_result_forward = pred_result_forward.cpu().numpy()
    pred_result_backward = pred_result_backward.cpu().numpy()
    pred_result = (pred_result_forward + pred_result_backward) / 2

    label_result = label_result_forward.cpu().numpy()

    stats = evaluate_stats(pred_result, label_result)
    print(stats)
    return stats


def NPENAS_Directed(search_space, train_archs, test_archs, learning_rate, batch_size, epochs, scaling_factor, weight_decay, device):

    Dataset = AvailableDataset[search_space]

    train_dataset = Dataset(arch_data_list=train_archs)
    test_dataset = Dataset(arch_data_list=test_archs)

    agent = DirectedGinPredictorAgent(input_dim=Dataset.input_dim)
    trainer = GinPredictorTrainer(agent, lr=learning_rate, epochs=epochs, batch_size=batch_size, scaling_factor=scaling_factor, weight_decay=weight_decay, device=device)
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
    stats = evaluate_stats(pred_result, label_result)
    print(stats)
    return stats