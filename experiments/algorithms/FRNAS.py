
# utils
from experiments.utils import evaluate_stats

# datas
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



def FRNAS(search_space, train_archs, test_archs, learning_rate, batch_size, epochs, scaling_factor, device, weight_factor, weight_decay, loss="IRG"):
    for arch in train_archs:
        arch["transform_matrix_list"] = multi_transform(arch["matrix"])

    for arch in test_archs:
        arch["transform_matrix_list"] = multi_transform(arch["matrix"])

    Dataset = AvailableMultiTransformDataset[search_space]

    train_dataset = Dataset(arch_data_list=train_archs, device=device)
    test_dataset = Dataset(arch_data_list=test_archs, device=device)

    agent_1 = FRGinPredictorAgent(input_dim=Dataset.input_dim)
    agent_2 = FRGinPredictorAgent(input_dim=Dataset.input_dim)

    trainer = FRGinPredictorTrainer(agent_1, agent_2, lr=learning_rate, epochs=epochs, batch_size=batch_size, scaling_factor=scaling_factor,
                                     weight_factor=weight_factor, weight_decay=weight_decay,
                                     device=device, loss=loss)

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


def FRNAS_finetune(search_space, train_archs, test_archs, learning_rate, batch_size, epochs, scaling_factor, device, weight_factor, weight_decay, loss="IRG"):
    for arch in train_archs:
        arch["transform_matrix_list"] = multi_transform(arch["matrix"])

    for arch in test_archs:
        arch["transform_matrix_list"] = multi_transform(arch["matrix"])

    Dataset = AvailableMultiTransformDataset[search_space]

    train_dataset = Dataset(arch_data_list=train_archs, device=device)
    test_dataset = Dataset(arch_data_list=test_archs, device=device)

    agent_1 = FRGinPredictorAgent(input_dim=Dataset.input_dim)
    agent_2 = FRGinPredictorAgent(input_dim=Dataset.input_dim)

    trainer = FRGinPredictorTrainer(agent_1, agent_2, lr=learning_rate, epochs=epochs, batch_size=batch_size, scaling_factor=scaling_factor,
                                     weight_factor=weight_factor, weight_decay=weight_decay,
                                     device=device, loss=loss)

    trainer.fit(train_dataset)

    # train dataset evaluation
    pred_result, label_result = trainer.test(train_dataset)
    pred_result = pred_result.cpu().numpy()
    label_result = label_result.cpu().numpy()
    stats = evaluate_stats(pred_result, label_result)
    print(stats)

    # finetune after training

    Dataset = AvailableDataset[search_space]

    train_dataset = Dataset(arch_data_list=train_archs)
    test_dataset = Dataset(arch_data_list=test_archs)

    new_agent = FRGinPredictorAgentAfter(agent_1.gin_encoder, agent_2.gin_encoder)
    trainer = GinPredictorTrainer(new_agent, lr=learning_rate, epochs=epochs, batch_size=batch_size, scaling_factor=scaling_factor, weight_decay=weight_decay, device=device)
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