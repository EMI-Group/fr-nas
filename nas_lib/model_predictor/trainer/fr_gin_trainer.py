import torch
import random

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from nas_lib.model_predictor.trainer.utils import make_agent_optimizer, get_lr_scheduler




class IRGLoss(torch.nn.Module):
    """
    A simpler version of IRG loss
    """
    def __init__(self, weight_factor=0.8):
        super(IRGLoss, self).__init__()

        self.logit_loss = torch.nn.MSELoss()
        self.feature_loss =  torch.nn.MSELoss()
        self.weight_factor = weight_factor

    def forward(self, pred, target, pred_features, target_features):
        """
        given a batch of prediction, with a batch of features
        """
        # print(pred_features)
        # print(target_features)

        label_loss = self.logit_loss(pred, target)

        num_features = len(pred_features)

        pred_irg = self.get_IRG(pred_features)
        target_irg = self.get_IRG(target_features)

        feature_loss = 0
        for i in range(0, len(pred_irg)):
            # feature_loss = feature_loss + torch.pow(torch.norm(target_irg[i] - pred_irg[i], p=2), 2) # euclidean distance should be calculated, not l2 norm.
            feature_loss = feature_loss + self.feature_loss(pred_irg[i], target_irg[i])

        # print(label_loss, feature_loss)

        return (1-self.weight_factor) * label_loss + self.weight_factor * feature_loss / num_features

    def get_IRG(self, features):
        dist_matrix_list = []
        for feature in features:
            dist_matrix = torch.norm(feature[:, None] - feature, dim=2, p=2)  # broadcast, calculate A
            dist_matrix_list.append(dist_matrix)
        return dist_matrix_list


class CELoss(torch.nn.Module):
    def __init__(self, weight_factor=0.8):
        super(CELoss, self).__init__()

        self.logit_loss = torch.nn.MSELoss()
        self.feature_loss = torch.nn.CrossEntropyLoss()
        self.weight_factor = weight_factor

    def forward(self, pred, target, pred_features, target_features):
        """
        given a batch of prediction, with a batch of features
        """
        num_features = len(pred_features)

        label_loss = self.logit_loss(pred, target)

        feature_loss = 0
        for i in range(0, len(pred_features)):
            feature_loss = feature_loss + self.feature_loss(pred_features[i], target_features[i])

        return (1-self.weight_factor) * label_loss + self.weight_factor * feature_loss / num_features

class MSELoss(torch.nn.Module):
    def __init__(self, weight_factor=0.8):
        super(MSELoss, self).__init__()

        self.logit_loss = torch.nn.MSELoss()
        self.feature_loss = torch.nn.MSELoss()
        self.weight_factor = weight_factor

    def forward(self, pred, target, pred_features, target_features):
        """
        given a batch of prediction, with a batch of features
        """
        # print(pred_features)
        # print(target_features)

        label_loss = self.logit_loss(pred, target)

        num_features = len(pred_features)

        feature_loss = 0
        for i in range(0, len(pred_features)):
            feature_loss = feature_loss + self.feature_loss(pred_features[i], target_features[i])

        # print(label_loss, feature_loss)

        return (1-self.weight_factor) * label_loss + self.weight_factor * feature_loss / num_features




class FRGinPredictorTrainer:
    def __init__(self,
                 predictor_agent_1,
                 predictor_agent_2,
                 lr=0.01,
                 device=None,
                 epochs=10,
                 batch_size=10,
                 scaling_factor=10,
                 weight_decay=1e-4,
                 bias_multiply=True,
                 loss = "IRG",
                 weight_factor = 0.8
                 ):

        self.predictor_agent_1 = predictor_agent_1
        self.predictor_agent_2 = predictor_agent_2


        if loss=="IRG":
            self.criterion = IRGLoss(weight_factor=weight_factor)
        elif loss=="CE":
            self.criterion = CELoss(weight_factor=weight_factor)
        elif loss=="MSE":
            self.criterion = MSELoss(weight_factor=weight_factor)


        self.device = device
        self.predictor_agent_1.to(self.device)
        self.predictor_agent_2.to(self.device)
        self.lr = lr

        self.batch_size = batch_size
        self.epoch = epochs
        self.scaling_factor = scaling_factor

        self.weight_decay = weight_decay

        self.bias_multiply = bias_multiply

    def fit(self, dataset):
        data_size = len(dataset)
        data_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        self.optimizer_1 = make_agent_optimizer(self.predictor_agent_1, base_lr=self.lr, weight_deacy=self.weight_decay,
                                                bias_multiply=self.bias_multiply)

        self.optimizer_2 = make_agent_optimizer(self.predictor_agent_2, base_lr=self.lr, weight_deacy=self.weight_decay,
                                                bias_multiply=self.bias_multiply)

        self.scheduler_1 = get_lr_scheduler(self.optimizer_1, self.epoch, data_size, self.batch_size)
        self.scheduler_2 = get_lr_scheduler(self.optimizer_2, self.epoch, data_size, self.batch_size)

        self.predictor_agent_1.train()
        self.predictor_agent_2.train()

        for epoch in range(self.epoch):
            num_batch = 0
            for batch in data_loader:
                num_batch += 1
                # for data in batch:
                #     data.to(self.device)
                val_tensor = batch[0].y

                pred_1 = self.predictor_agent_1(batch[0]) * self.scaling_factor
                _ = self.predictor_agent_2(batch[1]) * self.scaling_factor
                pred_1 = pred_1.squeeze()
                loss_1 = self.criterion(pred_1, val_tensor, self.predictor_agent_1.features,
                                        self.predictor_agent_2.features)

                self.optimizer_1.zero_grad()
                loss_1.backward()
                self.optimizer_1.step()

                _ = self.predictor_agent_1(batch[0]) * self.scaling_factor
                pred_2 = self.predictor_agent_2(batch[1]) * self.scaling_factor
                pred_2 = pred_2.squeeze()
                loss_2 = self.criterion(pred_2, val_tensor, self.predictor_agent_2.features,
                                        self.predictor_agent_1.features)

                self.optimizer_2.zero_grad()
                loss_2.backward()
                self.optimizer_2.step()

                self.scheduler_1.step()  # NOTE: the original code uses: self.scheduler.step(epoch + int(i/30))
                self.scheduler_2.step()
            # print(num_batch)

    def pred(self, dataset, batch_size = 64):
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
        pred_list = []

        self.predictor_agent_1.eval()
        self.predictor_agent_2.eval()
        with torch.no_grad():
            for batch in data_loader:

                # for data in batch:
                #     data.to(self.device)

                pred_1 = self.predictor_agent_1(batch[0]) * self.scaling_factor
                pred_2 = self.predictor_agent_2(batch[1]) * self.scaling_factor
                pred_1 = pred_1.squeeze()
                pred_2 = pred_2.squeeze()

                if len(pred_1.size()) == 0:
                    pred_1.unsqueeze(0)
                if len(pred_2.size()) == 0:
                    pred_2.unsqueeze(0)
                # print(pred_1)
                # print(pred_2)
                pred = (pred_1 + pred_2) / 2
                # print(pred)
                pred_list.append(pred)
        return torch.cat(pred_list, dim=0)

    def test(self, dataset, batch_size=64):
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
        pred_list = []
        valid_list = []

        self.predictor_agent_1.eval()
        self.predictor_agent_2.eval()
        with torch.no_grad():
            for batch in data_loader:
                # for data in batch:
                #     data.to(self.device)
                val_tensor = batch[0].y

                pred_1 = self.predictor_agent_1(batch[0]) * self.scaling_factor
                pred_2 = self.predictor_agent_2(batch[1]) * self.scaling_factor
                pred_1 = pred_1.squeeze()
                pred_2 = pred_2.squeeze()

                if len(pred_1.size()) == 0:
                    pred_1.unsqueeze(0)
                if len(pred_2.size()) == 0:
                    pred_2.unsqueeze(0)
                # print(pred_1)
                # print(pred_2)
                pred = (pred_1 + pred_2) / 2
                # print(pred)
                pred_list.append(pred)
                valid_list.append(val_tensor)

        # print(pred_list)
        # print(valid_list)
        return torch.cat(pred_list, dim=0), torch.cat(valid_list, dim=0)


    def test_feature_difference(self, dataset, batch_size=64):

        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
        pred_1_list = []
        pred_2_list = []
        valid_list = []

        self.predictor_agent_1.eval()
        self.predictor_agent_2.eval()
        with torch.no_grad():
            for batch in data_loader:
                # for data in batch:
                #     data.to(self.device)
                val_tensor = batch[0].y

                pred_1 = self.predictor_agent_1(batch[0]) * self.scaling_factor
                pred_2 = self.predictor_agent_2(batch[1]) * self.scaling_factor
                pred_1 = pred_1.squeeze()
                pred_2 = pred_2.squeeze()

                if len(pred_1.size()) == 0:
                    pred_1.unsqueeze(0)
                if len(pred_2.size()) == 0:
                    pred_2.unsqueeze(0)
                pred_1_list.append(pred_1)
                pred_2_list.append(pred_2)
                valid_list.append(val_tensor)

        # print(pred_list)
        # print(valid_list)
        return torch.cat(pred_1_list, dim=0), torch.cat(pred_2_list, dim=0), torch.cat(valid_list, dim=0)

