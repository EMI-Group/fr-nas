# Copyright (c) XiDian University and Xi'an University of Posts&Telecommunication. All Rights Reserved

import torch
import random

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from nas_lib.model_predictor.trainer.utils import make_agent_optimizer, get_lr_scheduler


class RankingLoss(torch.nn.Module):
    def __init__(self, margin=0.1):
        super(RankingLoss, self).__init__()
        self.margin = margin
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, pred, target):
        loss = torch.tensor(0)
        for i in range(len(target)):
            Tj = target < target[i]
            si = pred[i]
            sj = pred[Tj]
            if len(sj) == 0:
                continue
            loss = loss + torch.sum(torch.maximum(torch.tensor(0), self.margin - (si - sj))) / len(sj)
        loss = loss / len(pred)
        return loss


class GinPredictorTrainer:
    def __init__(self,
                 predictor_agent,
                 lr=0.01,
                 device=None,
                 epochs=10,
                 batch_size=10,
                 scaling_factor=10,
                 weight_decay=1e-4,
                 bias_multiply=True,
                 lr_curve="CosineAnnealingLR",
                 loss="MSE"
                 ):

        self.predictor_agent = predictor_agent

        if loss == "MSE":
            self.criterion = torch.nn.MSELoss()
        elif loss == "Ranking":
            self.criterion = RankingLoss()
        else:
            raise NotImplementedError()

        self.device = device
        self.predictor_agent.to(self.device)
        self.lr = lr

        self.batch_size = batch_size
        self.epoch = epochs
        self.scaling_factor = scaling_factor

        self.weight_decay = weight_decay

        self.bias_multiply = bias_multiply

        self.lr_curve = lr_curve

    def fit(self, dataset):
        data_size = len(dataset)
        data_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        self.optimizer = make_agent_optimizer(self.predictor_agent, base_lr=self.lr, weight_deacy=self.weight_decay,
                                              bias_multiply=self.bias_multiply)

        self.scheduler = get_lr_scheduler(self.optimizer, self.epoch, data_size, self.batch_size, lr=self.lr_curve)

        self.predictor_agent.train()

        for epoch in range(self.epoch):
            num_batch = 0
            for batch in data_loader:
                num_batch += 1
                if isinstance(batch, list):
                    for data in batch:
                        data.to(self.device)
                    val_tensor = batch[0].y
                else:
                    batch = batch.to(self.device)
                    val_tensor = batch.y

                pred = self.predictor_agent(batch) * self.scaling_factor
                pred = pred.squeeze()
                loss = self.criterion(pred, val_tensor)
                # print(loss)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.scheduler.step()  # NOTE: the original code uses: self.scheduler.step(epoch + int(i/30))
            # print(num_batch)

    def pred(self, dataset, batch_size = 64):
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
        pred_list = []

        self.predictor_agent.eval()
        with torch.no_grad():
            for batch in data_loader:

                if isinstance(batch, list):
                    for data in batch:
                        data.to(self.device)
                else:
                    batch = batch.to(self.device)
                pred = self.predictor_agent(batch) * self.scaling_factor
                pred = pred.squeeze()

                if len(pred.size()) == 0:
                    pred.unsqueeze(0)
                pred_list.append(pred)
        return torch.cat(pred_list, dim=0)

    def test(self, dataset, batch_size = 64):
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
        pred_list = []
        valid_list = []
        self.predictor_agent.eval()
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, list):
                    for data in batch:
                        data.to(self.device)
                    val_tensor = batch[0].y
                else:
                    batch = batch.to(self.device)
                    val_tensor = batch.y
                pred = self.predictor_agent(batch) * self.scaling_factor
                pred = pred.squeeze()

                if len(pred.size()) == 0:
                    pred.unsqueeze(0)
                pred_list.append(pred)
                valid_list.append(val_tensor)

        # print(pred_list)
        # print(valid_list)
        return torch.cat(pred_list, dim=0), torch.cat(valid_list, dim=0)


def test_ranking_loss():
    y = torch.tensor([2,1,5,4,3])
    x = torch.tensor([1,2,3,4,5])
    loss = RankingLoss()
    print(loss(y,x))

if __name__ == '__main__':
    test_ranking_loss()