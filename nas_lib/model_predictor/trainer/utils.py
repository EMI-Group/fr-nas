import torch
import math
import numpy as np

from torch.optim.lr_scheduler import CosineAnnealingLR, ConstantLR


def get_lr_scheduler(optimizer, epochs, train_date_size, batch_size, lr="CosineAnnealingLR"):
    total_steps = epochs*int(train_date_size / batch_size)
    if lr=="CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=total_steps, eta_min=0)
    elif lr=="Constant":
        scheduler = ConstantLR(optimizer=optimizer)
    else:
        raise NotImplementedError()
    return scheduler



def make_agent_optimizer(model, base_lr, weight_deacy=1e-4, bias_multiply=True):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = base_lr
        weight_decay = weight_deacy
        if "bias" in key:
            if bias_multiply:
                lr = base_lr*2.0
                weight_decay = 0.0
            else:
                lr = base_lr
                weight_decay = weight_deacy

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    optimizer = torch.optim.Adam(params, base_lr, (0.0, 0.9))
    return optimizer






if __name__ == '__main__':
    pass
