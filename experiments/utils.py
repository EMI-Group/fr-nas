import random

import numpy
import numpy as np
import torch

from scipy.stats import kendalltau
from scipy.stats import spearmanr

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_stats(x: np.ndarray, y: np.ndarray, metrics=None, prefix=""):
    if metrics is None:
        metrics = ["MAE", "MSE", "kendalltau", "spearman"]

    result = {}
    for metric in metrics:
        if metric == "MAE":
            result[prefix + metric] = calculate_mean_error(x, y)
        if metric == "MSE":
            result[prefix + metric] = calculate_mean_square_error(x, y)
        if metric == "kendalltau":
            result[prefix + metric] = calclulate_kendalltau_correlation(x, y)
        if metric == "spearman":
            result[prefix + metric] = calclulate_spearman_rank_correlation(x, y)

    return result


def calculate_mean_error(x: np.ndarray, y: np.ndarray):
    """
    input two numpy list
    retrun mean error
    """
    result = np.mean(np.abs(x - y)).tolist()
    return result


def calculate_mean_square_error(x: np.ndarray, y: np.ndarray):
    """
    input two numpy list
    retrun mean square error
    """
    result = np.mean((x - y) ** 2).tolist()
    return result


def calclulate_kendalltau_correlation(x: np.ndarray, y: np.ndarray):
    """
    input two numpy list
    retrun kendall tau correlation, pvalue
    """
    correlation, pvalue = kendalltau(x, y)
    return correlation, pvalue

def calclulate_spearman_rank_correlation(x: np.ndarray, y: np.ndarray):
    """
    input two numpy list
    retrun kendall tau correlation, pvalue
    """
    correlation, pvalue = spearmanr(x, y)
    return correlation, pvalue


def test():
    x = numpy.array([1,2,3,4,5])
    y = numpy.array([6,2,3,2,5])
    res = evaluate_stats(x,y)
    print(res)

if __name__ == '__main__':
    test()