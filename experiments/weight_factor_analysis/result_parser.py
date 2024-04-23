import json
import numpy as np
import argparse
import scipy.stats as stats


def merge_data(total_trial, trial):
    # data structures of every trials shall be same
    for key in trial.keys():
        stats = trial[key]
        if key in total_trial:
            total_trial[key].append(stats)
        else:
            total_trial[key] = [stats]


def merge_stats(total_stats, stats):
    # data structures of every stats shall be same
    for metric in stats.keys():
        s = stats[metric]
        if metric == "kendalltau":
            s = s[0]
        total_stats[metric].append(s)


def merge_trials(total_trials, trial):
    # data structures of every trial shall be same
    for algoname in trial.keys():
        stats = trial[algoname]
        if algoname in total_trials:
            merge_stats(total_trials[algoname], stats)
        else:
            new_stats = {}
            for metric in stats.keys():
                new_stats[metric] = []
            total_trials[algoname] = new_stats
            merge_stats(total_trials[algoname], stats)


def load_from_file(file_path):
    with open(file_path+"/results.json", 'r') as f:
        data = json.load(f)

    with open(file_path + "/config.json", 'r') as f:
        config = json.load(f)

    total_trials = {}
    num_trial = 0

    for k in data.keys():
        if not k.isdigit():
            raise NotImplementedError("such file is not supported")
        trial = data[k]
        merge_trials(total_trials, trial)
        num_trial += 1

    # assert num_trial == config["trials"]

    # print(json.dumps(total_trials, indent=2))
    return total_trials, num_trial, config


def load_and_evaluate(file_path, metrics=None):
    if metrics is None:
        metrics = [
            "MAE",
            "kendalltau"
        ]

    # load_old_file(SAVE_PATH)
    data, num_trial, config = load_from_file(file_path)

    result = {}
    for algoname in data.keys():
        algo_stats = data[algoname]
        result[algoname] = {}
        for metric in algo_stats.keys():
            if not metric in metrics:
                continue
            s = np.array(algo_stats[metric])
            result[algoname][metric] = (np.nanmean(s, axis=0).tolist(), np.nanstd(s, axis=0).tolist())

    return result, num_trial, config


def table(result_dict, algo_names):
    metric = "kendalltau"

    for algo in algo_names:
        result = []
        for sample_size in result_dict.keys():
            mean = result_dict[sample_size][algo][metric][0]
            std = result_dict[sample_size][algo][metric][1]
            mean = round(mean, 4)
            std = round(std, 4)
            result.append((mean,std))
            # result = result + "  &  " + str(mean) + " $\\pm$ " + str(std)
        print(algo, result)



def main():
    experiment_path_list = [
        "./result/weight_factor_new_#Replace With Date#_darts",
    ]

    algo_name = []

    result_dict = {}  # sample_size --> trail_result
    for file_path in experiment_path_list:
        result, num_trial, config = load_and_evaluate(file_path)
        for algo in result.keys():
            algo_name.append(algo)
        print(num_trial)
        if config["sample_size"] in result_dict:
            result_dict[config["sample_size"]].update(result)
        else:
            result_dict[config["sample_size"]] = result

    print(json.dumps(result_dict, indent=2))

    table(result_dict, algo_name)



if __name__ == '__main__':
    main()

    # print(data)
    # print(stats.ranksums(data["neural_predictor"], data["neural_predictor_reverse_only"]))
