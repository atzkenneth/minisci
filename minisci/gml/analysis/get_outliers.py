"""
© 2023, ETH Zurich
"""
from glob import glob

import numpy as np
import torch


def get_outliers(runs):
    diff_dict = {}

    for r in runs:
        # print(r)
        data = torch.load(r, map_location=torch.device("cpu"))
        x, y, rxn_ids = data[1], data[2], data[3]

        for idx, rxn_id in enumerate(rxn_ids):
            diff = abs(x[idx] - y[idx])

            if rxn_id not in diff_dict:
                diff_dict[rxn_id] = [diff]
            else:
                diff_dict[rxn_id] += [diff]

    for rxn in diff_dict:
        avg_diff = np.mean(np.array(diff_dict[rxn]))
        diff_dict[rxn] = avg_diff

    return diff_dict


def get_truth(runs):
    truth_dict = {}

    for r in runs:
        # print(r)
        data = torch.load(r, map_location=torch.device("cpu"))
        x, y, rxn_ids = data[1], data[2], data[3]

        for idx, rxn_id in enumerate(rxn_ids):
            if rxn_id not in truth_dict:
                truth_dict[rxn_id] = x[idx]
            else:
                pass

    return truth_dict


if __name__ == "__main__":
    runs = sorted(glob("../results/config_*"))

    runs_bin = [x for x in runs if "_700" in x]
    diff_dict = get_outliers(runs_bin)
    truth_dict = get_truth(runs_bin)

    for rxn in diff_dict:
        if diff_dict[rxn] >= 0.7:
            print(rxn, "&", int(diff_dict[rxn] * 1000) / 10, "&", int(truth_dict[rxn] * 1000) / 10)
