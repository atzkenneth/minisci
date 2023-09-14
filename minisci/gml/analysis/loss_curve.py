"""
© 2023, ETH Zurich
"""
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch


def get_lr(p, name):
    plt.figure(figsize=(10, 8))

    data = torch.load(p, map_location=torch.device("cpu"))
    tr = data[0]

    epochs = np.arange(0, len(tr))

    plt.plot(
        epochs,
        tr,
        "royalblue",
        label="Training Loss \nAvg. last 10 epochs = " + str(np.round(np.mean(tr[-10:]), 4)),
    )

    print(name, np.round(np.mean(tr[-10:]), 4))

    plt.legend(loc="best", fontsize=18)
    plt.tick_params(axis="x", labelsize=16)
    plt.tick_params(axis="y", labelsize=16)
    plt.ylabel("Average loss per epoch", fontsize=16)
    plt.xlabel("Epochs", fontsize=16)
    os.makedirs(os.path.dirname(name), exist_ok=True)
    plt.savefig(name, dpi=400)
    plt.clf()


if __name__ == "__main__":
    runs = sorted(glob("../results/config_*"))
    print("Number of files: ", len(runs))
    runs = [x.split("/")[-1] for x in runs]
    os.makedirs("plots/", exist_ok=True)

    for i, run in enumerate(runs):
        get_lr(p=f"../results/{run}", name=f"plots/learnings/learning_{run[:-3]}.png")
