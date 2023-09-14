"""
© 2023, ETH Zurich
"""
import argparse
import configparser
import os

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from minisci.gml.net import GraphTransformer
from minisci.gml.net_utils import DataLSF, get_rxn_ids
from minisci.utils import mae_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    model, optimizer, criterion, train_loader,
):
    model.train()
    training_loss = []

    for g, g2 in train_loader:
        g = g.to(DEVICE)
        g2 = g2.to(DEVICE)
        optimizer.zero_grad()

        pred = model(g, g2)

        loss = criterion(pred, g.rxn_trg)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            mae = mae_loss(pred, g.rxn_trg)
            training_loss.append(mae)

    return np.mean(training_loss)


def eval(
    model, eval_loader,
):
    model.eval()
    eval_loss = []

    preds = []
    ys = []
    rxn_ids = []

    with torch.no_grad():
        for g, g2 in eval_loader:
            g = g.to(DEVICE)
            g2 = g2.to(DEVICE)
            pred = model(g, g2)
            mae = mae_loss(pred, g.rxn_trg)
            eval_loss.append(mae)
            ys.append(g.rxn_trg)
            preds.append(pred)
            rxn_ids += g.rxn_id

    return (
        np.mean(eval_loss),
        ys,
        preds,
        rxn_ids,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", type=str, default="100")
    parser.add_argument("-cv", type=str, default="1")
    parser.add_argument("-testset", type=str, default="1")
    args = parser.parse_args()

    os.makedirs("results/", exist_ok=True)
    os.makedirs("models/", exist_ok=True)

    # Define Configuration form Model and Dataset
    config = configparser.ConfigParser()
    CONFIG_PATH = f"config/config_{args.config}.ini"
    config.read(CONFIG_PATH)
    print({section: dict(config[section]) for section in config.sections()})

    LR_FACTOR = float(config["PARAMS"]["LR_FACTOR"])
    LR_STEP_SIZE = int(config["PARAMS"]["LR_STEP_SIZE"])
    N_KERNELS = int(config["PARAMS"]["N_KERNELS"])
    POOLING_HEADS = int(config["PARAMS"]["POOLING_HEADS"])
    D_MLP = int(config["PARAMS"]["D_MLP"])
    D_KERNEL = int(config["PARAMS"]["D_KERNEL"])
    D_EMBEDDING = int(config["PARAMS"]["D_EMBEDDING"])
    BATCH_SIZE = int(config["PARAMS"]["BATCH_SIZE"])
    SPLIT = str(config["PARAMS"]["SPLIT"])
    ELN = str(config["PARAMS"]["ELN"])
    GEOMETRY = int(config["PARAMS"]["GEOMETRY"])
    TARGET = str(config["PARAMS"]["TARGET"])
    GEOMETRY = True if GEOMETRY >= 1 else False
    GRAPH_DIM = "edge_3d" if GEOMETRY >= 1 else "edge_2d"

    tran_ids, eval_ids, test_ids = get_rxn_ids(split=SPLIT, eln=ELN, testset=args.testset)

    train_data = DataLSF(rxn_ids=tran_ids, graph_dim=GRAPH_DIM, rxn_trg=TARGET)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    test_data = DataLSF(rxn_ids=test_ids, graph_dim=GRAPH_DIM, rxn_trg=TARGET)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model = GraphTransformer(
        n_kernels=N_KERNELS,
        pooling_heads=POOLING_HEADS,
        mlp_dim=D_MLP,
        kernel_dim=D_KERNEL,
        embeddings_dim=D_EMBEDDING,
        geometry=GEOMETRY,
    )
    model = model.to(DEVICE)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_parameters = sum([np.prod(e.size()) for e in model_parameters])
    print("\nmodel_parameters", model_parameters)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR_FACTOR, weight_decay=1e-10)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=0.5, verbose=False)

    tr_losses = []

    for epoch in range(1000):
        tr_l = train(model, optimizer, criterion, train_loader)
        tr_losses.append(tr_l)
        scheduler.step()
        print(f"MAEs (Epoch = {epoch + 1}): {tr_l}")

        if epoch >= 999:
            te_l, te_ys, te_pred, rxn_ids = eval(model, test_loader)
            ys_saved = [float(item) for sublist in te_ys for item in sublist]
            pred_saved = [float(item) for sublist in te_pred for item in sublist]

            print(len(ys_saved), len(pred_saved), len(rxn_ids))

            for idx, x in enumerate(rxn_ids):
                print(x, ys_saved[idx], pred_saved[idx], "diff", abs(pred_saved[idx] - ys_saved[idx]))

            torch.save(
                model.state_dict(), f"models/config_{args.config}_{args.testset}_{args.cv}.pt",
            )

            torch.save(
                [tr_losses, ys_saved, pred_saved, rxn_ids],
                f"results/config_{args.config}_{args.testset}_{args.cv}.pt",
            )
