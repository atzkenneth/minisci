"""
© 2023, ETH Zurich
"""
import argparse
import configparser
import os
from io import BytesIO

import numpy as np
import pandas as pd
import torch
import xlsxwriter
from net import GraphTransformer
from rdkit import Chem
from rdkit.Chem import Draw
from torch_geometric.data import Data
from tqdm import tqdm

from minisci.preprocess import get_3dG_from_smi
from minisci.utils import cndtns, wash_smiles

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_stratmat_smiles(acids="template_acids", subst="template_substrates"):
    df_ac = pd.read_csv(f"../data/{acids}.tsv", sep=",", encoding="unicode_escape")
    acids_smi = list(df_ac["SMILES"])

    df_su = pd.read_csv(f"../data/{subst}.tsv", sep=",", encoding="unicode_escape")
    subst_smi = list(df_su["SMILES"])

    prd_dict = {}

    for idx_sub, sub in enumerate(subst_smi):
        for idx_acs, acs in enumerate(acids_smi):
            prd_id = f"{subst_smi[idx_sub]}__{acids_smi[idx_acs]}"
            prd_dict[prd_id] = [subst_smi[idx_sub], acids_smi[idx_acs]]

    return prd_dict


def SaveXlsxFromFrame(frame, outFile, molCols=["ROMol"], size=(300, 300)):
    cols = list(frame.columns)

    dataTypes = dict(frame.dtypes)

    workbook = xlsxwriter.Workbook(outFile)  # New workbook
    worksheet = workbook.add_worksheet()  # New work sheet
    worksheet.set_column("A:A", size[0] / 6.0)  # column width

    # Write first row with column names
    c2 = 0
    molCol_names = [f"{x}_img" for x in molCols]
    for x in molCol_names + cols:
        worksheet.write_string(0, c2, x)
        c2 += 1

    c = 1
    for _, row in tqdm(frame.iterrows(), total=len(frame)):
        for k, molCol in enumerate(molCols):
            image_data = BytesIO()
            img = Draw.MolToImage(Chem.MolFromSmiles(row[molCol]), size=size)
            img.save(image_data, format="PNG")

            worksheet.set_row(c, height=size[1])  # looks like height is not in px?
            worksheet.insert_image(c, k, "f", {"image_data": image_data})

        c2 = len(molCols)
        for x in cols:
            if str(dataTypes[x]) == "object":
                # string length is limited in xlsx
                worksheet.write_string(c, c2, str(row[x])[:32000])
            elif ("float" in str(dataTypes[x])) or ("int" in str(dataTypes[x])):
                if (row[x] != np.nan) or (row[x] != np.inf):
                    worksheet.write_number(c, c2, row[x])
            elif "datetime" in str(dataTypes[x]):
                worksheet.write_datetime(c, c2, row[x])
            c2 += 1
        c += 1

    workbook.close()
    image_data.close()


def get_predictions(prd_dict, config_id, model_postfix):
    # Load config
    config = configparser.ConfigParser()
    config.read(f"config/config_{config_id}.ini")
    N_KERNELS = int(config["PARAMS"]["N_KERNELS"])
    D_MLP = int(config["PARAMS"]["D_MLP"])
    D_KERNEL = int(config["PARAMS"]["D_KERNEL"])
    D_EMBEDDING = int(config["PARAMS"]["D_EMBEDDING"])
    POOLING_HEADS = int(config["PARAMS"]["POOLING_HEADS"])
    GEOMETRY = int(config["PARAMS"]["GEOMETRY"])
    GEOMETRY = True if GEOMETRY >= 1 else False
    GRAPH_DIM = "edge_3d" if GEOMETRY >= 1 else "edge_2d"

    # Load model
    model_path = "models/"
    model1 = GraphTransformer(
        n_kernels=N_KERNELS,
        pooling_heads=POOLING_HEADS,
        mlp_dim=D_MLP,
        kernel_dim=D_KERNEL,
        embeddings_dim=D_EMBEDDING,
        geometry=True,
    )
    model1.load_state_dict(
        torch.load(f"{model_path}config_{config_id}_{model_postfix}.pt", map_location=torch.device("cpu"))
    )
    model1 = model1.to(DEVICE)

    rxn_ids = list(prd_dict.keys())

    subs_dict = {}
    acid_dict = {}

    fin_scores = []
    smiles_sub = []
    smiles_acd = []

    # for rxn in tqdm(rxn_ids):
    for rxn in tqdm(rxn_ids):
        try:
            subst_og, acids_og = prd_dict[rxn]
            subst = wash_smiles(subst_og)
            acids = wash_smiles(acids_og)
            # print(subst, acids)

            (
                atom_id_1,
                ring_id_1,
                hybr_id_1,
                arom_id_1,
                edge_2d_1,
                edge_3d_1,
                crds_3d_1,
                to_keep_1,
            ) = get_3dG_from_smi(subst, 0xF00D)

            (
                atom_id_2,
                ring_id_2,
                hybr_id_2,
                arom_id_2,
                edge_2d_2,
                edge_3d_2,
                crds_3d_2,
                to_keep_2,
            ) = get_3dG_from_smi(acids, 0xF00D)

            num_nodes_1 = torch.LongTensor(atom_id_1).size(0)
            num_nodes_2 = torch.LongTensor(atom_id_2).size(0)

            if GRAPH_DIM == "edge_3d":
                edge_index_1 = edge_3d_1
                edge_index_2 = edge_3d_2
            elif GRAPH_DIM == "edge_2d":
                edge_index_1 = edge_2d_1
                edge_index_2 = edge_2d_2

            graph_data = Data(
                atom_id=torch.LongTensor(atom_id_1),
                ring_id=torch.LongTensor(ring_id_1),
                hybr_id=torch.LongTensor(hybr_id_1),
                to_keep=torch.LongTensor(to_keep_1),
                arom_id=torch.LongTensor(arom_id_1),
                condtns=torch.FloatTensor(cndtns),
                crds_3d=torch.FloatTensor(crds_3d_1),
                edge_index=torch.LongTensor(edge_index_1),
                num_nodes=num_nodes_1,
                rea_id=torch.LongTensor(np.array([[0]])),
                so1_id=torch.LongTensor(np.array([[0]])),
                so2_id=torch.LongTensor(np.array([[0]])),
                cat_id=torch.LongTensor(np.array([[0]])),
                add_id=torch.LongTensor(np.array([[0]])),
                atm_id=torch.LongTensor(np.array([[0]])),
            )

            graph_data2 = Data(
                atom_id=torch.LongTensor(atom_id_2),
                ring_id=torch.LongTensor(ring_id_2),
                hybr_id=torch.LongTensor(hybr_id_2),
                arom_id=torch.LongTensor(arom_id_2),
                crds_3d=torch.FloatTensor(crds_3d_2),
                edge_index=torch.LongTensor(edge_index_2),
                num_nodes=num_nodes_2,
            )

            graph_data = graph_data.to(DEVICE)
            graph_data2 = graph_data2.to(DEVICE)

            pred1 = model1(graph_data, graph_data2).detach().cpu().numpy()[0]
            # print(subst_og, acids_og, pred1)

            fin_scores.append(pred1)
            smiles_sub.append(subst_og)
            smiles_acd.append(acids_og)

            if subst_og not in subs_dict:
                subs_dict[subst_og] = pred1
            else:
                subs_dict[subst_og] += pred1

            if acids_og not in acid_dict:
                acid_dict[acids_og] = pred1
            else:
                acid_dict[acids_og] += pred1

        except:
            pass

    df = pd.DataFrame(
        {
            "Smiles_substrate": smiles_sub,
            "Smiles_acid": smiles_acd,
            "Score": fin_scores,
        }
    )

    subs_df = pd.DataFrame([(k, v) for k, v in subs_dict.items()], columns=["SMILES", "Score"])
    acid_df = pd.DataFrame([(k, v) for k, v in acid_dict.items()], columns=["SMILES", "Score"])

    return df, subs_df, acid_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", type=str, default="700")
    parser.add_argument("-model_postfix", type=str, default="1_2")
    parser.add_argument("-acids", type=str, default="template_acids")
    parser.add_argument("-substrates", type=str, default="template_substrates")
    args = parser.parse_args()

    os.makedirs("predictions/", exist_ok=True)

    prd_dict = get_stratmat_smiles(acids=args.acids, subst=args.substrates)
    print(f"Number of products: {len(list(prd_dict.keys()))}")

    df, subs_df, acid_df = get_predictions(prd_dict, args.config, args.model_postfix)

    # Reaction to csv
    df.sort_values(by="Score", ascending=False, inplace=True, ignore_index=True)

    df.to_csv(
        f"predictions/rxn_pred_{args.config}.csv",
        index=False,
    )

    # Substrates to csv
    subs_df.sort_values(by="Score", ascending=False, inplace=True, ignore_index=True)

    subs_df.to_csv(
        f"predictions/rxn_substrates_scoring_{args.config}.csv",
        index=False,
    )

    # Acids to csv
    acid_df.sort_values(by="Score", ascending=False, inplace=True, ignore_index=True)

    acid_df.to_csv(
        f"predictions/rxn_acids_scoring_{args.config}.csv",
        index=False,
    )

    # Visualize molecules for substrates and acids
    SaveXlsxFromFrame(
        subs_df,
        f"predictions/rxn_substrate_scoring_{args.config}.xlsx",
        molCols=[
            "SMILES",
        ],
        size=(300, 300),
    )

    SaveXlsxFromFrame(
        acid_df,
        f"predictions/rxn_acids_scoring_{args.config}.xlsx",
        molCols=[
            "SMILES",
        ],
        size=(300, 300),
    )
