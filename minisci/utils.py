"""
© 2023, ETH Zurich
"""
import random
from collections import Counter

import numpy as np
import torch.nn.functional as F
from rdkit import Chem

DATASET_NAME = "20230830_minisci_pub_all_data"

mae_loss = lambda x, y: F.l1_loss(x, y).item()

cndtns = np.array([[6.00e00, 2.00e01, 1.60e01, 4.00e01, 1.80e01, 5.00e-01, 0.00e00, 0.00e00, 9.68e-01, 3.20e-02]])


def get_dict_for_embedding(list):
    """Creates a dict x: 0 - x: N for x in List with len(list(set(List))) = N"""
    list_dict = {}
    list_counter = Counter(list)
    for idx, x in enumerate(list_counter):
        list_dict[x] = idx
    return list_dict


HYBRIDISATIONS = [
    "SP3",
    "SP2",
    "SP",
    "UNSPECIFIED",
    "S",
]
AROMATOCITY = [
    "True",
    "False",
]
IS_RING = [
    "True",
    "False",
]
ATOMTYPES = [
    "H",
    "C",
    "N",
    "O",
    "F",
    "P",
    "S",
    "Cl",
    "Br",
    "I",
]

HYBRIDISATION_DICT = get_dict_for_embedding(HYBRIDISATIONS)
AROMATOCITY_DICT = get_dict_for_embedding(AROMATOCITY)
IS_RING_DICT = get_dict_for_embedding(IS_RING)
ATOMTYPE_DICT = get_dict_for_embedding(ATOMTYPES)


def neutralize_atoms(mol):
    """Neutralizes a given SMILES-string.

    :param mol: RDKit molecule object
    :type mol: rdkit.Chem.rdchem.Mol
    :return: RDKit molecule object
    :rtype: rdkit.Chem.rdchem.Mol
    """
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
    return mol


def wash_smiles(smiles):
    """_summary_

    Args:
        smiles (_type_): _description_

    Returns:
        _type_: _description_
    """
    smiles_list = smiles.split(".")
    longest_smiles = max(smiles_list, key=len)
    mol = Chem.MolFromSmiles(longest_smiles)
    neutralize_atoms(mol)
    neutr_ranodm_smiles = randomSmiles(mol)

    return neutr_ranodm_smiles


def randomSmiles(m1):
    m1.SetProp("_canonicalRankingNumbers", "True")
    idxs = list(range(0, m1.GetNumAtoms()))
    random.shuffle(idxs)
    for i, v in enumerate(idxs):
        m1.GetAtomWithIdx(i).SetProp("_canonicalRankingNumber", str(v))
    return Chem.MolToSmiles(m1)
