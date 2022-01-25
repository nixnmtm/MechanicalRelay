from __future__ import print_function
import numpy as np
import logging
import pandas as pd

aa_3to1map = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K', 'ILE': 'I', 'PRO': 'P',
              'THR': 'T', 'PHE': 'F', 'ASN': 'N',  'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R',
              'TRP': 'W', 'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

aa = {"describe": "hydrophobic residues",
      "hpho": ['ALA', 'VAL', 'LEU', 'ILE', 'PHE', 'PRO', 'TRP', 'MET']}


def parse_pdb(pdb_handle):
    """
    Parses the PDB file as a pandas DataFrame object.

    Backbone chain atoms are ignored for the calculation
    of interacting residues.
    """
    atomic_data = []
    with open(pdb_handle, "r") as f:
        for line in f.readlines():
            data = dict()
            if line[0:4] == "ATOM":
                data["Record name"] = line[0:5].strip(" ")
                data["serial_number"] = int(line[6:11].strip(" "))
                data["atom"] = line[12:15].strip(" ")
                data["resi_name"] = line[17:20]
                data["chain_id"] = line[21]
                data["resid"] = line[23:27].strip(" ")
                data["x"] = float(line[30:37])
                data["y"] = float(line[38:45])
                data["z"] = float(line[46:53])
                atomic_data.append(data)
    atomic_df = pd.DataFrame(atomic_data)
    atomic_df["node_id"] = (
            atomic_df["chain_id"]
            + atomic_df["resid"].map(str)
            + atomic_df["resi_name"]
    )

    ### Add continuous resnumber(resno)
    dfs = []
    for i in np.unique(atomic_df["chain_id"]):
        tmp = atomic_df[atomic_df["chain_id"] == i]
        resnum = range(1, len(np.unique(tmp["resid"])) + 1)
        resids = tmp.resid.unique()
        zipped = dict(list(zip(resids, resnum)))
        tmp['resid'].replace(zipped, inplace=True)
        dfs.append(tmp)
    tmp = pd.concat(dfs, axis=0)
    atomic_df["resno"] = tmp["resid"].astype(str)
    return atomic_df


def get_pdb_resid(pdb_path):
    """Get the pdb resid of the given pdb file"""
    pdb_data = parse_pdb(pdb_path)
    return pdb_data.resid.unique()


def get_pdb_aaresid(pdb_path):
    """Get the resid with single letter amino acid codon
        eg: I16
    """
    pdb_data = parse_pdb(pdb_path)
    pdb_aa = pd.DataFrame()
    pdb_aa["aa_1"] = [aa_3to1map[i] for i in pdb_data.resi_name.values]
    pdb_aa["pdb_aa"] = pdb_aa["aa_1"] + pdb_data.resid

    return pdb_aa.pdb_aa.unique()


def get_submat(df, rescut, colcut):
    """
    Get a sub table from a full Kb table

    :param df: input full dataframe
    :param rescut: interactions upto residue id
    :param colcut: upto windows columns
    :return: sub matrix dataframe

    """
    df = df.loc[(df['resI'] <= rescut) & (df['resJ'] <= rescut)].loc[:, :str(colcut)]
    return df


def significant_residues(eigenVector, pers_modes, cutoff, resids):
    """
    Get significant residues from persistant modes

    :param eigenVector: numpy eigenvector matrix
    :param pers_modes: index of significant modes to check for important residues
    :param cutoff: weight cutoff
    :return: significant residues
    :rtype: list

    """
    # Get sigificant residues from significant modes
    r = list()
    for j in pers_modes:
        for i in np.where(eigenVector[:, j - 1] * eigenVector[:, j - 1] >= cutoff):
            r.append(resids[i])
    top_res = np.sort(np.unique(np.concatenate(r, axis=0)))
    # print "Residues from significant modes: \n {} and size is {}".format(
    # top_res, top_res.size)
    return top_res

def annotate_modes(eigen_vec, modes, ndx, aa):
    """
    Annotate the modes in eigenvector

    :param eigen_vec: eigenvector numpy matrix
    :param modes: sindices of modes
    :param ndx: continuous indices (ex: range(1,224))
    :param aa: 3 letter aminoacid array

    """
    res = list()
    count = 0
    for i in np.array(modes):
        if count == 0:
            for m, l, n in zip(ndx, aa, (eigen_vec[:, i - 1] * eigen_vec[:, i - 1])):
                res.append([l, n])
            count += 1
        else:
            for m, l, n in zip(ndx, aa, (eigen_vec[:, i - 1] * eigen_vec[:, i - 1])):
                if res[m][1] < n:
                    res[m][1] = n
    return res


def remove_reverse_duplicates_indices(df):
    _index = df.index.names
    if isinstance(df.index, pd.core.index.MultiIndex):
        df = df.reset_index().set_index(["resI", "resJ"])  # setting resi and resJ as index
    else:
        df = df.set_index(["resI", "resJ"])
    non_rev_idx = {tuple(item) for item in map(sorted, df.index.values)}  # reverse duplicates removed

    non_dups = df.loc[list(non_rev_idx)]  # get the non-duplicate table
    non_dups = non_dups.reset_index().set_index(_index)
    return non_dups.sort_values(["resI"])


def get_hydrophobic_interactions(df, hydrphobic_res=None):
    tmp = df[df.index.get_level_values("resnI").isin(hydrphobic_res)]
    hypho_interactions = tmp[tmp.index.get_level_values("resnJ").isin(hydrphobic_res)]
    return remove_reverse_duplicates_indices(hypho_interactions)
