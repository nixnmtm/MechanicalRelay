import numpy as np
import pandas as pd
from natsort import natsorted
import matplotlib.pyplot as plt

# The prominent modes obtained from each rigidity graphs and their corresponding counter modes are also included
# to obtain their union list of residues


promModes = dict()
promModes["holo"] = dict()
promModes["apo"] = dict()
promModes["holo"]["BB"] =  np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,15,16,17,18,19,20, 21, 22,23,24,26,27,33,47,64])
promModes["apo"]["BB"] = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 21, 22,17,27,38,47])
promModes["holo"]["BS"] =  np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 18, 22,13, 14, 15, 17, 21])
promModes["apo"]["BS"] = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 21,22,39,18])
promModes["holo"]["SS"] =  np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,26,27,32])
promModes["apo"]["SS"] = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,21,95])


def clean_eigvec(eigvec, wt_cut=0.1):
    """Clean the eigen vectors using its weight cutoff

    Basically cleaning the vectors to include only the significant residues in the
    rigidity matrix constructed from the prominent modes. So the residues that has
    a weight less than the weight citoff is set to 0.

    Parameters:

    :eigvec: input eigen vector
    :w_cut: weight cutoff to use(default: 0.1).
            Note: weight implies the squared value of the eoigenvector components.

    Returs:
    Array of cleaned vector

    """
    evec = np.copy(eigvec)
    indxes = np.where(np.square(evec) < wt_cut)[0]
    evec[indxes] = 0

    return evec


def build_rigidity_matrix_prommodes(*, val, vec, pmodes, clean=True, wt_cut=0.1):
    """ Construct rigidity matrix using only the prominent modes

    Paramaters:
    :param val: eigenvalues
    :param vec: eigenvector matrix *Symmetric so left and right are equal
    :param pmodes: The list of persistent modes

    Returns:
    Rigidity (N X N) matrix

    """
    Npos = len(val)
    mat = np.zeros([Npos, Npos])
    for I in range(Npos):
        for J in range(Npos):
            if I == J:
                mat[I, J] = 0
            else:
                comps = []
                for i in pmodes:
                    if clean:
                        eigvec = clean_eigvec(vec[:, i - 1], wt_cut=wt_cut)
                    else:
                        eigvec = vec[:, i - 1]
                    comps.append(val[i - 1] * (eigvec[I]) * (eigvec[J]))
                mat[I, J] = np.sum(comps)
    return mat


def get_sliced_kIJ(data, *, cutoff=5.0, Npos=None, ssres=None):
    """
    Get the sliced DataFrame based on the K_IJ cutoff


    """
    if ssres is not None:
        mat = pd.DataFrame(data[:Npos, :Npos], columns=ssres, index=ssres).unstack()
    else:
        mat = pd.DataFrame(data[:Npos, :Npos], columns=range(1, Npos + 1), index=range(1, Npos + 1)).unstack()
    mat.index.names = ["resI", "resJ"]
    kIJ_sliced = mat[mat.abs() > cutoff]
    return kIJ_sliced


def drop_rev(df):
    mask = df.index.get_level_values('resJ') > df.index.get_level_values('resI')
    return df[mask]


def rename_IJ_df(df, mapping):
    """Rename the dataframe residues based on PDB ID

    The input df must be indexed with resI,resJ
    """
    if len(df.index.names) == 2:
        df.index.names = ["resI", "resJ"]
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    if "resI" in df.columns:
        df["resI"].replace(mapping, inplace=True)
        df["resJ"].replace(mapping, inplace=True)
        renamed = df.set_index(["resI", "resJ"])
        return renamed
    else:
        print("Dataframe shape is wrong")


def get_pairkb(tab, pair):
    if isinstance(tab.index, pd.MultiIndex):
        tab = tab.reset_index()
    if len(pair) < 2:
        sliced = tab[(tab["resI"] == pair[0])]
    if len(pair) == 2:
        sliced = tab[(tab["resI"] == pair[0]) & (tab["resJ"] == pair[1])]
    sliced = sliced.set_index(["resI", "resJ"])
    return sliced


def construct_df_from_respairs(rigmat, respairs):
    df = []
    for i in respairs:
        df.append(get_pairkb(tab=rigmat, pair=i))
    df = pd.concat(df)
    return df


def build_cleaned_prom_Rmat(indices, df, Npos=224):
    """Construct the new cleaned prominent rigigidity matrix

    The cleaned prominent matrix contains the original K_IJ values to overcome the adjacent
    residue KIJ's from additive effects.

    :indices: index obtained from :func: get_ridity_indices, index 0
    :df: dataframe obtained from :func: get_ridity_indices, index 1

    Returns:
    Cleaned prominent KIJ matrix, array

    """

    cleaned_hRmat = np.zeros([Npos, Npos])
    for i in range(Npos):
        for j in range(Npos):
            if (i + 1, j + 1) in list(indices):
                cleaned_hRmat[i, j] = df.loc[i + 1, j + 1][0]
            else:
                cleaned_hRmat[i, j] = 0.
    cleaned_hRmat = cleaned_hRmat + cleaned_hRmat.T
    return cleaned_hRmat


def create_dups(df):
    _df = df.reset_index()
    dup_cols = ["resJ", "resI", "a-h"]
    dups = _df.copy()
    dups.columns = dup_cols
    _df = pd.concat([_df, dups]).set_index(["resI", "resJ"])
    return _df


def get_ridity_indices(*, rigpro_mat, rig_mat, mapping=None, Npos=224, ssres=None):
    """
    Get Ridity matrix indices from prominent rigidity matrix

    The r'$K_{IJ}^{\Pi}$' values has cartain values that come due to the additive effects of adjacent residues
    which would otherwise be cancelled from all the modes. SO it is decided to get the r'$K_{IJ}^{\Pi}$' from
    prominent modes and then get their actual original values r'$K_{IJ}$' from original matrix.


    Parameters:
    :rigpro_mat: Prominent regidity matrix constructed from prominent modes
    :rig_mat: The complete matrix
    :mapping: dict map for renaming the residues
    :Npos: number of residues
    :ssres: the residue numbers of SS rigidity graph. (199)

    Returns:
    A list with orginal matrix indices

    """
    pro_mat = drop_rev(get_sliced_kIJ(rigpro_mat, cutoff=0.1, Npos=Npos, ssres=ssres))
    pro_idx = pro_mat.index.values
    # rename to get the similar index
    if mapping is None:
        rig_mat = rig_mat.iloc[:Npos, :Npos].unstack()
        rig_mat.index.names = ["resI", "resJ"]
    else:
        rig_mat = rename_IJ_df(rig_mat.iloc[:Npos, :Npos].unstack(), mapping=mapping)
    mask = (rig_mat.index.get_level_values("resI") != rig_mat.index.get_level_values("resJ"))
    rig_mat = rig_mat[mask]

    rig_idx = rig_mat[rig_mat > 0.1].dropna().index.values

    # get indices of Kij from original matrix based on the prominent Kij
    df_idx = [i for i in pro_idx if i in list(rig_idx)]

    return df_idx


def merge_list_of_tuples(lst1, lst2):
    _merge = lst1 + lst2
    merged = []
    for i in _merge:
        if i not in merged:
            merged.append(i)
    return natsorted(merged)


def construct_df_from_respairs(rigmat, respairs, mapping=None, Npos=224):
    rigmat = rename_IJ_df(rigmat.iloc[:Npos, :Npos].unstack(), mapping=mapping)
    mask = (rigmat.index.get_level_values("resI") != rigmat.index.get_level_values("resJ"))
    rig_mat = rigmat[mask]
    df = []
    for i in respairs:
        df.append(get_pairkb(tab=rig_mat, pair=i))
    df = pd.concat(df)
    return df


def allostery(*, merged_pairs, arigmat, hrigmat, mapping=None):
    hprom_pairs = construct_df_from_respairs(hrigmat, respairs=merged_pairs, mapping=mapping)

    aprom_pairs = construct_df_from_respairs(arigmat, respairs=merged_pairs)

    prominent_pairs = pd.concat([aprom_pairs, hprom_pairs], axis=1)
    prominent_pairs.columns = ["a", "h"]
    prominent_pairs = prominent_pairs.fillna(0.)
    prominent_pairs["a-h"] = prominent_pairs["a"] - prominent_pairs["h"]
    return prominent_pairs


def plot_get_whiskers(data):
    import scipy.stats as stats
    pos_data = abs(data)
    Q3 = np.quantile(pos_data, 0.75)
    Q1 = np.quantile(pos_data, 0.25)
    max_whisker = Q3 + (1.25 * stats.iqr(pos_data))  # Q3+1.5IQR
    maxwhisk = pos_data[data <= max_whisker].max()
    min_whisker = Q1 - (1.25 * stats.iqr(pos_data))  # Q1-1.5IQR
    minwhisk = pos_data[pos_data >= min_whisker].min()
    f, ax = plt.subplots(1, figsize=(8, 1))
    ax.boxplot(pos_data, showbox=True, showcaps=True, vert=False)
    ax.axvline(Q3)
    # ax.axvline(min(maxwhisk, abs(minwhisk)))
    # ax.axvline(-min(maxwhisk, abs(minwhisk)))
    plt.show()
    return Q3, maxwhisk