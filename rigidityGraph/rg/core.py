import pandas as pd
import logging
import numpy as np
from natsort import natsorted
from rg.base import BaseRG, LoadKbTable


class RGCore(BaseRG):
    """
    Core Class for The Molecular Graph Theory Analysis

    """
    def __init__(self, table, **kwargs):
        super(RGCore, self).__init__(table, **kwargs)

        self.interSegs = kwargs.get('interSegs', None)
        if self.interSegs is not None:
            self.segid = self.interSegs
        else:
            self.segid = kwargs.get("segid")
        self.sskey = kwargs.get("sskey")

    def get_sum_table(self):
        """
        Get table sum of specific secondary structure and segid

        :param segid: segid
        :param sskey: secondary structure key
        :return: Dataframe of sum table
        """
        return self.table_sum()[self.segid][self.sskey]

    def get_mean_table(self):
        """
        Get mean table of specific secondary structure and segid
        :return: Dataframe of mean table
        """
        return self.table_mean()[self.segid][self.sskey]

    def get_table(self):
        """
        Get table based on sskey and segid

        :return: Dataframe
        """

        splits = self.splitSS()[self.sskey]
        return self.get_intraseg_df(splits, self.segid)

    @staticmethod
    def get_intraseg_df(table, segid):
        logging.info(f"Getting only the {segid} segment interactions")
        if isinstance(table.index, pd.MultiIndex):
            mask = (table.index.get_level_values("segidI") == segid) & \
                   (table.index.get_level_values("segidJ") == segid)
            return table[mask]
        else:
            mask = (table["segidI"] == segid) & (table["segidJ"] == segid)
            return table[mask]

    def get_resids(self):
        """
        Ger number of residues in a segment
        :return:
        """

        tab = self.get_mean_table()
        if isinstance(tab, pd.Series) and isinstance(tab.index, pd.MultiIndex):
            tab = tab.reset_index()
        rmat = tab.drop(["segidI", "segidJ", "resnI", "resnJ"], axis=1).set_index(['resI', 'resJ']).unstack(fill_value=0)
        return rmat.index.values

    def rigidity_mat(self, tab=None):
        """
        Build Rigidity Matrix.
        | Input should be a :class:`pd.Series`

        :param tab: Mean series or window series
        :return:  rigidity matrix, type dataframe

        :Example:

        >>> load = LoadKbTable(filename="holo_pdz.txt.bz2")
        >>> kb_aa = load.load_table()
        >>> core = RGCore(kb_aa, segid="CRPT", sskey="BB", ressep=1)
        >>> print(core.rigidity_mat())
                    5           6           7           8           9
        5  172.692126  169.063123    3.482413    0.139217    0.007373
        6  169.063123  364.543558  193.112981    2.314533    0.052921
        7    3.482413  193.112981  390.274191  192.792781    0.886016
        8    0.139217    2.314533  192.792781  390.518684  195.272153
        9    0.0073`73    0.052921    0.886016  195.272153  196.218462

        """
        if tab is None:
            tab = self.get_mean_table()
        if isinstance(tab, pd.Series) and isinstance(tab.index, pd.MultiIndex):
            tab = tab.reset_index()
        if tab.groupby("resI").sum().shape[1] > 1:
            diag_val = tab.groupby("resI").sum().drop("resJ", axis=1).values.ravel()
        else:
            diag_val = tab.groupby("resI").sum().values.ravel()
        rmat = tab.drop(["segidI", "segidJ", "resnI", "resnJ"], axis=1).set_index(['resI', 'resJ']).unstack(fill_value=0)
        ref_mat = rmat.values
        ref_mat[[np.arange(rmat.shape[0])] * 2] = diag_val
        mat = pd.DataFrame(ref_mat, index=rmat.index.values, columns=rmat.index.values)
        mat = mat.reindex(natsorted(mat.index.values)).T.reindex(natsorted(mat.index.values))
        return mat

    def eigh_decom(self, kmat=None):
        """
        Return the eigenvectors and eigenvalues, ordered by decreasing values of the
        eigenvalues, for a real symmetric matrix M. The sign of the eigenvectors is fixed
        so that the mean of its components is non-negative.

        :param kmat: symmetric matrix to perform eigenvalue decomposition

        :return: eigenvalues and eigenvectors

        :Example:

        >>> load = LoadKbTable(filename="holo_pdz.txt.bz2")
        >>> kb_aa = load.load_table()
        >>> core = RGCore(kb_aa, segid="CRPT", sskey="BB", ressep=1)
        >>> egval, egvec = core.eigh_decom()
        >>> assert egval.shape[0] == egvec.shape[0]

        """
        if kmat is None:
            eigval, eigvec = np.linalg.eigh(self.rigidity_mat())
        else:
            eigval, eigvec = np.linalg.eigh(kmat)
        idx = (-eigval).argsort()
        eigval = eigval[idx]
        eigvec = eigvec[:, idx]
        for k in range(eigvec.shape[1]):
            if np.sign(np.mean(eigvec[:, k])) != 0:
                eigvec[:, k] = np.sign(np.mean(eigvec[:, k])) * eigvec[:, k]
        return eigval, eigvec

    def windows_eigen_decom(self):
        """
        Return csm_mat, eigenVectors and eigenValues of all windows

        """

        stab = self.get_sum_table()
        nwind = stab.columns.size
        npos = stab.index.get_level_values('resI').unique().size
        t_mat = np.zeros((nwind, npos, npos))
        t_vec = np.zeros((nwind, npos, npos))
        t_val = np.zeros((nwind, npos))
        assert npos == len(self.get_resids()), "Mismatch in number of residues"
        for i in range(nwind):
            time_mat = self.rigidity_mat(tab=stab.iloc[:, i])
            tval, tvec = self.eigh_decom(time_mat)
            t_val[i, :] = tval
            t_vec[i, :, :] = tvec
            t_mat[i, :, :] = time_mat
        return t_mat, t_val, t_vec

    def evec_dotpdts(self, kmat=None):
        """
        Evaluate dot products between eigen vectors of decomposed mean mgt matrix and each window mgt matrix


        .. math::
            \\mathbf{M_{ij} = {U^r_i \\cdot U^w_j}}


        r - reference mean matrix eigen vectors

        w - windows eigen vectors

        """
        logging.info("This may take some time based on the number of windows analyzed")
        tmat, tval, tvec = self.windows_eigen_decom()
        logging.info("Decomposition of rigidity matrix in each segemnt is completed")
        meval, mevec = self.eigh_decom(kmat=kmat)
        logging.info("Decomposition of Mean rigidity matrix done")
        logging.info("Calculating similarity between eigenmodes")
        dps = np.zeros(tvec.shape)
        for t in range(tvec.shape[0]):
            dps[t] = np.dot(mevec.T, tvec[t])
        return dps

    def average_mode_content(self, dot_mat=None, kmat=None):
        """
        Calculate average mode content of eigenmodes

        :param dot_mat: matrix with dot product values of each mode.
        :return: persistence of each mode
        :rtype: list
        """

        if dot_mat is None:
            dot_mat = self.evec_dotpdts(kmat=kmat)
        mpers = []
        logging.info("Calculating average mode content")
        for m in range(dot_mat.shape[1]):
            # sort and get max value (ie) last element
            ps = [sorted(abs(dot_mat[w][m]))[-1] for w in range(dot_mat.shape[0])]
            mpers.append(np.asarray(ps).mean())
        return mpers

