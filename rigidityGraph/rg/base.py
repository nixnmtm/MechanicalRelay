import pandas as pd
import os, sys
import logging
import numpy as np
import string


class LoadKbTable(object):
    """
    Load the complete table from given location

    :param filename: Name of the file to be loaded
    :key input_path: path of file input, should be a str, if not given searches filename in Inputs folder

    :returns Dataframe of the data based on CG bead vs Time Windows
    """

    def __init__(self, filename, **kwargs):
        self.filename = filename
        self.module_path = os.path.dirname(os.path.realpath(__file__))
        self.input_path = kwargs.get("input_path", os.path.join(self.module_path + "/../Inputs"))

    def load_table(self) -> pd.DataFrame:
        """
        Load Coupling Strength DataFrame

        :return: Processed Coupling Strength DataFrame

        """
        logging.info("Loading '{}' from {}".format(self.filename, self.input_path))
        _, fileext = os.path.splitext(self.filename)

        if not (fileext[-3:] == "txt" or fileext[-3:] == "bz2"):
            logging.error("Please provide a appropriate file, with extension either txt or bz2")
            exit(1)
        filepath = os.path.join(self.input_path, self.filename)
        try:
            table = pd.read_csv(filepath, sep=' ')
            if str(table.columns[0]).startswith("Unnamed"):
                table = table.drop(table.columns[0], axis=1)
            logging.info("File loaded.")
            return table
        except IOError as e:
            logging.error(f'Error in loading file: {str(e)}')


class BaseRG(object):
    """
    Base class for building Coupling strength Table sum and mean for all segids including
    their secondary structure splits.

    :key ressep: residue separation( >= I,I + ressep), (default=3)
    :key interSegs: two segments names for inter segment analysis, should be a tuple

    """

    def __init__(self, table: pd.DataFrame, **kwargs):

        self.grouping = ["segidI", "resI", "resnI", "segidJ", "resJ", "resnJ"]
        self._index = ["segidI", "resI", "I", "segidJ", "resJ", "J"]
        self._complete_idx = ["segidI", "resI", "resnI", "I", "segidJ", "resJ", "resnJ", "J"]
        self.splitkeys = ["BB", "BS", "SS"]

        # kwargs
        self.ressep = kwargs.get('ressep', 3)
        self.interSegs = kwargs.get('interSegs', None)  # should be a tuple
        self.table = table
        self.tot_nres = len(self.table.resI.unique())
        self.exclude_disul = kwargs.get('exclude_disul', True)
        self.disul_bonds = kwargs.get('disul_bonds', None)

        if self.exclude_disul and self.disul_bonds is None:
            logging.error('The -exclude_disul argument requires the -disul_bonds provided with list of bonds')
            sys.exit()

    def splitSS(self, write: bool = False) -> dict:
        """
        Split based on secondary structures.

        The CG sites are named as N, O for the backbone amide Nitrogen and corboxyl Oxygen.
        The sidechain site is named CB.

        | BB - Backbone-Backbone Interactions
        | BS - Backbone-Sidechain Interactions
        | SS - Sidechain-Sidehain Interactions

        :param df: Dataframe to split. If None, df initialized during class instance is taken
        :param write: write after splitting
        :param exclude_disul: exclude disulphide interactions (default: True)
        :return: dict of split DataFrames

        """

        sstable = dict()
        tmp = self.table.copy(deep=True)
        # try:
        # BACKBONE-BACKBONE
        sstable['BB'] = tmp[((tmp["I"] == 'N') | (tmp["I"] == 'O') | (tmp["I"] == 'ions'))
                            & ((tmp["J"] == 'N') | (tmp["J"] == 'O') | (tmp["J"] == 'ions'))]

        # BACKBONE-SIDECHAIN
        BS = tmp[((tmp["I"] == "N") | (tmp["I"] == 'O')) & (tmp["J"] == 'CB')]
        SB = tmp[(tmp["I"] == 'CB') & ((tmp["J"] == "N") | (tmp["J"] == 'O'))]
        sstable['BS'] = pd.concat([BS, SB], axis=0, ignore_index=True)

        if self.exclude_disul:
            _bs = sstable['BS'].set_index(self._complete_idx)
            df_BS = sstable['BS'].set_index(["resI", "resJ"])
            for ds in self.disul_bonds:
                df_mask = ((df_BS.index.get_level_values("resI") == ds[0]) & (
                        df_BS.index.get_level_values("resJ") == ds[1]) |
                           (df_BS.index.get_level_values("resI") == ds[1]) & (
                                   df_BS.index.get_level_values("resJ") == ds[0]))
                _tmp = df_BS[df_mask].reset_index().set_index(self._complete_idx)
                _bs = _bs.drop(_tmp.index)
            sstable['BS'] = _bs.reset_index()

        # SIDECHAIN-SIDECHAIN
        sstable['SS'] = tmp[((tmp["I"] == "CB") | (tmp["I"] == "ions")) & ((tmp["J"] == "CB") | (tmp["J"] == "ions"))]
        if self.exclude_disul:
            _ss = sstable['SS'].set_index(self._complete_idx)
            df_SS = sstable['SS'].set_index(["resI", "resJ"])
            for ds in self.disul_bonds:
                df_mask = ((df_SS.index.get_level_values("resI") == ds[0]) & (
                        df_SS.index.get_level_values("resJ") == ds[1]) |
                           (df_SS.index.get_level_values("resI") == ds[1]) & (
                                   df_SS.index.get_level_values("resJ") == ds[0]))
                _tmp = df_SS[df_mask].reset_index().set_index(self._complete_idx)
                _ss = _ss.drop(_tmp.index)
            sstable['SS'] = _ss.reset_index()

        # write the file, if needed
        if write:
            # write the files in current directory
            sstable['BB'].to_csv("kb_BB.txt", header=True, sep=" ", index=False)
            sstable['BS'].to_csv("kb_BS.txt", header=True, sep=" ", index=False)
            sstable['SS'].to_csv("kb_SS.txt", header=True, sep=" ", index=False)

        return sstable

    def sepres(self, table) -> object:
        """
        Residue Separation

        :param table: table for sequence separation
        :param ressep: sequence separation to include (eg.  >= I,I + ressep), default is I,I+3)
        :return: DataFrame after separation
        """

        ressep = self.ressep
        # logging.info("DataFrame is populated with ressep: {}".format(ressep))
        tmp = table[table["segidI"] == table["segidJ"]]
        tmp = tmp[
            (tmp["resI"] >= tmp["resJ"] + ressep) |
            (tmp["resJ"] >= tmp["resI"] + ressep)
            ]
        diff = table[table["segidI"] != table["segidJ"]]
        df = pd.concat([tmp, diff], axis=0)
        return df

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

    def _nres(self, segid=None, ss=None):
        """
        Return number of residues in a segment or inter-segments

        :param segid: segment id
        :param ss: splitkey
        :return: number of residue (int)

        """
        return len(self.table_sum()[segid][ss].index.get_level_values("resI").unique())

    def _resids(self, segid=None, ss=None):
        """
        Return resids of given segments.

        :param seg: segid
        :return: array of residue ids

        """

        return self.table_sum[segid][ss].index.get_level_values("resI").unique()

    def _segids(self):
        """
        Return segids of given table.

        :return: list of segids of given table.

        """

        return self.table.segidI.unique()

    def _refactor_resid(self, df):
        """
        Rename the resids of inter segments.
        | Method should be called only if segments have overlapping resids

        :param df: Dataframe with overalpping resids in segments
        :return: DataFrame with renamed resids
        """
        alphs = list(string.ascii_uppercase)
        if self.interSegs is not None:
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()
            for n, seg in enumerate(self.interSegs):
                resids = df[df.segidI == seg].resI.unique()
                if not resids.dtype == str:
                    mapseg = [alphs[n] + str(i) for i in resids]
                    mapd = dict(zip(resids, mapseg))
                    df.loc[df['segidI'] == seg, 'resI'] = df['resI'].map(mapd)
                    df.loc[df['segidJ'] == seg, 'resJ'] = df['resJ'].map(mapd)
            renamed = df.set_index(self.grouping)
            return renamed
        else:
            logging.warning(f"interSegs argument is None, but {self._refactor_resid.__name__} invoked")

    def _comp_resid(self, df):
        """
        Compare resids of the two segments and return True if overlap exists in resids

        :param df: DataFrame to check for comparision
        :return: Boolean
        """

        if self.interSegs is not None:
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()
            r1 = df[df.segidI == self.interSegs[0]].resI.unique()
            r2 = df[df.segidI == self.interSegs[1]].resI.unique()
            return np.intersect1d(r1, r2).size > 0

    def table_sum(self):
        """
        Returns the sum table based on the self.grouping

        :return: dict of sum tables

        """

        smtable = dict()
        sstable = self.splitSS()
        if self.interSegs is not None:
            seg1 = self.interSegs[0]
            seg2 = self.interSegs[1]

        for seg in self.table.segidI.unique():
            smtable[seg] = dict()
            for key in self.splitkeys:
                if not sstable[key].empty:
                    tmp = self.sepres(table=sstable[key]).groupby(self.grouping).sum()
                    mask = (tmp.index.get_level_values("segidI") == seg) & \
                           (tmp.index.get_level_values("segidJ") == seg)
                    smtable[seg][key] = tmp[mask]

        if self.interSegs is not None and isinstance(self.interSegs, tuple):
            smtable[self.interSegs] = dict()
            if seg1 == seg2:
                raise IOError("Inter segments should not be same")
            for key in self.splitkeys:
                tmp = self.sepres(table=sstable[key]).groupby(self.grouping).sum()
                mask = (tmp.index.get_level_values("segidI") == seg1) & \
                       (tmp.index.get_level_values("segidJ") == seg2)
                revmask = (tmp.index.get_level_values("segidI") == seg2) & \
                          (tmp.index.get_level_values("segidJ") == seg1)
                diff = pd.concat([tmp[mask], tmp[revmask]], axis=0)
                same = pd.concat([smtable[seg1][key], smtable[seg2][key]], axis=0)
                inter = pd.concat([same, diff], axis=0)
                if self._comp_resid(inter):
                    # logging.warning("resids overlap, refactoring resids")
                    inter = self._refactor_resid(inter)
                smtable[self.interSegs][key] = inter
        return smtable

    def table_mean(self):
        """
        Return Mean of table

        :return: dict of mean tables, format as table_sum()

        """
        stab = self.table_sum()
        mntable = dict()
        for seg in stab.keys():
            mntable[seg] = {key: stab[seg][key].mean(axis=1) for key in stab[seg].keys()}
        return mntable