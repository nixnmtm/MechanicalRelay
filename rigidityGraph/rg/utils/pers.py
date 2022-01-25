
import logging
import math
import scipy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rg.utils.dist import CheckPersDist
from rg.extras import significant_residues
from matplotlib.ticker import FixedLocator


class PersistenceUtils(object):
    """
    Utilities for Persistence analysis

    :param pers: Persistence values of modes
    :param eigenval: Eigenvalues of Decomposed Mean MGT matrix

    """
    def __init__(self, eigenval, pers, eigcut="outliers",
                 pcut_range=np.arange(0.5, 0.91, 0.01),
                 wt_cut_range=np.arange(0.01, 0.41, 0.01)):
        self.pers = pers
        self.eigenval = eigenval
        self.eigcut = eigcut
        self.pcut_range = pcut_range
        self.wt_cut_range = wt_cut_range

    def get_pcutoff(self, cdf_cut):
        cd = CheckPersDist(self.pers)
        pcut, cdf_name = cd.calc_pcutoff(cdf_cut=cdf_cut)
        return pcut, cdf_name

    def get_allpers_modes(self):
        _, mode_cut = self.get_eigval_outliers()
        allmodes = {str(np.round(pcut, 2)): [i + 1 for i, j in enumerate(self.pers) if j > pcut and i < mode_cut]
                    for pcut in self.pcut_range}
        return allmodes

    def get_pers_modes(self, pcut):
        """
        Get persistence modes of specific pcut

        """
        return self.get_allpers_modes()[pcut]

    def get_eigval_outliers(self):

        if isinstance(self.eigcut, str):
            # get the point to draw cutoff outline
            q3 = np.quantile(self.eigenval, 0.75)
            max_whisker = q3 + (1.5 * scipy.stats.iqr(self.eigenval))  # Q3+1.5IQR
            maxwhisk = self.eigenval[self.eigenval <= max_whisker].max()
            if self.eigcut == "outliers":
                mode_cut = (np.where(self.eigenval[self.eigenval >= np.round(maxwhisk, 2)])[0] + 1).max()
                return maxwhisk, mode_cut
            if self.eigcut == "Q3":
                mode_cut = (np.where(self.eigenval[self.eigenval >= np.round(q3, 2)])[0] + 1).max()
                return q3, mode_cut
        elif isinstance(self.eigcut, float) or isinstance(self.eigcut, int):
            mode_cut = (np.where(self.eigenval[self.eigenval > np.round(self.eigcut, 2)])[0] + 1).max()
            return self.eigcut, mode_cut
        else:
            logging.error("Unknow variable datatype for 'eigcut'")

    def get_residues_persmodes(self, vec, resids):
        # populate dictionary with residues based on cutoffs
        res_dict = dict()
        for pc in [str(np.round(i, 2)) for i in self.pcut_range]:
            res_dict[pc] = dict()
            for cut in self.wt_cut_range:
                cut = np.round(cut, 2)
                res = significant_residues(vec, self.get_allpers_modes()[pc], cut, resids=resids)
                res_dict[pc][str(cut)] = res
        return res_dict

    #########################################################################################
    # Persistence plot utils
    #########################################################################################
    @staticmethod
    def choose_subplot_dimensions(k):
        if k < 5:
            return k, 2
        else:
            # I've chosen to have a maximum of 3 columns
            return math.ceil(k / 3), 3

    def generate_subplots(self, k, row_wise=False):
        nrow, ncol = self.choose_subplot_dimensions(k)
        # Choose your share X and share Y parameters as you wish:
        figure, axes = plt.subplots(nrow, ncol, sharex="all", sharey="all",
                                    figsize=(ncol * 4, nrow * 3), squeeze=True, gridspec_kw = {'wspace':0.05, 'hspace':0.02})

        # Check if it's an array. If there's only one plot, it's just an Axes obj
        if not isinstance(axes, np.ndarray):
            return figure, [axes]
        else:
            # Choose the traversal you'd like: 'F' is col-wise, 'C' is row-wise
            axes = axes.flatten(order=('C' if row_wise else 'F'))
            # Delete any unused axes from the figure, so that they don't show
            # blank x- and y-axis lines
            for idx, ax in enumerate(axes[k:]):
                figure.delaxes(ax)

                # Turn ticks on for the last ax in each column, wherever it lands
                idx_to_turn_on_ticks = idx + k - ncol if row_wise else idx + k - 1
                for tk in axes[idx_to_turn_on_ticks].get_xticklabels():
                    tk.set_visible(True)
            axes = axes[:k]
            return figure, axes

    def eigval_plots(self, xlabel=None, ylabel=None, title=None, figsize=(9, 8),
                     bins=None, fontsize=16):

        """
        Plot Boxplot and histogram distribution of eigenvalues

        :param data: 1-d data array (eigenvalues)
        :param xlabel: xlabel
        :param ylabel: ylabel
        :param title: title
        :param figsize: size of fig (default (9,8))
        :param bins: number of bins (default None / auto)
        :param eigcut: plot vline (max whisker, min whisker, Q3, Q1)
        :param fontsize: default:16

        .. example
        >>> self.eigval_plots(self.eigenval, xlabel="Eigenvalues", bins=20)
        """

        from matplotlib.ticker import FormatStrFormatter
        sns.set(style="ticks")
        f2, (ax_box2, ax_hist2) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)}, figsize=figsize)
        sns.boxplot(self.eigenval, ax=ax_box2)
        hist_kws = {"color": "k", 'edgecolor': 'black', 'alpha': 1.0}
        sns.distplot(self.eigenval, ax=ax_hist2, bins=bins, hist_kws=hist_kws) \
            if bins else sns.distplot(self.eigenval, ax=ax_hist2)

        # label and ticks params
        if xlabel: ax_hist2.set_xlabel(xlabel=xlabel, fontsize=fontsize)
        if ylabel: ax_hist2.set_ylabel(ylabel=ylabel, fontsize=fontsize)
        ax_hist2.tick_params(labelsize=fontsize - 10)
        ax_hist2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        if title: ax_box2.set(title=title)

        left, bottom, width, height = [0.55, 0.35, 0.4, 0.4]
        ax2 = f2.add_axes([left, bottom, width, height])
        ax2.plot(self.eigenval, marker='o', color='k', label="decay plot")
        ax2.set_ylabel(r"Eigenvalue", fontsize=fontsize-10)
        ax2.set_xlabel("Index", fontsize=fontsize-10)
        ax2.tick_params(labelsize=fontsize-15)

        eigcutline, _ = self.get_eigval_outliers()
        ax2.axhline(eigcutline, color='r', linestyle='--')
        ax_box2.axvline(eigcutline, color='r', linestyle='--')
        ax_hist2.axvline(eigcutline, color='r', linestyle='--')
        plt.tight_layout()
        plt.show()

    def plot_eigpers(self, cdf_cut=0.90, figsize=(9, 5), tick_fontsize=20, skip_modes=None,
                     label_fontsize=32, show_cutoff=True, title=None, savepath=None, annotate=False, fontsize=14):

        """
        Eigenvalue boxplot, persistance vs eigenvalue scatter plot and persistance distribution in a single plot and
        cutoffs shown

        :param eigcut: the eigenvalue cutoff to use
        :param figsize: size of fig
        :param tick_fontsize:
        :param label_fontsize:
        :param show_cutoff: show cutoff lines
        :param save_fig: save fig
        :param annotate: annotate the figure
        :param fontsize: fontsize
        :param skip_modes: modes to skip if they are from only substrtae or inhibitor
        :return: Plot figure

        """

        sns.set(style="ticks")
        f2, ax_scat2 = plt.subplots(1, sharex=True, figsize=figsize)

        if skip_modes is not None:
            skip_modes = np.array(skip_modes)-1
            xdata = np.delete(self.eigenval, skip_modes)
            ydata = np.delete(self.pers, skip_modes)
            sns.scatterplot(xdata, ydata, ax=ax_scat2)
        else:
            sns.scatterplot(self.eigenval, self.pers, ax=ax_scat2)

        pcut, cdf_name = self.get_pcutoff(cdf_cut=cdf_cut)
        logging.info(f"The corresponding p_cut of given CDF cutoff({cdf_cut}) is: {pcut}")
        # Append new plots
        divider = make_axes_locatable(ax_scat2)

        # Add plot in right
        axHisty = divider.append_axes("right", 1.5, pad=0.1, sharey=ax_scat2)
        axHisty.tick_params(labelleft=False, left=False, labelsize=tick_fontsize,
                            bottom=True, labelbottom=True, top=False, labeltop=False)
        axHisty.set_xlabel("counts", fontsize=label_fontsize)

        # add plot above eighist
        axBox = divider.append_axes("top", 0.5, pad=0.1)
        axBox.tick_params(labelleft=False, left=False, labelsize=tick_fontsize,
                          bottom=False, labelbottom=False, top=False, labeltop=False)

        ax_scat2.set_ylabel(r"$r_{\alpha}$", fontsize=label_fontsize)
        ax_scat2.set_xlabel(r"$\overline{\lambda}_{\alpha}$", fontsize=label_fontsize)
        ax_scat2.tick_params(labelsize=tick_fontsize)
        ax_scat2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax_scat2.set_ylim(0, 1.2)

        # Plot pers distribution with cutoff
        if annotate:
            sns.distplot(self.pers, bins=max(20, int(len(self.eigenval) / 10)), vertical=True, norm_hist=True,
                         kde=False, fit=eval("scipy.stats." + cdf_name), ax=axHisty, color='k', label=cdf_name)
            axHisty.legend()
        else:
            sns.distplot(self.pers, bins=max(20, int(len(self.eigenval) / 10)), vertical=True, norm_hist=True,
                         kde=False, fit=eval("scipy.stats." + cdf_name), ax=axHisty, color='k')
        # plot eigenvalue box plot
        sns.boxplot(self.eigenval, ax=axBox)

        if show_cutoff:
            # Add pers cutoff lines
            axHisty.axhline(pcut, color='g', linestyle='--')
            ax_scat2.axhline(pcut, color='g', linestyle='--')

            # retrieve eigval cutoff from boxplot
            Q3 = np.quantile(self.eigenval, 0.75)
            max_whisker = Q3 + (1.5 * scipy.stats.iqr(self.eigenval))  # Q3+1.5IQR
            maxwhisk = self.eigenval[self.eigenval <= max_whisker].max()

            _q3 = self.eigenval[self.eigenval <= Q3].max()

            if isinstance(self.eigcut, int) or isinstance(self.eigcut, float):
                logging.info(f"eigcut: {self.eigcut}")
                ax_scat2.axvline(self.eigcut, color='r', linestyle='--')
                axBox.axvline(self.eigcut, color='r', linestyle='--')
                if annotate:
                    for a in range(len(self.eigenval)):
                        if self.pers[a] > pcut and self.eigenval[a] >= np.round(self.eigcut, 2):
                            ax_scat2.annotate(str(a + 1), xy=(self.eigenval[a], self.pers[a]), fontsize=fontsize)

            if self.eigcut == "outliers":
                logging.info(f"maxwhisk: {maxwhisk}")
                ax_scat2.axvline(maxwhisk, color='r', linestyle='--')
                axBox.axvline(maxwhisk, color='r', linestyle='--')

                if annotate:
                    for a in range(len(self.eigenval)):
                        if self.pers[a] > pcut and self.eigenval[a] >= np.round(maxwhisk, 2):
                            ax_scat2.annotate(str(a + 1), xy=(self.eigenval[a], self.pers[a]), fontsize=fontsize)

                # ev = np.asarray(self.eigenval)
                # ps = np.asarray(self.pers)
                # col = np.where((ps > pcut) & (ev > maxwhisk), 'r',
                #                np.where((ps < pcut) & (ev < maxwhisk), 'gray',
                #                         np.where(ps < pcut, 'gray', "gray")))
                # sns.scatterplot(ev, ps, ax=ax_scat2, c=col)
            if self.eigcut == "Q3":
                print(f"Q3: {Q3}")
                ax_scat2.axvline(Q3, color='r', linestyle='--')
                axBox.axvline(Q3, color='r', linestyle='--')
                if annotate:
                    for a in range(len(self.eigenval)):
                        if self.pers[a] > pcut and self.eigenval[a] >= np.round(_q3, 2):
                            ax_scat2.annotate(str(a + 1), xy=(self.eigenval[a], self.pers[a]), fontsize=fontsize)
        f2.tight_layout()
        if title:
            f2.suptitle(title, fontsize=label_fontsize)
        plt.show()
        if savepath is not None:
            f2.savefig(savepath, dpi=300)

    def plot_persmodes(self, vec, allmodes, cdf_cut=0.90, wt_cut=0.15, pdbaa=None, resids=None, fontsize=12,
                       title="Persistent Eigenmodes", struc=None, tick_rotation='horizontal',
                       tickpositions=None, xticklabels=None, savepath=None, annotate_angle=90, idx_segvline=None):

        """
        Plots the persistent eigenmodes with significant residues annotated.

        :param vec:
        :param allmodes:
        :param cdf_cut:
        :param wt_cut:
        :param pdbaa:
        :param resids:
        :param fontsize:
        :param title:
        :param label: user defined label
        :param tick_rotation:
        :param tickpositions:
        :param xticklabels:
        :param savepath:
        :param annotate_angle:
        :param idx_segvline:  segment splitting vertical line index (eg:S1A: 224)
        :return: plots persistent modes with residues annotated

        """

        legend_properties = {'family': 'Arial', 'weight': 'normal', 'size': fontsize}
        font = {'family': 'Arial', 'weight': 'normal', 'size': fontsize + 4}
        font_ticks = {'family': 'Arial', 'weight': 'normal', 'size': fontsize}

        pcut, cdf_name = self.get_pcutoff(cdf_cut=cdf_cut)
        npos = len([i for i in resids if i[0] == "A"])
        prot_modes = [m for m in allmodes[str(pcut)]
                      if not (np.square(vec[:, m - 1])[:npos] < 0.1).all()]  # exclude only substrate/inhibitor modes
        fig, axes = self.generate_subplots(len(prot_modes), row_wise=True)
        fig.subplots_adjust(hspace=0.5)
        plt.rc('font', **font)

        count = 1
        for ax, m in zip(axes.flatten(), prot_modes):
            aa_residues = []
            cont_residues = []
            m = m - 1  # modes are exact indices
            ax.plot(resids, vec[:, m] * vec[:, m], label=f"{count}$^{{{struc}}}$")
            ax.axhline(wt_cut, color='grey', linestyle='--', linewidth=0.75)
            for k, p, l in zip(resids, pdbaa, vec[:, m] * vec[:, m]):
                if l > wt_cut:
                    ax.annotate(str(p), xy=(k, l), rotation=annotate_angle, fontsize=fontsize)
                    aa_residues.append(str(p))
                    cont_residues.append(str(k))
            #  print(f"Mode {count} : {aa_residues} <--> {cont_residues}  EigVal:{np.round(self.eigenval[m],2)}")
            ax.legend(loc='best', frameon=False, prop=legend_properties)
            if xticklabels is None:
                xticklab = [i[1:] for i in pdbaa][::50]
                xticks = [indx for indx in range(0, len(self.eigenval))][::50]
                ax.set_xticks(xticks)
                ax.set_xticklabels(xticklab, fontdict=font_ticks, rotation=tick_rotation)
            else:
                try:
                    ax.get_xaxis().set_major_locator(FixedLocator(tickpositions))
                    ax.set_xticklabels(xticklabels, fontdict=font_ticks, rotation=tick_rotation)
                except IOError as e:
                    logging.error('Input arguments missing: %s', e)

            yticklab = [str(np.round(yindx, 2)) for yindx in np.arange(0.0, 1.0, 0.2)]
            yticks = [np.round(yindx, 2) for yindx in np.arange(0.0, 1.0, 0.2)]
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklab, fontdict=font_ticks)
            if idx_segvline is not None:
                ax.axvline(idx_segvline, linestyle="--", color="grey", linewidth=0.75)
            count += 1
        #  fig.suptitle(title, fontsize=fontsize)
        fig.tight_layout(rect=[0, 0.1, 0.9, 0.8], h_pad=0.2, w_pad=0.2)
        if savepath:
            fig.savefig(savepath, dpi=600)

        plt.show()
