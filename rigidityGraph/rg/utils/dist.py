import scipy.stats
import warnings
import random
import math
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import logging

# just for surpressing warnings
warnings.simplefilter('ignore')


class CheckPersDist(object):
    """
    Check for the distribution of average mean mode content
    Get the best fit out of the given distributions

    """

    def __init__(self, pers, **kwargs):

        """
        This is :func:`__init__` docstring

        :param iteration: iterations (default=1)
        :param exclude:
        :param verbose: verbosity (default=False)
        :param top: rank and give top results (default=10)
        :param processes: number o processes (default=-1) meaning use all
        """

        # list of all available distributions
        self.pers = pers
        self.iteration = kwargs.get("iteration", 1)
        self.exclude = kwargs.get("exclude", 10.)
        self.verbose = kwargs.get("verbose", False)
        self.top = kwargs.get("top", 10)
        self.processes = kwargs.get("processes", -1)
        self.cdfs = {
            "alpha": {"p": [], "D": []},  # Alpha
            "anglit": {"p": [], "D": []},  # Anglit
            "arcsine": {"p": [], "D": []},  # Arcsine
            "beta": {"p": [], "D": []},  # Beta
            "betaprime": {"p": [], "D": []},  # Beta Prime
            "bradford": {"p": [], "D": []},  # Bradford
            "burr": {"p": [], "D": []},  # Burr
            "cauchy": {"p": [], "D": []},  # Cauchy
            "chi": {"p": [], "D": []},  # Chi
            "chi2": {"p": [], "D": []},  # Chi-squared
            "cosine": {"p": [], "D": []},  # Cosine
            "dgamma": {"p": [], "D": []},  # Double Gamma
            "dweibull": {"p": [], "D": []},  # Double Weibull
            "erlang": {"p": [], "D": []},  # Erlang
            "expon": {"p": [], "D": []},  # Exponential
            "exponweib": {"p": [], "D": []},  # Exponentiated Weibull
            "exponpow": {"p": [], "D": []},  # Exponential Power
            "f": {"p": [], "D": []},  # F (Snecdor F)
            "fatiguelife": {"p": [], "D": []},  # Fatigue Life (Birnbaum-Sanders)
            "fisk": {"p": [], "D": []},  # Fisk
            "foldcauchy": {"p": [], "D": []},  # Folded Cauchy
            "foldnorm": {"p": [], "D": []},  # Folded Normal
            "frechet_l": {"p": [], "D": []},  # Frechet Left Sided, Weibull_max
            "gamma": {"p": [], "D": []},  # Gamma
            "gausshyper": {"p": [], "D": []},  # Gauss Hypergeometric
            "genexpon": {"p": [], "D": []},  # Generalized Exponential
            "genextreme": {"p": [], "D": []},  # Generalized Extreme Value
            "gengamma": {"p": [], "D": []},  # Generalized gamma
            "genhalflogistic": {"p": [], "D": []},  # Generalized Half Logistic
            "genlogistic": {"p": [], "D": []},  # Generalized Logistic
            "genpareto": {"p": [], "D": []},  # Generalized Pareto
            "gilbrat": {"p": [], "D": []},  # Gilbrat
            "gompertz": {"p": [], "D": []},  # Gompertz (Truncated Gumbel)
            "gumbel_l": {"p": [], "D": []},  # Left Sided Gumbel, etc.
            "gumbel_r": {"p": [], "D": []},  # Right Sided Gumbel
            "halfcauchy": {"p": [], "D": []},  # Half Cauchy
            "halflogistic": {"p": [], "D": []},  # Half Logistic
            "halfnorm": {"p": [], "D": []},  # Half Normal
            "hypsecant": {"p": [], "D": []},  # Hyperbolic Secant
            "invgamma": {"p": [], "D": []},  # Inverse Gamma
            "invgauss": {"p": [], "D": []},  # Inverse Normal
            "invweibull": {"p": [], "D": []},  # Inverse Weibull
            "johnsonsb": {"p": [], "D": []},  # Johnson SB
            "johnsonsu": {"p": [], "D": []},  # Johnson SU
            "laplace": {"p": [], "D": []},  # Laplace
            "logistic": {"p": [], "D": []},  # Logistic
            "loggamma": {"p": [], "D": []},  # Log-Gamma
            "loglaplace": {"p": [], "D": []},  # Log-Laplace (Log Double Exponential)
            "lognorm": {"p": [], "D": []},  # Log-Normal
            "lomax": {"p": [], "D": []},  # Lomax (Pareto of the second kind)
            "maxwell": {"p": [], "D": []},  # Maxwell
            "mielke": {"p": [], "D": []},  # Mielke's Beta-Kappa
            "nakagami": {"p": [], "D": []},  # Nakagami
            "ncx2": {"p": [], "D": []},  # Non-central chi-squared
            "ncf": {"p": [], "D": []},  # Non-central F
            "nct": {"p": [], "D": []},  # Non-central Student's T
            "norm": {"p": [], "D": []},  # Normal (Gaussian)
            "pareto": {"p": [], "D": []},  # Pareto
            "pearson3": {"p": [], "D": []},  # Pearson type III
            "powerlaw": {"p": [], "D": []},  # Power-function
            "powerlognorm": {"p": [], "D": []},  # Power log normal
            "powernorm": {"p": [], "D": []},  # Power normal
            "rdist": {"p": [], "D": []},  # R distribution
            "reciprocal": {"p": [], "D": []},  # Reciprocal
            "rayleigh": {"p": [], "D": []},  # Rayleigh
            "rice": {"p": [], "D": []},  # Rice
            "recipinvgauss": {"p": [], "D": []},  # Reciprocal Inverse Gaussian
            "semicircular": {"p": [], "D": []},  # Semicircular
            "t": {"p": [], "D": []},  # Student's T
            "triang": {"p": [], "D": []},  # Triangular
            "truncexpon": {"p": [], "D": []},  # Truncated Exponential
            "truncnorm": {"p": [], "D": []},  # Truncated Normal
            "tukeylambda": {"p": [], "D": []},  # Tukey-Lambda
            "uniform": {"p": [], "D": []},  # Uniform
            "vonmises": {"p": [], "D": []},  # Von-Mises (Circular)
            "wald": {"p": [], "D": []},  # Wald
            "weibull_min": {"p": [], "D": []},  # Minimum Weibull (see Frechet)
            "weibull_max": {"p": [], "D": []},  # Maximum Weibull (see Frechet)
            "wrapcauchy": {"p": [], "D": []},  # Wrapped Cauchy
            "ksone": {"p": [], "D": []},  # Kolmogorov-Smirnov one-sided (no stats)
            "kstwobign": {"p": [], "D": []}}  # Kolmogorov-Smirnov two-sided test for Large N

    def check(self, data, fct, verbose=False):
        """
        :paran data: data to check input from ::func run
        :param fct: distribution to test
        :param verbose: verbosity
        :return: tuple of (distribution name, probability, D)
        """
        # fit our data set against every probability distribution
        parameters = eval("scipy.stats." + fct + ".fit(data)")
        # Applying the Kolmogorov-Smirnof two sided test
        D, p = scipy.stats.kstest(self.pers, fct, args=parameters)

        if math.isnan(p): p = 0
        if math.isnan(D): D = 0

        if verbose:
            print(fct.ljust(16) + "p: " + str(p).ljust(25) + "D: " + str(D))

        return fct, p, D

    def plot(self, fct, p_cut, xlabel=None, ylabel=None, fontsize=16, figsize=None):
        """
        :return plots image and returns pcut values
        """
        # plot data
        params = eval("scipy.stats." + fct + ".fit(data)")
        f = eval("scipy.stats." + fct + ".freeze" + str(params))
        x = np.linspace(f.ppf(0.001), f.ppf(0.999), len(self.pers))
        plt.plot(x, f.pdf(x), lw=3, label=fct)
        plt.axvline(p_cut, color='r', linestyle='--')

        font = {'family': 'Arial',
                'weight': 'bold',
                'size': 16
                }
        sns.set(style="ticks")
        f2, ax_hist2 = plt.subplots(1, figsize=figsize)
        hist_kws = {"color": "k", 'edgecolor': 'black', 'alpha': 1.0}
        sns.distplot(self.pers, ax=ax_hist2, bins=max(10, int(len(self.pers) / 10)), hist_kws=hist_kws, kde=False, norm_hist=True)

        # label and ticks params
        if xlabel: ax_hist2.set_xlabel(xlabel=xlabel, fontsize=fontsize)
        if ylabel: ax_hist2.set_ylabel(ylabel=ylabel, fontsize=fontsize)
        ax_hist2.tick_params(labelsize=fontsize - 10)
        ax_hist2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax_hist2.legend(loc='best', frameon=False)
        #plt.title("Top " + str(len(fcts)) + " Results")
        plt.tight_layout()
        plt.show()

    def run(self, sort_it=True):
        """
        :param data: data to check distribution
        :param sort_it: whether to sort the results
        :return: sort_it: True: sorted list of tuples with (distribution name, probability, D)
                          False: dictionary with distribution functions {"distribution name": {"p":float, "D":float}}
        """
        for i in range(self.iteration):
            if self.iteration == 1:
                data = self.pers
            else:
                data = [value for value in self.pers if random.random() >= self.exclude / 100]

            results = Parallel(n_jobs=self.processes)(delayed(self.check)(data, fct, self.verbose) for fct in self.cdfs.keys())

            for res in results:
                key, p, D = res
                self.cdfs[key]["p"].append(p)
                self.cdfs[key]["D"].append(D)
            if sort_it:
                # print("-------------------------------------------------------------------")
                # print("Top %d after %d iteration(s)" % (self.top, i + 1,))
                # print("-------------------------------------------------------------------")
                best = sorted(self.cdfs.items(), key=lambda elem: scipy.median(elem[1]["p"]), reverse=True)
                for t in range(self.top):
                    fct, values = best[t]
                    # print(str(t + 1).ljust(4), fct.ljust(16),
                    #      "\tp: ", scipy.median(values["p"]),
                    #      "\tD: ", scipy.median(values["D"]),
                    #      end="")
                    if len(values["p"]) > 1:
                        pass
                    #    print("\tvar(p): ", scipy.var(values["p"]),
                    #          "\tvar(D): ", scipy.var(values["D"]), end="")
                    # print()
        return best

    def calc_pcutoff(self, cdf_cut=0.90, top=1):

        # Get the probability distribution and CDF data
        best_fcts = self.run()
        top_fcts = best_fcts[:top]
        pcuts = dict()
        # plot fitted probability
        for i in range(len(top_fcts)):
            fct = best_fcts[i][0]
            params = eval("scipy.stats." + fct + ".fit(self.pers)")
            f = eval("scipy.stats." + fct + ".freeze" + str(params))
            x = np.linspace(f.ppf(0.001), f.ppf(0.999), len(self.pers))
            cd = f.cdf(x)
            tmp = f.pdf(x).argmax()
            if abs(max(self.pers)) > abs(min(self.pers)):
                tail = cd[tmp:len(cd)]
            else:
                cd = 1 - cd
                tail = cd[0:tmp]
            diff = abs(tail - cdf_cut)
            x_pos = diff.argmin()
            p_cut = np.round(x[x_pos + tmp], 2)
            pcuts[fct] = p_cut

        if top == 1:
            pcut = max([v for k, v in pcuts.items()])
            cdf_name = [k for k, v in pcuts.items()][0]
        elif top > 1:
            pcut = [v for k, v in pcuts.items()]
            cdf_name = [k for k, v in pcuts.items()]
        else:
            logging.error("Top value must be greatr than 0")

        return pcut, cdf_name
