import time
import logging
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import numpy.random as random

from matplotlib.colors import LogNorm
from scipy.stats import nbinom, betabinom, poisson
from scipy.optimize import approx_fprime, check_grad, minimize
from scipy.special import logsumexp, digamma
from scipy.spatial import KDTree
from scipy.optimize import minimize
from sklearn.mixture import GaussianMixture
from cna_mixture.cna_sim import CNA_sim
from cna_mixture.cna_mixture_params import CNA_mixture_params
from cna_mixture.plotting import plot_rdr_baf_flat, plot_rdr_baf_genome
from cna_mixture.encoding import onehot_encode_states
from cna_mixture.gaussian_mixture import fit_gaussian_mixture
from cna_mixture.cna_mixture import CNA_mixture
from cna_mixture.negative_binomial import reparameterize_nbinom
from cna_mixture.beta_binomial import reparameterize_beta_binom
from cna_mixture_rs.core import (
    betabinom_logpmf,
    nbinom_logpmf,
    grad_cna_mixture_em_cost_nb_rs,
    grad_cna_mixture_em_cost_bb_rs,
)

np.random.seed(1234)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

"""
TODOs:

  - kmean++ like.
  - multiple starts + best likelihood. 
  - regularizer for state overlap.
  - prior to prevent single-state occupancy.                                                                                            
  - callback forward.
  - unit tests.

"""
        
def main():
    start = time.time()
    
    cna_sim = CNA_sim()
    # cna_sim.plot_realization_true_flat()
    
    # plot_rdr_baf_genome("plots/rdr_baf_genome.pdf", cna_sim.rdr_baf)
    
    # fit_gaussian_mixture(cna_sim.rdr_baf)
    
    CNA_mixture(cna_sim.realized_genome_coverage, cna_sim.data).fit()
    
    logger.info(f"\n\nDone ({time.time() - start:.3f} seconds).\n\n")


if __name__ == "__main__":
    main()
