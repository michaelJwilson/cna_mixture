from __future__ import annotations

import numpy as np
import numpy.testing as npt

from scipy.special import logsumexp
from scipy.stats import nbinom, betabinom
from cna_mixture.cna_emission import CNA_emission, reparameterize_beta_binom, reparameterize_nbinom

np.random.seed(314)


def test_cna_emission():
    num_states, normal_coverage, snp_coverage = 1, 10, 100
    
    # NB nbinom.rvs(lost_reads, dropout_rate, size=1)[0]
    means, phi = np.array(num_states * [100]), 1.e-2
    
    rr_pp = reparameterize_nbinom(means, phi)
    ks = nbinom.rvs(rr_pp[:,0], rr_pp[:,1], size=10_000)

    bafs, tau = np.array(num_states * [0.2]), 10.
    pseudo_counts = reparameterize_beta_binom(bafs, tau)
    
    xs = betabinom.rvs(snp_coverage, pseudo_counts[:,1], pseudo_counts[:,0], size=10_000)
    ns = snp_coverage * np.ones_like(xs)

    # NB RDR-like params are read depths, not RDR.
    params = np.array([*(normal_coverage * means), phi, *bafs, tau])
    emission = CNA_emission(num_states, normal_coverage, ks, xs, ns)

    unpacked = emission.unpack_params(params)
    states_bag = emission.get_states_bag(params)

    assert np.array_equal(states_bag, np.array([[100., 0.2]]))
    assert unpacked == (normal_coverage * means, phi,bafs, tau)

    
