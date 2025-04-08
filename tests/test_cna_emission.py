from __future__ import annotations

import numpy as np
import numpy.testing as npt
from cna_mixture.cna_emission import (
    CNA_emission,
    reparameterize_beta_binom,
    reparameterize_nbinom,
)
from scipy.stats import betabinom, nbinom

np.random.seed(314)


def test_cna_emission():
    num_states, normal_coverage, snp_coverage = 1, 10, 100

    # NB nbinom.rvs(lost_reads, dropout_rate, size=1)[0]
    means, phi = np.array(num_states * [100]), 1.0e-2

    rr_pp = reparameterize_nbinom(normal_coverage * means, phi)
    ks = nbinom.rvs(rr_pp[:, 0], rr_pp[:, 1], size=10_000).astype(np.float64)

    bafs, tau = np.array(num_states * [0.2]), 10.0
    pseudo_counts = reparameterize_beta_binom(bafs, tau)

    xs = betabinom.rvs(
        snp_coverage, pseudo_counts[:, 1], pseudo_counts[:, 0], size=10_000
    ).astype(np.float64)
    ns = snp_coverage * np.ones_like(xs).astype(np.float64)

    # NB RDR-like params are read depths, not RDR.
    params = np.array([*(normal_coverage * means), phi, *bafs, tau])
    emission = CNA_emission(num_states, normal_coverage, ks, xs, ns)

    unpacked = emission.unpack_params(params)
    states_bag = emission.get_states_bag(params)

    assert np.array_equal(states_bag, np.array([[100.0, 0.2]]))
    assert unpacked == (normal_coverage * means, phi, bafs, tau)

    # NB >>>>>>  beta-binomial checks.
    rust_bb_update, rust_state_alpha_betas = emission.cna_mixture_betabinom_update(
        params
    )

    assert np.array_equal(rust_state_alpha_betas, pseudo_counts)

    emission.RUST_BACKEND = False

    bb_update, state_alpha_betas = emission.cna_mixture_betabinom_update(params)

    npt.assert_allclose(rust_bb_update, bb_update, rtol=1.0e-5, atol=1.0e-8)

    # NB all log probabilites should be <= 0
    assert np.all(bb_update <= 0.0)

    # NB >>>>>>  nbinom checks.
    emission.RUST_BACKEND = True

    rust_nb_update, rust_state_rs_ps = emission.cna_mixture_nbinom_update(params)

    assert np.array_equal(rust_state_rs_ps, rr_pp)

    emission.RUST_BACKEND = False

    nb_update, state_rs_ps = emission.cna_mixture_nbinom_update(params)

    npt.assert_allclose(rust_nb_update, nb_update, rtol=1.0e-5, atol=1.0e-8)

    # NB all log probabilites should be <= 0
    assert np.all(nb_update <= 0.0)

    # NB >>>>>>  beta-binomial grad checks.
    state_posteriors = np.ones(shape=(10_000, 1))

    emission.RUST_BACKEND = True
    rust_grad = emission.grad_em_cost(params, state_posteriors)

    emission.RUST_BACKEND = False
    grad = emission.grad_em_cost(params, state_posteriors)

    npt.assert_allclose(rust_grad, grad, rtol=1.0e-5, atol=1.0e-8)
