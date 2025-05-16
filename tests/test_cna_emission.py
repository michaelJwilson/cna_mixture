from __future__ import annotations

import pytest
import numpy as np
import numpy.testing as npt
from cna_mixture.cna_emission import (
    CNA_emission,
    cna_mixture_nbinom_eval,
    reparameterize_beta_binom,
    reparameterize_nbinom,
)
from cna_mixture_rs.core import CnaEmissionRs, nbinom_rs, betabinom_rs
from scipy.stats import betabinom, nbinom

np.random.seed(314)


@pytest.fixture()
def emission_params():
    # TODO
    num_states = 1

    # NB nbinom.rvs(lost_reads, dropout_rate, size=1)[0]
    rdrs, phi = 1.0 + np.arange(num_states), 1.0e-2
    bafs, tau = 0.2 * (1.0 + np.arange(num_states)), 10.0

    return rdrs, phi, bafs, tau


@pytest.fixture()
def emission(emission_params):
    num_states, normal_coverage, snp_coverage = 1, 100, 10
    rdrs, phi, bafs, tau = emission_params

    params = np.array([*rdrs, phi, *bafs, tau])

    # NB required for scipy sampling calls.
    assert num_states == 1

    # NB segment reads covering a snp < all segment reads.
    assert snp_coverage < normal_coverage
    assert len(rdrs) == num_states

    # NB rr == 1/phi, i.e. equivalent for all states.
    rr, pp = reparameterize_nbinom(normal_coverage * rdrs, phi)
    alphas, betas = reparameterize_beta_binom(bafs, tau)

    # NB (fairly) assumes each spot and each segment has the same normal coverage.
    ks = nbinom.rvs(rr, pp, size=10_000).astype(np.float64)
    xs = normal_coverage * np.ones_like(ks)

    bs = betabinom.rvs(snp_coverage, betas, alphas, size=10_000).astype(np.float64)

    ns = snp_coverage * np.ones_like(bs).astype(np.float64)

    return CNA_emission(num_states, ks, xs, bs, ns)


def test_emission_fixture(emission, emission_params):
    assert emission is not None
    assert emission_params is not None


# ----  rust backend tests  ----
def test_nbinom_rs(benchmark, emission, emission_params):
    rdrs, phi, _, _ = emission_params

    benchmark(lambda: nbinom_rs(emission.ks, emission.xs, rdrs, phi))


def test_betabinom_rs(benchmark, emission, emission_params):
    _, _, bafs, tau = emission_params
    alphas, betas = reparameterize_beta_binom(bafs, tau)

    benchmark(lambda: betabinom_rs(emission.bs, emission.ns, betas, alphas))

    
@pytest.mark.parametrize("compress", [True, False])
def test_cna_emission_rs_nb(benchmark, emission, emission_params, compress):
    rdrs, phi, _, _ = emission_params

    weights = np.random.uniform(size=(len(emission.ks), len(rdrs)))
    
    cna_em = CnaEmissionRs(
        emission.ks, emission.xs, emission.bs, emission.ns, weights, compress=compress
    )

    result = benchmark(lambda: cna_em.nbinom_reduce(rdrs, phi))

    if compress is False:
        exp = (weights * nbinom_rs(emission.ks, emission.xs, rdrs, phi)).sum()
        
        npt.assert_allclose(result, exp, rtol=1.0e-2, atol=1.0e-2)


# TODO reduce with compress
@pytest.mark.parametrize("compress", [True, False])
def test_cna_emission_rs_bb(benchmark, emission, emission_params, compress):
    _, _, bafs, tau = emission_params
    alphas, betas = reparameterize_beta_binom(bafs, tau)

    weights = np.random.uniform(size=(len(emission.ks), len(alphas)))
    
    cna_em = CnaEmissionRs(
        emission.ks, emission.xs, emission.bs, emission.ns, weights, compress=compress
    )
    
    # TODO accept bafs, dispersion
    result = benchmark(lambda: cna_em.betabinom_reduce(alphas, betas))

    if compress is False:
        base = betabinom_rs(emission.ks, emission.xs, alphas, betas)

        valid = np.isclose(cna_em.betabinom(alphas, betas), base, rtol=1.0e-2, atol=1.0e-2)

        print(valid.mean())
        
        # assert all(valid)
        
        exp = (weights * base).sum()

        # TODO fails.
        # npt.assert_allclose(result, exp, rtol=1.0e-2, atol=1.0e-2)


def test_CNA_emission_bb(emission, emission_params):
    rdrs, phi, bafs, tau = emission_params
    params = np.array([*rdrs, phi, *bafs, tau])

    unpacked = emission.unpack_params(emission_params)
    states_bag = emission.get_states_bag(emission_params)

    # NB (rdr, baf) for each state.
    assert np.array_equal(states_bag, np.array([[1.0, 0.2]]))
    assert unpacked == (rdrs, phi, bafs, tau)

    # NB >>>>>>  beta-binomial checks.
    rs_bb_update = emission.cna_mixture_betabinom_update(params)

    emission.RUST_BACKEND = False

    bb_update = emission.cna_mixture_betabinom_update(params)

    npt.assert_allclose(rs_bb_update, bb_update, rtol=1.0e-5, atol=1.0e-8)

    # NB all log probabilites should be <= 0
    assert np.all(bb_update <= 0.0)


def test_CNA_emission_nb(emission, emission_params):
    rdrs, phi, bafs, tau = emission_params
    params = np.array([*rdrs, phi, *bafs, tau])

    # NB >>>>>>  nbinom checks.
    emission.RUST_BACKEND = True

    rs_nb_update = emission.cna_mixture_nbinom_update(params)

    emission.RUST_BACKEND = False

    nb_update = emission.cna_mixture_nbinom_update(params)

    npt.assert_allclose(rs_nb_update, nb_update, rtol=1.0e-5, atol=1.0e-8)

    # NB all log probabilites should be <= 0
    assert np.all(nb_update <= 0.0)


@pytest.mark.skip(reason="TODO rework gradient calc.")
def test_CNA_emission_grad(emission, emission_params):
    rdrs, phi, bafs, tau = emission_params
    params = np.array([*rdrs, phi, *bafs, tau])

    # NB >>>>>>  beta-binomial grad checks.
    state_posteriors = np.ones(shape=(10_000, 1))

    emission.RUST_BACKEND = True
    rs_grad = emission.grad_em_cost(params, state_posteriors)

    emission.RUST_BACKEND = False
    grad = emission.grad_em_cost(params, state_posteriors)

    npt.assert_allclose(rust_grad, grad, rtol=1.0e-5, atol=1.0e-8)
