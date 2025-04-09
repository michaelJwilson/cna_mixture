from __future__ import annotations

import numpy as np
import pylab as pl
import numpy.testing as npt
from collections import Counter
from numba import njit
from cna_mixture.state_priors import CNA_categorical_prior, CNA_markov_prior
from scipy.special import logsumexp
from scipy.stats import norm


def test_CNA_categorical_prior(mixture_params, rdr_baf):
    equal_priors = CNA_categorical_prior(10, mixture_params.num_states)
    equal_priors.ln_lambdas_equal()

    state_priors = CNA_categorical_prior(10, mixture_params.num_states)
    state_priors.ln_lambdas_closest(rdr_baf, mixture_params.cna_states)

    assert logsumexp(equal_priors.ln_lambdas) == 0.0
    assert len(state_priors.ln_lambdas) == len(mixture_params.cna_states)
    assert np.abs(logsumexp(state_priors.ln_lambdas)) < 1.5e-16

    # NB rdr_baf fixture does not populate the first state when assigned closest.
    assert state_priors.ln_lambdas[0] == -np.inf

    ln_state_posteriors = np.log(np.array([[0.25, 0.25, 0.1, 0.4]]))

    state_priors.update(ln_state_posteriors)

    npt.assert_allclose(
        ln_state_posteriors[0], state_priors.ln_lambdas, rtol=1e-5, atol=1e-8
    )

    assert np.abs(logsumexp(state_priors.ln_lambdas)) < 1.5e-16

    ln_state_priors = state_priors.get_ln_state_priors()

    npt.assert_allclose(
        np.tile(ln_state_posteriors, (10, 1)), ln_state_priors, rtol=1e-5, atol=1e-8
    )


def hamming(first_states, second_states):
    return np.count_nonzero(first_states == second_states)

@njit
def transfers(num_states, states):
    mat = np.zeros(shape=(num_states, num_states))

    for i,j in zip(states[:-1], states[1:]):
        mat[i,j] += 1
        
    return mat


def test_CNA_markov_prior(plot=True):
    scale = 10.
    num_segments, num_states, jump_rate = 5_000, 4, 1.0e-20
    
    markov_prior = CNA_markov_prior(
        num_segments=num_segments,
        num_states=num_states,
    )

    markov_prior.initialize(jump_rate=jump_rate)

    # NB states have zero correlation length.
    states = np.array([np.random.randint(0, num_states) for ii in range(num_segments)])
    samples = np.array([norm.rvs(loc=10 * ss, scale=scale, size=1) for ss in states])

    ln_state_emission = np.hstack(
        [-0.5 * ((samples - 10.0 * ii) / scale) ** 2.0 for ii in range(num_states)]
    )

    assert ln_state_emission.shape == (num_segments, num_states)

    ln_state_priors = markov_prior.get_ln_state_priors(ln_state_emission=ln_state_emission)
    ln_state_posteriors = markov_prior.get_ln_state_posteriors(ln_state_emission=ln_state_emission)

    decoded_states = np.argmax(ln_state_emission, axis=1)

    markov_prior_states = np.argmax(ln_state_priors, axis=1)
    markov_decoded_states = np.argmax(ln_state_posteriors, axis=1)

    _, true_state_counts = np.unique(states, return_counts=True)

    pl.plot(range(num_segments), markov_decoded_states, marker=',', c="k", lw=0.0)
    pl.ylim(-0.5, num_states + 0.5)
    pl.show()
    
    """
    print("\n\n")

    print(f"True state distribution: {true_state_counts}")
    
    print(
        f"Hamming distance and transfers for emission only: {hamming(states, decoded_states)},\n{transfers(num_states, decoded_states)}"
    )

    print()

    print(
        f"Hamming distance and transfers for Markov prior only: {hamming(states, markov_prior_states)},\n{transfers(num_states, markov_prior_states)}"
    )

    print()
    
    print(
        f"Hamming distance and transfers for Markov prior & emission: {hamming(states, markov_decoded_states)},\n{transfers(num_states, markov_decoded_states)}"
    )
    
    print(states)
    print(markov_decoded_states)

    print(markov_prior.transfer)

    markov_prior.update(ln_state_emission=ln_state_emission)

    print(markov_prior.transfer)
    """
