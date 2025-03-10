from __future__ import annotations

import pytest
import numpy as np
import numpy.testing as npt
from cna_mixture.state_priors import CNA_categorical_prior, CNA_markov_prior
from scipy.special import logsumexp
from scipy.stats import norm

np.random.seed(314)


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

def transfers(states):
    interim = states != np.roll(states, 1)
    return np.count_nonzero(interim[:-1])
    
def test_CNA_markov_prior():
    num_segments, num_states, jump_rate = 1_000, 4, 1.e-2

    markov_prior = CNA_markov_prior(
        num_segments=num_segments,
        jump_rate=jump_rate,
        num_states=num_states,
    )

    states = np.array([np.random.randint(0, num_states) for ii in range(num_segments)])
    samples = np.array([norm.rvs(loc=10 * ss, scale=10., size=1) for ss in states])
    
    ln_state_emission = np.hstack([-0.5 * (samples - 10. * ii)**2. for ii in range(num_states)])
    
    assert ln_state_emission.shape == (num_segments, num_states)

    # print("\n", ln_state_emission)
    # markov_prior.update(ln_state_emission)
    
    state_priors = markov_prior.get_ln_state_priors()
    state_posteriors = markov_prior.get_ln_state_posteriors(ln_state_emission)
    
    # print("\n", state_priors)

    decoded_states = np.argmax(ln_state_emission, axis=1)
    markov_decoded_states = np.argmax(state_posteriors, axis=1)
    
    print("\n\n")
    
    print(f"Hamming distance and transfers for emission: {hamming(states, decoded_states)}, {transfers(decoded_states)}")
    print(f"Hamming distance and transfers for Markov: {hamming(states, markov_decoded_states)}, {transfers(markov_decoded_states)}")

    # TODO
    with pytest.raises(NotImplementedError):
        _ = markov_prior.update()
    
