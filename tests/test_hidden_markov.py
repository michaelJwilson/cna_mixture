from __future__ import annotations
import numpy as np
import numpy.testing as npt
from cna_mixture.state_priors import CNA_categorical_prior, CNA_markov_prior
from scipy.special import logsumexp
from scipy.stats import norm


def hamming(first_states, second_states):
    return np.count_nonzero(first_states == second_states)


def transfers(states):
    interim = states != np.roll(states, 1)
    return np.count_nonzero(interim[:-1])


def test_CNA_markov_prior():
    num_segments, num_states, jump_rate = 5, 4, 1.0e-2

    markov_prior = CNA_markov_prior(
        num_segments=num_segments,
        num_states=num_states,
    )

    markov_prior.initialize(jump_rate=jump_rate)

    states = np.array([np.random.randint(0, num_states) for ii in range(num_segments)])
    samples = np.array([norm.rvs(loc=10 * ss, scale=1.0, size=1) for ss in states])

    ln_state_emission = np.hstack(
        [-0.5 * (samples - 10.0 * ii) ** 2.0 for ii in range(num_states)]
    )

    assert ln_state_emission.shape == (num_segments, num_states)

    ln_state_priors = markov_prior.get_ln_state_priors()
    ln_state_posteriors = markov_prior.get_ln_state_posteriors(ln_state_emission)

    decoded_states = np.argmax(ln_state_emission, axis=1)
    markov_decoded_states = np.argmax(ln_state_posteriors, axis=1)

    print("\n\n")

    print(
        f"Hamming distance and transfers for emission: {hamming(states, decoded_states)}, {transfers(decoded_states)}"
    )
    print(
        f"Hamming distance and transfers for Markov: {hamming(states, markov_decoded_states)}, {transfers(markov_decoded_states)}"
    )

    print(states)
    print(markov_decoded_states)

    print(markov_prior.transfer)

    markov_prior.update(ln_state_emission)

    print(markov_prior.transfer)
