from __future__ import annotations
import pytest
import numpy as np
import pylab as pl
import numpy.testing as npt
from cna_mixture.hidden_markov import forward, backward, CNA_transfer, agnostic_transfer
from cna_mixture.state_priors import CNA_markov_prior
from cna_mixture.utils import uniform_ln_probs, cosine_similarity_origin
from scipy.special import logsumexp
from scipy.stats import norm


@pytest.fixture
def transfer_params():
    num_segments, num_states, jump_rate = 1_000, 4, 1.e-2
    return num_segments, num_states, jump_rate


@pytest.fixture
def transfer(transfer_params):
    num_segments, num_states, jump_rate = transfer_params
    return agnostic_transfer(num_states, jump_rate)


def test_hidden_markov_det(transfer):
    det = np.linalg.det(transfer)
    inv = np.linalg.inv(transfer)

    print(f"\nDET:\n{det}")
    print(f"\nINV:\n{inv}")


def test_hidden_markov_forward(transfer_params, transfer):
    num_segments, num_states, jump_rate = transfer_params

    ln_start_prior = np.log(np.array([0.2, 0.4, 0.3, 0.1]))
    ln_state_emission = np.zeros(shape=(num_segments, num_states))

    result = forward(ln_start_prior, transfer, ln_state_emission)
    sims = cosine_similarity_origin(result)

    pl.plot(range(num_segments), sims)
    pl.show()
    
def test_hidden_markov_backward():
    num_segments, num_states, jump_rate = 1_000, 2, 0.1

    transfer = agnostic_transfer(num_states, jump_rate)
    ln_start_prior = np.log(np.array([0.1, 0.9]))
    ln_state_emission = np.zeros(shape=(num_segments, num_states))

    result = backward(ln_start_prior, transfer, ln_state_emission)

def test_hidden_markov_sample_hidden():
    num_segments, num_states, jump_rate = 1_000, 4, 1.0e-2

    markov_prior = CNA_markov_prior(
        num_segments=num_segments,
        num_states=num_states,
    )

    markov_prior.initialize(jump_rate=jump_rate)

    hidden = markov_prior.sample_hidden()

    assert len(hidden) == num_segments
