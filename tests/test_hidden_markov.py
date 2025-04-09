from __future__ import annotations
import pytest
import numpy as np
import pylab as pl
import numpy.testing as npt
import matplotlib.pyplot as plt
from cna_mixture.hidden_markov import forward, backward, CNA_transfer, agnostic_transfer
from cna_mixture.state_priors import CNA_markov_prior
from cna_mixture.utils import uniform_ln_probs, cosine_similarity_origin
from scipy.special import logsumexp
from scipy.stats import norm


@pytest.fixture
def transfer_params():
    num_segments, num_states, jump_rate = 1_000, 2, 0.1
    return num_segments, num_states, jump_rate


@pytest.fixture
def transfer(transfer_params):
    num_segments, num_states, jump_rate = transfer_params
    return agnostic_transfer(num_states, jump_rate)


def test_hidden_markov_det(transfer):
    # NB transfer matrix is not a rotation matrix as the
    #    normalization constraint is on sum p, not p.p;
    det = np.linalg.det(transfer)
    inv = np.linalg.inv(transfer)


def test_hidden_markov_forward(transfer_params, transfer, plot=False):
    num_segments, num_states, jump_rate = transfer_params

    # NB correlation length definition  marginalizes over start.
    start_prior = np.array([0.6, 0.4])
    ln_start_prior = np.log(start_prior)
    ln_state_emission = np.zeros(shape=(num_segments, num_states))

    ln_fs = forward(ln_start_prior, transfer, ln_state_emission)
    sims = cosine_similarity_origin(np.exp(ln_fs))

    print(f"\n{np.exp(ln_fs)}")
    
    if plot:
        asymptotic = np.exp(-1.)
 
        # pl.axhline(asymptotic, c="k", lw=0.5)
        pl.plot(range(num_segments)[:50], sims[:50])
        
        pl.ylabel(r"Cosine similarity, $p \cdot p$")
        pl.xlabel("Jumps")
        plt.tight_layout()
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
