from __future__ import annotations
import numpy as np
import numpy.testing as npt
from cna_mixture.hidden_markov import forward, backward, CNA_transfer
from cna_mixture.state_priors import CNA_markov_prior
from scipy.special import logsumexp
from scipy.stats import norm


def test_hidden_markov_forward():
    num_segments, num_states = 100, 5

    ln_start_prior = 
    
    ln_start_prior, transfer, ln_state_emission
    

    
    
def test_hidden_markov_sample_hidden():
    num_segments, num_states, jump_rate = 1_000, 4, 1.0e-2

    markov_prior = CNA_markov_prior(
        num_segments=num_segments,
        num_states=num_states,
    )

    markov_prior.initialize(jump_rate=jump_rate)

    hidden = markov_prior.sample_hidden()

    assert len(hidden) == num_segments

