from __future__ import annotations

import numpy as np
import numpy.testing as npt
from cna_mixture.state_priors import CNA_categorical_prior, CNA_markov_prior
from scipy.special import logsumexp
from scipy.stats import norm

np.random.seed(314)


def test_CNA_categorical_prior(mixture_params, rdr_baf):
    equal_priors = CNA_categorical_prior.ln_lambdas_equal(5)
    state_priors = CNA_categorical_prior(mixture_params, rdr_baf)
    
    assert logsumexp(equal_priors) == 0.
    assert len(state_priors.ln_lambdas) == len(mixture_params.cna_states)
    assert np.abs(logsumexp(state_priors.ln_lambdas)) < 1.5e-16

    # NB rdr_baf fixture does not populate the first state when assigned closest.
    assert state_priors.ln_lambdas[0] == -np.inf

    ln_state_posteriors = np.log(np.array([[0.25, 0.25, 0.1, 0.4]]))

    state_priors.update(ln_state_posteriors)
    
    npt.assert_allclose(ln_state_posteriors[0], state_priors.ln_lambdas, rtol=1e-5, atol=1e-8)

    assert np.abs(logsumexp(state_priors.ln_lambdas)) < 1.5e-16
    
    state_priors = state_priors.get_state_priors(5)

    npt.assert_allclose(np.tile(ln_state_posteriors, (5, 1)), state_priors, rtol=1e-5, atol=1e-8)


def test_CNA_markov_prior():
    markov_prior = CNA_markov_prior(num_segments=10, jump_rate=0.1, num_states=4, ln_start_prior=None)

    xs = norm.rvs(loc=0., scale=1., size=10, random_state=None)
    ln_state_emission = np.tile(norm.logpdf(xs, loc=0., scale=1.), (4, 1)).T
    
    markov_prior.forward(ln_state_emission)

    print("\n", markov_prior.ln_fs)
