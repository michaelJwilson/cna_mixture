from __future__ import annotations

import numpy as np
from cna_mixture.state_priors import CNA_categorical_prior

np.random.seed(314)


def test_CNA_categorical_prior(mixture_params, rdr_baf):    
    state_prior = CNA_categorical_prior(mixture_params, rdr_baf)
    
    """
    assert np.allclose(
        mixture_params.cna_states,
        exp,
        atol=1e-1,
    )
    """
