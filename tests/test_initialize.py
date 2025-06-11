from __future__ import annotations

import numpy as np
import pytest
from cna_mixture.cna_mixture_params import CNA_mixture_params
from cna_mixture.initialize import CNA_mixture_initialize
from cna_mixture.cna_sim import get_sim_params


def test_initialize_nonnormal(mixture_params):
    # NB num_cna_states: 3,  num_states: 4,  rdr_overdispersion: 0.02,  baf_overdispersion: 50.0,  normal_state: [1.  0.5],  cna_states: None
    initializer = CNA_mixture_initialize(mixture_params, mode="random")
    initializer.run()

    exp = np.array([[1.0, 0.5], [5.0, 1./5.], [7.0, 1./7.], [8.0, 1./8.]])

    assert np.allclose(
        initializer.params.cna_states,
        exp,
        atol=1e-2,
    )

def test_initialize_nonnormal(mixture_params, rdr_baf):
    initializer = CNA_mixture_initialize(mixture_params, mode="nonnormal")

    # TODO
    initializer.run(rdr_baf=rdr_baf)

    # TODO exp changes whether the test is run individually, or all tests run.
    # NB matches rdr_baf realization
    exp = [1., 5.92539798, 8.75376189, 9.065359]

    assert np.allclose(
        initializer.params.cna_states[:, 0],
        exp,
        atol=1e-2,
    )

@pytest.mark.skip(reason="TODO")
def test_cna_mixture_params_reproducibility(mixture_params):
    initializer = CNA_mixture_initialize(mixture_params, mode="random", seed=314)
    initializer.run()
    
    new_params = CNA_mixture_params(seed=314)
    new_params.initialize()

    assert np.all(mixture_params.params == new_params.params)

@pytest.mark.skip(reason="TODO")
def test_cna_mixture_params_seeding(mixture_params):
    new_params = CNA_mixture_params(seed=42)
    new_params.initialize()

    with pytest.raises(AssertionError):
        assert np.all(mixture_params.params == new_params.params)

