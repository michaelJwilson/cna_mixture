from __future__ import annotations

import numpy as np
import pytest
from cna_mixture.cna_mixture_params import CNA_mixture_params
from cna_mixture.cna_mixture_initialize import CNA_mixture_initialize
from cna_mixture.cna_sim import get_sim_params


def test_initialize_nonnormal(mixture_params):
    initializer = CNA_mixture_initialize(mixture_params)    
    initializer.random()

    exp = np.array([[1.0, 0.5], [3.0, 0.33], [4.0, 0.25], [10.0, 0.1]])

    assert np.allclose(
        initializer.cna_states[1:, 0],
        exp,
        atol=1e-2,
    )

"""
def test_initialize_nonnormal(mixture_params, rdr_baf):
    mixture_params.initialize_random_nonnormal_rdr_baf(rdr_baf)

    # TODO exp changes whether the test is run individually, or all tests run.
    # NB matches rdr_baf realization
    exp = [5.88586121, 6.11213014, 9.22835512]

    assert np.allclose(
        mixture_params.cna_states[1:, 0],
        exp,
        atol=1e-2,
    )


def test_cna_mixture_params_reproducibility(mixture_params):
    new_params = CNA_mixture_params(seed=314)
    new_params.initialize()

    assert np.all(mixture_params.params == new_params.params)


def test_cna_mixture_params_seeding(mixture_params):
    new_params = CNA_mixture_params(seed=42)
    new_params.initialize()

    with pytest.raises(AssertionError):
        assert np.all(mixture_params.params == new_params.params)
"""
