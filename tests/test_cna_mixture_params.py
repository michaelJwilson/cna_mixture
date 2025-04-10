from __future__ import annotations

import numpy as np
import pytest
from cna_mixture.cna_mixture_params import CNA_mixture_params
from cna_mixture.cna_sim import get_sim_params


def test_cna_mixture_params_dict_update(mixture_params):
    assert mixture_params.num_states == len(mixture_params.cna_states)
    assert np.array_equal(mixture_params.cna_states[0], np.array([1.0, 0.5]))

    # TODO HACK?
    mixture_params.dict_update(
        get_sim_params()
        | {
            "genome_coverage": 500,
            "num_cna_states": 3,
            "seed": 42,
            "rng": np.random.default_rng(42),
        }
    )

    exp = np.array([[1.0, 0.5], [3.0, 0.33], [4.0, 0.25], [10.0, 0.1]])

    assert mixture_params.overdisp_tau == 45.0
    assert mixture_params.overdisp_phi == 0.01
    assert np.allclose(
        mixture_params.cna_states,
        exp,
        atol=1e-1,
    )


def test_initialize_random_nonnormal_rdr_baf(mixture_params, rdr_baf):
    mixture_params.initialize_random_nonnormal_rdr_baf(rdr_baf)

    # TODO exp changes whether the test is run individually, or all tests run.
    # NB matches rdr_baf realization
    exp = [5.64958905, 8.76022735, 9.61271087]

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
