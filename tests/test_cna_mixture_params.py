from __future__ import annotations

import numpy as np
import pytest
from cna_mixture.cna_mixture_params import CNA_mixture_params
from cna_mixture.cna_sim import get_sim_params


def test_initialization():
    params = CNA_mixture_params()

    assert params.num_cna_states == 3
    assert params.num_states == 4
    assert params.rdr_overdispersion == 2.0e-2
    assert params.baf_overdispersion == 50.0
    assert np.array_equal(params.normal_state, np.array([1.0, 0.5]))
    assert params.cna_states is None


def test_custom_initialization():
    params = CNA_mixture_params(
        num_cna_states=5, rdr_overdispersion=0.03, baf_overdispersion=60.0
    )

    assert params.num_cna_states == 5
    assert params.num_states == 6
    assert params.rdr_overdispersion == 0.03
    assert params.baf_overdispersion == 60.0


def test_dict_update():
    params = CNA_mixture_params()

    input_dict = {
        "num_cna_states": 4,
        "rdr_overdispersion": 0.04,
        "baf_overdispersion": 55.0,
        "cna_states": np.array([[1.0, 0.5], [2.0, 0.3], [3.0, 0.2], [4.0, 0.1]]),
    }

    params.dict_update(input_dict)

    assert params.num_cna_states == 4
    assert params.num_states == 5
    assert params.rdr_overdispersion == 0.04
    assert params.baf_overdispersion == 55.0
    assert np.array_equal(
        params.cna_states, np.array([[1.0, 0.5], [2.0, 0.3], [3.0, 0.2], [4.0, 0.1]])
    )


def test_params_property():
    params = CNA_mixture_params()
    params.cna_states = np.array([[1.0, 0.5], [2.0, 0.3], [3.0, 0.2]])
    
    expected_params = np.array([1.0, 2.0, 3.0, 0.02, 0.5, 0.3, 0.2, 50.0])
    
    assert np.array_equal(params.params, expected_params)


def test_str_representation():
    params = CNA_mixture_params(num_cna_states=4, rdr_overdispersion=0.03, baf_overdispersion=60.0)
    params.cna_states = np.array([[1.0, 0.5], [2.0, 0.3], [3.0, 0.2], [4.0, 0.1]])
    
    assert "num_cna_states: 4" in str(params)
    assert "rdr_overdispersion: 0.03" in str(params)
    assert "baf_overdispersion: 60.0" in str(params)
    assert "cna_states: [[1.  0.5]\n [2.  0.3]\n [3.  0.2]\n [4.  0.1]]" in str(params)
