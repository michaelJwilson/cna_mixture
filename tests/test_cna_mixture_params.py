from __future__ import annotations

import numpy as np
import pytest
from cna_mixture.cna_mixture_params import CNA_mixture_params
from cna_mixture.cna_sim import get_sim_params


def test_cna_mixture_params_dict_update(mixture_params):
    assert mixture_params.num_states == 1 + mixture_params.num_cna_states
    
    mixture_params.dict_update(
        get_sim_params()
        | {
            "num_cna_states": 3,
        }
    )

    assert mixture_params.cna_states ==	None
    assert mixture_params.overdisp_tau == 45.0
    assert mixture_params.overdisp_phi == 0.01

    print(mixture_params.params)
