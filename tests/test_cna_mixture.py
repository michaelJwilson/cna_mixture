from __future__ import annotations

import numpy as np
from cna_mixture.cna_inference import CNA_inference
from cna_mixture.cna_sim import CNA_sim, get_sim_params
from scipy.optimize import approx_fprime

np.random.seed(314)


def test_cna_mixture_params_dict_update(mixture_params):
    assert mixture_params.num_states == len(mixture_params.cna_states)
    assert np.array_equal(mixture_params.cna_states[0], np.array([1.0, 0.5]))

    mixture_params.dict_update(get_sim_params())

    exp = np.array([[1.0, 0.5], [3.0, 0.33], [4.0, 0.25], [10.0, 0.1]])

    assert mixture_params.overdisp_tau == 45.0
    assert mixture_params.overdisp_phi == 0.01
    assert np.allclose(
        mixture_params.cna_states,
        exp,
        atol=1e-1,
    )


def test_cna_mixture_params_rdr_baf_choice_update(mixture_params, rdr_baf):    
    mixture_params.rdr_baf_choice_update(rdr_baf)

    # TODO exp changes whether the test is run individually, or all tests run.
    # NB matches rdr_baf realization
    exp = [5.22208937, 8.13003172, 9.69945894]

    assert np.allclose(
        mixture_params.cna_states[1:, 0],
        exp,
        atol=1e-2,
    )


def test_em_cost_grad():
    cna_sim = CNA_sim()
    cna_model = CNA_inference(cna_sim.realized_genome_coverage, cna_sim.data)

    grad = cna_model.jac(cna_model.initial_params)
    approx_grad = approx_fprime(
        cna_model.initial_params, cna_model.em_cost, np.sqrt(np.finfo(float).eps)
    )
    
    assert np.allclose(approx_grad, grad, atol=3e-1)
