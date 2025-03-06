from __future__ import annotations

import numpy as np

from cna_mixture.cna_mixture import CNA_mixture
from cna_mixture.cna_sim import CNA_sim, get_sim_params
from scipy.optimize import approx_fprime, check_grad

np.random.seed(314)


def test_cna_mixture_params(mixture_params):
    assert mixture_params.num_states == len(mixture_params.cna_states)
    assert np.array_equal(mixture_params.cna_states[0], np.array([1.0, 0.5]))

    sim_params = get_sim_params()

    print("\n", sim_params)
    print("\n", mixture_params)

    mixture_params.dict_update(sim_params)

    exp = np.array([[1.0, 0.5], [3.0, 0.33], [4.0, 0.25], [10.0, 0.1]])

    assert mixture_params.overdisp_tau == 45.0
    assert mixture_params.overdisp_phi == 0.01
    assert np.allclose(
        mixture_params.cna_states,
        exp,
        atol=1e-1,
    )

def test_em_cost_grad():
    cna_sim = CNA_sim()
    cna_model = CNA_mixture(cna_sim.realized_genome_coverage, cna_sim.data)

    grad = cna_model.jac(cna_model.initial_params)
    approx_grad = approx_fprime(
        cna_model.initial_params, cna_model.em_cost, np.sqrt(np.finfo(float).eps)
    )

    err = check_grad(cna_model.em_cost, cna_model.jac, cna_model.initial_params)

    # print()
    # print(grad)
    # print(approx_grad)
    # print(err)

    assert err < 0.51
