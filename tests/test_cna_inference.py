from __future__ import annotations

import numpy as np
from cna_mixture.cna_inference import CNA_inference
from cna_mixture.cna_sim import CNA_sim
from scipy.optimize import approx_fprime

np.random.seed(314)

def test_em_cost_grad():
    cna_sim = CNA_sim()
    cna_model = CNA_inference(cna_sim.realized_genome_coverage, cna_sim.data)

    grad = cna_model.jac(cna_model.initial_params)
    approx_grad = approx_fprime(
        cna_model.initial_params, cna_model.em_cost, np.sqrt(np.finfo(float).eps)
    )
    
    assert np.allclose(approx_grad, grad, atol=3e-1)
