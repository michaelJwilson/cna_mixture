from __future__ import annotations

import pytest
import numpy as np
import numpy.testing as npt
from cna_mixture.cna_inference import CNA_inference
from cna_mixture.cna_sim import CNA_sim
from scipy.optimize import approx_fprime

np.random.seed(1234)

@pytest.mark.regression
def test_cna_inference():
    cna_sim = CNA_sim()
    cna_model = CNA_inference(cna_sim.num_states, cna_sim.realized_genome_coverage, cna_sim.data)

    res = cna_model.fit()
    params = cna_model.emission_model.unpack_params(res.x)

    exp = np.array([0.50015511, 0.28714097, 0.09091853, 0.10101394])
    bafs = params[2]
    
    npt.assert_allclose(bafs, exp, rtol=1.e-2, atol=2.3) 
    
def test_cna_inference_grad():
    cna_sim = CNA_sim()
    cna_model = CNA_inference(cna_sim.num_states, cna_sim.realized_genome_coverage, cna_sim.data)
    
    # NB  to be set by initialize method.
    assert not hasattr(cna_model, "ln_state_prior")
    assert not hasattr(cna_model, "ln_state_emission")
    assert not hasattr(cna_model, "ln_state_posteriors")
    assert not hasattr(cna_model, "state_posteriors")
    assert not hasattr(cna_model, "initial_params")

    with pytest.raises(AttributeError, match="'CNA_inference' object has no attribute 'state_posteriors'"):
        _ = cna_model.em_cost(np.zeros(2 + 2 * cna_sim.num_states))

    with pytest.raises(AttributeError, match="'CNA_inference' object has no attribute 'state_posteriors'"):
        _ = cna_model.jac(np.zeros(2 + 2 * cna_sim.num_states))

    cna_model.initialize()

    params = cna_model.initial_params
    
    cost = cna_model.em_cost(params)
    grad = cna_model.jac(params)
    approx_grad = approx_fprime(
        params, cna_model.em_cost, np.sqrt(np.finfo(float).eps)
    )
    
    npt.assert_allclose(approx_grad, grad, rtol=1.e-2, atol=2.3)    

