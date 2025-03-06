from __future__ import annotations

import numpy as np

from cna_mixture.cna_mixture import CNA_mixture
from cna_mixture.cna_sim import CNA_sim
from scipy.optimize import approx_fprime, check_grad

np.random.seed(314)

def test_em_cost_grad():
    cna_sim = CNA_sim()
    cna_model = CNA_mixture(cna_sim.realized_genome_coverage, cna_sim.data)

    grad = cna_model.jac(cna_model.initial_params)
    approx_grad = approx_fprime(                                                                                                                                                                                                                         
        cna_model.initial_params, cna_model.em_cost, np.sqrt(np.finfo(float).eps)                                                                                                                                                                    
    )                                                                                                                                                                                                                                                    
    
    err = check_grad(                                                                                                                                                                                                                                    
        cna_model.em_cost, cna_model.jac, cna_model.initial_params                                                                                                                                                                                  
    )                                                                                                                                                                                                                                                    

    # print()
    # print(grad)
    # print(approx_grad)
    # print(err)
    
    assert err < 0.51

