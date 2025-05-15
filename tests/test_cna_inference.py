from __future__ import annotations

import time
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import pylab as pl
import pytest
from cna_mixture.cna_inference import CNA_inference
from cna_mixture.cna_sim import CNA_sim
from scipy.optimize import approx_fprime


@pytest.mark.regression
def test_cna_inference(cna_sim):
    cna_inf = CNA_inference(
        cna_sim.num_states - 1,
        cna_sim.data,
        state_prior="categorical",
        initialize_mode="random",
        seed=np.random.default_rng(42),
    )

    cna_inf.initialize()

    res = benchmark(cna_inf.fit)
    params = cna_inf.emission_model.unpack_params(res.x)

    exp = [
        1.00002014e00,
        3.60385422e00,
        4.95173795e00,
        9.97756571e00,
        1.80162482e-02,
        5.02089621e-01,
        2.82880246e-01,
        1.66554174e-01,
        1.00492437e-01,
        4.98151885e01,
    ]
    
    npt.assert_allclose(res.x, exp, rtol=1.0e-2, atol=1.0e-2)


@pytest.mark.benchmark
def test_bench_cna_inference(cna_sim):
    cna_inf = CNA_inference(
        cna_sim.num_states,
        cna_sim.data,
        state_prior="categorical",
        initialize_mode="random",
        seed=np.random.default_rng(42),
    )

    # NB runtime of order XXs.
    start = time.perf_counter()
    
    cna_inf.initialize()

    end = time.perf_counter()

    runtime = end - start

    assert np.abs(runtime - 1.e-2) <= 0.01, f"{runtime - 1.e-2}"

    
@pytest.mark.skip(reason="TODO")
@pytest.mark.parametrize("state_prior", ["categorical", "markov"])
def test_cna_inference_pre_initialize(state_prior, cna_sim):
    cna_inf = CNA_inference(
        cna_sim.num_states,
        cna_sim.genome_coverage,
        cna_sim.data,
        state_prior=state_prior,
    )

    # NB  to be set by initialize method.
    assert not hasattr(cna_inf, "ln_state_prior")
    assert not hasattr(cna_inf, "ln_state_emission")
    assert not hasattr(cna_inf, "ln_state_posteriors")
    assert not hasattr(cna_inf, "state_posteriors")
    assert not hasattr(cna_inf, "initial_params")

    with pytest.raises(
        AttributeError,
        match="'CNA_inference' object has no attribute 'state_posteriors'",
    ):
        _ = cna_inf.em_cost(np.zeros(2 + 2 * cna_sim.num_states))

    with pytest.raises(
        AttributeError,
        match="'CNA_inference' object has no attribute 'state_posteriors'",
    ):
        _ = cna_inf.jac(np.zeros(2 + 2 * cna_sim.num_states))


@pytest.mark.skip(reason="TODO")
@pytest.mark.parametrize("state_prior", ["categorical", "markov"])
def test_cna_inference_grad(state_prior, cna_sim):
    cna_inf = CNA_inference(
        cna_sim.num_states,
        cna_sim.genome_coverage,
        cna_sim.data,
        state_prior=state_prior,
    )

    if state_prior == "categorical":
        cna_inf.initialize()
    else:
        cna_inf.initialize(jump_rate=0.1)

    params = cna_inf.initial_params

    cost = cna_inf.em_cost(params)
    grad = cna_inf.jac(params)
    approx_grad = approx_fprime(params, cna_inf.em_cost, np.sqrt(np.finfo(float).eps))

    if state_prior == "categorical":
        npt.assert_allclose(approx_grad, grad, rtol=1.0e-2, atol=2.3)
    else:
        # BUG? TODO?  numerical error or bug?  note: grad tau is correctly evalued;
        #             numerical is zeros => no dependence?  killed by posterior?
        npt.assert_allclose(approx_grad, grad, rtol=1.0, atol=7.7)


@pytest.mark.skip(reason="TODO")
@pytest.mark.slow
def test_cna_inference_mixture_initialize(num_trials=50):
    modes = ["random", "mixture_plusplus"]

    cna_sim = CNA_sim()
    result = []

    for initialize_mode in modes:
        cna_inf = CNA_inference(
            cna_sim.num_states,
            cna_sim.genome_coverage,
            cna_sim.data,
            initialize_mode=initialize_mode,
        )

        interim = []

        for initialization in range(num_trials):
            cna_inf.initialize()

            # NB initial objective
            # objective = cna_inf.em_cost(cna_inf.initial_params)

            # NB final objective
            objective = cna_inf.fit().fun

            interim.append(objective)

        result.append(interim)

    result = np.array(result).T

    invalid = np.any(np.isnan(result), axis=1)
    result = result[~invalid]

    print(f"\nFound {100. * invalid.mean()}% invalid with good results:\n{result}")

    mins = np.minimum.accumulate(result, axis=0)

    result = np.cumsum(result, axis=0)

    trials = 1 + np.arange(len(result))
    result = (result.T / trials).T

    pl.clf()

    # color_cycle = plt.gca().prop_cycler
    # colors = [item['color'] for item in color_cycle]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ii in range(result.shape[1]):
        pl.plot(trials, result[:, ii], alpha=0.25, c=colors[ii])
        pl.plot(trials, mins[:, ii], label=modes[ii], c=colors[ii])

    pl.ylim(100_000, 200_000)

    pl.xlabel("Initializations")
    pl.ylabel("EM cost")
    pl.legend(frameon=False)
    plt.tight_layout()
    pl.show()
