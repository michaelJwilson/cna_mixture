from __future__ import annotations

import numpy as np
import pytest
from cna_mixture.cna_sim import CNA_sim, CNA_transfer

np.random.seed(314)


def reduce_by_state(states, values):
    """
    Calculate the per-state means of values, for aligned states,
    values arrays.
    """
    ustates, state_counts = np.unique(states, return_counts=True)
    result = np.zeros_like(ustates)

    for s, v in zip(states, values):
        result[int(s)] += v

    return result / state_counts


def test_transfer_matrix():
    transfer_matrix = CNA_transfer(jump_rate=0.1, num_states=4).transfer_matrix

    assert transfer_matrix[0, 1] == 0.1 / 3.0
    assert np.array_equal(np.diag(transfer_matrix), 0.9 * np.ones(4))

    assert np.array_equal(transfer_matrix.sum(axis=0), np.ones(4))
    assert np.array_equal(transfer_matrix.sum(axis=1), np.ones(4))


@pytest.mark.regression
def test_cna_sim_states(cna_sim):
    ustates, state_counts = np.unique(cna_sim.data["state"], return_counts=True)

    # NB approx. equal state distribution
    assert np.array_equal(state_counts, [2514, 2573, 2536, 2377])


def test_cna_sim_rdr_baf(cna_sim):
    state_rdrs = reduce_by_state(cna_sim.data["state"], cna_sim.rdr)
    state_bafs = reduce_by_state(cna_sim.data["state"], cna_sim.baf)

    assert np.allclose(state_rdrs, np.array([1.0, 3.0, 4.0, 10.0]), atol=1e-1)
    assert np.allclose(state_bafs, np.array([0.5, 0.33, 0.25, 0.1]), atol=1e-2)


def test_cna_sim_plot(cna_sim, tmp_path):
    cna_sim.plot_realization_true_flat(tmp_path)


def test_cna_sim_save_and_load(cna_sim, tmp_path):
    cna_sim.save(tmp_path)
    
    cna_sim = CNA_sim.load(tmp_path, 0)
    cna_sim.print()
    
def test_cna_sim_reproducibility(cna_sim):
    new_cna_sim = CNA_sim(seed=314)

    # NB tested test fails if cna_sim.data[0] is updated.
    assert np.all(new_cna_sim.data == cna_sim.data)

def test_cna_sim_seeding(cna_sim):
    new_cna_sim = CNA_sim(seed=42)
    
    with pytest.raises(AssertionError):
        assert np.all(new_cna_sim.data == cna_sim.data)
    
