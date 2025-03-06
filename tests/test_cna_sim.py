from __future__ import annotations

import numpy as np

from cna_mixture.cna_mixture import CNA_mixture
from cna_mixture.cna_sim import CNA_sim, CNA_transfer
from scipy.optimize import approx_fprime, check_grad

np.random.seed(314)

def test_transfer_matrix():
    transfer_matrix = CNA_transfer(jump_rate=0.1, num_states=4).transfer_matrix

    assert transfer_matrix[0,1] == 0.1 / 3.
    assert np.array_equal(np.diag(transfer_matrix), 0.9 * np.ones(4))

    assert np.array_equal(transfer_matrix.sum(axis=0), np.ones(4))
    assert np.array_equal(transfer_matrix.sum(axis=1), np.ones(4))

