import numpy as np
import numpy.testing as npt
from cna_mixture.utils import logmatexp, assign_closest


def test_logmatexp():
    transfer = np.diag(np.array([1, 2, 3], dtype=float))
    ln_probs = -np.log(3.0) * np.ones(3)

    exp = np.log(np.dot(transfer, np.ones(3) / 3.0))
    result = logmatexp(transfer, ln_probs)

    npt.assert_allclose(result, exp, rtol=1e-5, atol=1e-8)


def test_assign_closest(rdr_baf):
    centers = [[1.0, 0.5], [5.5, 0.25], [10.75, 0.3]]

    # NB zero baseline_coverage
    rdr_baf[10,0] = np.inf
    
    idx = assign_closest(rdr_baf, centers)

    assert len(idx) == len(rdr_baf) - 1

