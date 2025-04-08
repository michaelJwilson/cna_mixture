import numpy as np
import numpy.testing as npt
from cna_mixture.utils import logmatexp


def test_logmatexp():
    transfer = np.diag(np.array([1, 2, 3], dtype=float))
    ln_probs = -np.log(3.0) * np.ones(3)

    exp = np.log(np.dot(transfer, np.ones(3) / 3.0))
    result = logmatexp(transfer, ln_probs)

    npt.assert_allclose(result, exp, rtol=1e-5, atol=1e-8)
