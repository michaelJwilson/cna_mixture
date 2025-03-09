import numpy as np
import numpy.testing as npt

from cna_mixture.utils import logmatexp

def test_logmatexp():
    transfer = np.diag([1, 2, 3])
    ln_probs = -np.log(3.) * np.ones(3)

    exp = np.log(np.dot(transfer, np.ones(3) / 3.))
    result = logmatexp(transfer, ln_probs)

    npt.assert_allclose(result, exp, rtol=1e-5, atol=1e-8)
    
