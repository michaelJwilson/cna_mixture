from __future__ import annotations

import numpy as np
import pytest
from cna_mixture.cna_mixture_params import CNA_mixture_params
from cna_mixture.cna_sim import CNA_sim

# NB scope defines the event for which a new instance is generated.
#    e.g. for every module, of every test function.
@pytest.fixture
def cna_sim():
    return CNA_sim(seed=314)


@pytest.fixture
def mixture_params():
    params = CNA_mixture_params(seed=314)
    params.initialize()

    return params


@pytest.fixture
def rdr_baf():
    return 5 * (1.0 + np.random.uniform(size=(3, 2)))
