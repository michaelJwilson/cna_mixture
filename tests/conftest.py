from __future__ import annotations

import pytest
from cna_mixture.cna_mixture_params import CNA_mixture_params
from cna_mixture.cna_sim import CNA_sim


# NB scope defines the event for which a new instance is generated.
#    e.g. for every module, of every test function.
@pytest.fixture
def cna_sim():
    cna_sim = CNA_sim()
    cna_sim.realize()

    return cna_sim


@pytest.fixture
def mixture_params():
    return CNA_mixture_params()
