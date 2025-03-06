from __future__ import annotations

import pytest
from cna_mixture.cna_sim import CNA_sim, CNA_transfer


@pytest.fixture(scope="module")
def cna_sim():
    cna_sim = CNA_sim()
    cna_sim.realize()
    
    return cna_sim
