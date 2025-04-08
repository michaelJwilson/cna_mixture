from __future__ import annotations

import numpy as np
import pytest
from cna_mixture.cna_mixture_params import CNA_mixture_params
from cna_mixture.cna_sim import CNA_sim


# NB scope defines the event for which a new instance is generated.
#    e.g. for every module, of every test function.
@pytest.fixture
def rng():
    return np.random.default_rng(314)


@pytest.fixture
def cna_sim():
    return CNA_sim(seed=314)


@pytest.fixture
def mixture_params():
    params = CNA_mixture_params(seed=314)
    params.initialize()

    return params


@pytest.fixture
def rdr_baf(rng):
    return 5 * (1.0 + rng.uniform(size=(3, 2)))


def pytest_addoption(parser):
    parser.addoption(
        "--run_slow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run_slow"):
        return

    skip_slow = pytest.mark.skip(reason="need --run_slow option to run")

    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
