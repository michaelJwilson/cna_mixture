from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def none_fixture():
    return None
