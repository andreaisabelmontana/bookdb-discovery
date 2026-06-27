import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from discovery import DiscoveryEngine, load_catalog, load_ratings  # noqa: E402


@pytest.fixture(scope="session")
def catalog():
    return load_catalog()


@pytest.fixture(scope="session")
def ratings():
    return load_ratings()


@pytest.fixture(scope="session")
def engine():
    # alpha=0.6 is the tuned operating point used throughout (see evaluate.py).
    return DiscoveryEngine.from_data(alpha=0.6)
