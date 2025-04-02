from collections.abc import Generator

import numpy as np
import pytest

@pytest.fixture(scope="session")
def random_seed() -> int:
    """Set random seed for reproducibility."""
    seed = 42
    np.random.seed(seed)
    return seed

@pytest.fixture(scope="session")
def temp_dir(tmp_path_factory) -> Generator[str, None, None]:
    """Create a temporary directory for test files."""
    temp_dir = tmp_path_factory.mktemp("test_data")
    yield str(temp_dir)

def test_smoke():
    assert True
