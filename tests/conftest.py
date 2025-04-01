import pytest
import numpy as np
from typing import Generator


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
    # Cleanup will be handled automatically by pytest
