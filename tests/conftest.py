from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def data_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="session")
def model_path() -> Path:
    return Path(__file__).resolve().parents[1] / "models"
