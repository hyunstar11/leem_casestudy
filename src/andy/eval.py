import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> tuple[float, float]:
    f1 = average_precision_score(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    return f1, auc

# tests/conftest.py
from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir(tmp_path: Path) -> Generator[Path, None, None]:
    yield tmp_path
