import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> tuple[float, float]:
    f1 = average_precision_score(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    return f1, auc
