import matplotlib.pyplot as plt
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def plot_pr_curve(y_true, y_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)

    plt.plot(recall, precision, label=f"PRAUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.show()
    return pr_auc

def print_scores(y_true, y_pred, y_proba):
    f1 = f1_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_proba)
    print(f"F1 Score: {f1:.4f}")
    print(f"Average Precision Score: {ap:.4f}")
    return f1, ap

def evaluate_model(y_true, y_pred_proba, threshold=0.5, plot_pr=True):
    """
    Evaluate classification performance using ROC AUC, F1, and PR AUC.

    Args:
        y_true (array-like): Ground truth binary labels.
        y_pred_proba (array-like): Predicted probabilities for the positive class.
        threshold (float): Threshold to convert probabilities to binary predictions.
        plot_pr (bool): Whether to show the Precision-Recall curve.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    f1 = f1_score(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)

    print("📊 Evaluation Metrics:")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_true, y_pred))

    if plot_pr:
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, marker='.', label='PR Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        plt.legend()
        plt.show()

    return {
        "roc_auc": roc_auc,
        "f1_score": f1,
        "pr_auc": pr_auc
    }
