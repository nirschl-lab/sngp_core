import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

def plot_roc_curve(probs: np.ndarray, targets: np.ndarray, num_classes: int):
    """
    Create an ROC curve matplotlib figure.
    
    Args:
        probs (np.ndarray): Array of predicted probabilities (N, C) or (N,) for binary.
        targets (np.ndarray): Array of true labels (N,).
        num_classes (int): Number of classes.

    Returns:
        matplotlib.figure.Figure: The ROC curve figure.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    if num_classes == 1 or probs.ndim == 1:  # binary
        p = probs if probs.ndim == 1 else probs.squeeze(1)
        fpr, tpr, _ = roc_curve(targets, p)
        ax.plot(fpr, tpr, label=f"AUC={auc(fpr,tpr):.3f}")
    else:  # multiclass
        y_bin = label_binarize(targets, classes=list(range(num_classes)))
        for k in range(num_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, k], probs[:, k])
            ax.plot(fpr, tpr, label=f"class {k} AUC={auc(fpr,tpr):.3f}")
        ax.plot([0, 1], [0, 1], "--", linewidth=1)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig