import torch


def brier_score(probs: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    """Multiclass Brier score: mean squared error between predicted class
    probabilities and one-hot targets.

    Args:
        probs: Predicted probabilities, shape (n_samples, num_classes).
        targets: True class indices, shape (n_samples,).
        num_classes: Number of classes.

    Returns:
        Scalar Brier score. 0 is a perfect prediction; the multiclass upper
        bound is 2 (maximally confident and wrong).

    References:
        Brier, G. W. (1950). Verification of forecasts expressed in terms of
        probability. Monthly Weather Review.
    """
    one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes).float()
    return float(torch.mean(torch.sum((probs - one_hot) ** 2, dim=1)).item())
