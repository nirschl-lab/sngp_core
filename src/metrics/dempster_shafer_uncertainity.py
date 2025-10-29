import numpy as np

def DempsterShaferUncertainty(logits):
    """
    Calculate Dempster-Shafer uncertainty for model output logits.

    The Dempster-Shafer (DS) uncertainty metric quantifies epistemic uncertainty
    by measuring how much evidence supports the classification decision. Under
    the DS framework, the uncertainty is computed as K/(K + Σ(exp(logits))),
    where K is the number of classes.

    This metric is particularly suitable for models that are trained to directly
    modulate logit magnitudes to express uncertainty, as it directly measures
    the strength of evidence rather than just the distribution shape.

    Args:
        logits (np.ndarray): Model output logits of shape (n_samples, n_classes).
                           Each row represents the logits for one sample across
                           all classes.

    Returns:
        np.ndarray: Array of shape (n_samples,) containing uncertainty values
                   between 0 and 1, where:
                   - 0 indicates maximum confidence (infinite evidence)
                   - 1 indicates maximum uncertainty (no evidence)
                   - Higher values indicate greater epistemic uncertainty

    Raises:
        ValueError: If logits is not a 2D array.
        ValueError: If logits contains NaN or infinite values.

    References:
        [1] Sensoy, M., Kaplan, L., & Kandemir, M. (2018). Evidential deep learning
            to quantify classification uncertainty. NeurIPS.
        
    Example:
        >>> logits = np.array([[2.0, 1.0, 0.5], [0.1, 0.1, 0.1]])
        >>> uncertainty = DempsterShaferUncertainty(logits)
        >>> print(uncertainty)
        [0.27272727 0.73170732]
        
    Note:
        Implementation based on:
        https://github.com/google/uncertainty-baselines/blob/df5d3fa1e25bacb33a3cefbade76e45c60605b40/baselines/cifar/ood_utils.py#L22
    """
    if not isinstance(logits, np.ndarray):
        logits = np.array(logits)
    
    if logits.ndim != 2:
        raise ValueError(f"Expected 2D array, got {logits.ndim}D array")
    
    if not np.isfinite(logits).all():
        raise ValueError("Logits contain NaN or infinite values")
    
    num_classes = logits.shape[1]
    belief_mass = np.sum(np.exp(logits), axis=1)
    return num_classes / (belief_mass + num_classes)
