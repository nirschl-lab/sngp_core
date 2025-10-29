import numpy as np
import matplotlib.pyplot as plt
from src.metrics.dempster_shafer_uncertainity import DempsterShaferUncertainty

def DempsterShaferUncertaintyPlot(logits: np.ndarray) -> np.ndarray:
    """Generates a Dempster-Shafer uncertainty plot from model logits.

    Args:
        logits: n_samples x c array of model output logits.
    Returns:
        Matplotlib figure containing the Dempster-Shafer uncertainty plot.
    """
    ds_uncertainty = DempsterShaferUncertainty(logits)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(
        ds_uncertainty,
        bins=25,
        alpha=0.7,
        edgecolor="black",
        weights=np.ones_like(ds_uncertainty) / len(ds_uncertainty),
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))

    ax.set_xlabel("Dempster-Shafer Uncertainty")
    ax.set_ylabel("Percentage")
    ax.set_title("Dempster-Shafer Uncertainty Distribution")
    return fig

if __name__ == "__main__":
    # Example usage
    sample_logits = np.array([[2.0, 1.0, 0.5], [0.1, 0.2, 0.3], [1.5, 2.5, 0.5]])
    fig = DempsterShaferUncertaintyPlot(sample_logits)
    plt.show()
    
