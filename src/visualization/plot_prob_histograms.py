import matplotlib.pyplot as plt
import numpy as np

def single_model_probablity_histogram(probs: np.ndarray, bins=25, alpha=0.6, label=None):

    fig, ax = plt.subplots(figsize=(6,5))
    # Expecting both SNGP and Base probabilities as input
    
    ax.hist(probs, bins=bins, alpha=alpha, label=label, density=False)

    ax.set_xlabel("Probability")
    ax.set_ylabel("Count")
    ax.set_title("Output logits distribution")
    ax.legend()
    return fig

