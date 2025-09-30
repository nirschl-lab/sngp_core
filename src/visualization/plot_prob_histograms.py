import matplotlib.pyplot as plt
import numpy as np

# def single_model_probablity_histogram(probs: np.ndarray, bins=25, alpha=0.6, label=None):

#     fig, ax = plt.subplots(figsize=(6,5))
#     # Expecting both SNGP and Base probabilities as input
    
#     ax.hist(probs, bins=bins, alpha=alpha, label=label, density=False)

#     ax.set_xlabel("Probability")
#     ax.set_ylabel("Count")
#     ax.set_title("Output logits distribution")
#     ax.legend()
#     return fig

def single_model_probability_histogram(probs: np.ndarray, bins=25, alpha=0.6, label=None):

    fig, ax = plt.subplots(figsize=(6,5))

    # Plot histogram normalized to 1
    counts, bins, patches = ax.hist(probs, bins=bins, alpha=alpha, label=label, density=True, edgecolor='black')

    # Convert density to percentage on y-axis
    # ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}%'.format(y*100)))

    # Force x-axis to stay in probability range
    ax.set_xlim(0, 1)

    ax.set_xlabel("Probability")
    ax.set_ylabel("Percentage")
    ax.set_title("Output logits distribution")
    ax.legend()
    return fig
