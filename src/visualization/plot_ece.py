import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize
from typing import Optional
import pdb

def compute_ece(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(y_prob)
    for i in range(n_bins):
        # Left-inclusive for all bins, right-inclusive for the last bin.
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i+1])
        else:
            in_bin = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i+1])
        bin_count = np.sum(in_bin)
        if bin_count > 0:
            avg_confidence = np.mean(y_prob[in_bin])
            avg_accuracy = np.mean(y_true[in_bin])
            ece += np.abs(avg_confidence - avg_accuracy) * bin_count / total
    return ece

def plot_calibration_curve(preds, targets, num_classes, n_bins=10, image_classes=None):

    '''
    preds = (N, n_classes)
    labels = (n, )

    '''
    if not image_classes:
        image_classes = [i for i in range(num_classes)]

    targets = label_binarize(targets, classes=list(range(num_classes))) # (N, n_classes)

    fig = plt.figure(figsize=(6, 6))
    for i in range(preds.shape[1]):
        prob_true, prob_pred = calibration_curve(targets[:, i], preds[:, i], n_bins=n_bins)
        plt.plot(prob_pred, prob_true, marker='o', label=f'{image_classes[i]}')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    plt.xlabel('Mean predicted value')
    plt.ylabel('Accuracy')
    plt.title('Calibration Curves')
    plt.legend()
    plt.xticks(np.linspace(0, 1, n_bins + 1))
    plt.yticks(np.linspace(0, 1, n_bins + 1))
    fig.tight_layout()
    return fig

def plot_class_combined_calibration_curve(preds, targets, num_classes, n_bins=10, image_classes=None):

    '''
    preds = (N, n_classes)
    labels = (n, )

    '''
    if not image_classes:
        image_classes = [i for i in range(num_classes)]

    targets = label_binarize(targets, classes=list(range(num_classes))) # (N, n_classes)

    fig = plt.figure(figsize=(6, 6))
    for i in range(preds.shape[1]):
        prob_true, prob_pred = calibration_curve(targets[:, i], preds[:, i], n_bins=n_bins)
        plt.plot(prob_pred, prob_true, marker='o', label=f'{image_classes[i]}')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    plt.xlabel('Mean predicted value')
    plt.ylabel('Accuracy')
    plt.title('Calibration Curves')
    plt.legend()
    plt.xticks(np.linspace(0, 1, n_bins + 1))
    plt.yticks(np.linspace(0, 1, n_bins + 1))
    fig.tight_layout()
    return fig

if __name__ == '__main__':
    # Generate synthetic test data for demonstration
    np.random.seed(42)
    
    # Parameters
    n_samples = 1000
    n_classes = 3
    class_names = ['Class A', 'Class B', 'Class C']
    
    # Generate ground truth labels
    targets = np.random.choice(n_classes, size=n_samples)
    
    # Generate predictions with varying calibration quality
    # Class 0: Well-calibrated
    # Class 1: Over-confident 
    # Class 2: Under-confident
    preds = np.zeros((n_samples, n_classes))
    
    for i, true_class in enumerate(targets):
        # Start with one-hot encoding and add noise
        base_pred = np.zeros(n_classes)
        base_pred[true_class] = 0.8
        base_pred += np.random.dirichlet([1, 1, 1]) * 0.2
        
        # Apply calibration effects
        if true_class == 0:  # Well calibrated
            preds[i] = base_pred
        elif true_class == 1:  # Over-confident
            preds[i] = base_pred ** 0.5  # Make predictions more extreme
        else:  # Under-confident  
            preds[i] = base_pred ** 2  # Make predictions less extreme
        
        # Normalize to ensure probabilities sum to 1
        preds[i] = preds[i] / np.sum(preds[i])
    
    # Test ECE computation for each class
    print("Expected Calibration Error (ECE) for each class:")
    for class_idx in range(n_classes):
        # Convert to binary classification problem for ECE
        binary_targets = (targets == class_idx).astype(int)
        binary_preds = preds[:, class_idx]
        ece = compute_ece(binary_targets, binary_preds)
        print(f"{class_names[class_idx]}: {ece:.4f}")
    
    # Create and display calibration plot
    fig = plot_calibration_curve(preds, targets, n_classes, n_bins=10, image_classes=class_names)
    plt.show()
    
    # Save the plot
    fig.savefig('test_plots/calibration_curve_test.png', dpi=300, bbox_inches='tight')
    print("\nCalibration curve saved as 'calibration_curve_test.png'")
    