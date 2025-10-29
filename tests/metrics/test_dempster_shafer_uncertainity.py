import numpy as np
import pytest
from src.metrics.dempster_shafer_uncertainity import DempsterShaferUncertainty


class TestDempsterShaferUncertainty:
    """Test suite for Dempster-Shafer uncertainty calculation."""

    def test_basic_functionality(self):
        """Test basic functionality with simple inputs."""
        logits = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
        uncertainty = DempsterShaferUncertainty(logits)
        
        assert isinstance(uncertainty, np.ndarray)
        assert uncertainty.shape == (2,)
        assert np.all(uncertainty >= 0) and np.all(uncertainty <= 1)

    def test_single_sample(self):
        """Test with single sample input."""
        logits = np.array([[1.0, 2.0]])
        uncertainty = DempsterShaferUncertainty(logits)
        
        expected = 2 / (np.exp(1.0) + np.exp(2.0) + 2)
        assert uncertainty.shape == (1,)
        assert np.isclose(uncertainty[0], expected)

    def test_uniform_logits(self):
        """Test with uniform logits (maximum uncertainty case)."""
        logits = np.array([[0.0, 0.0, 0.0]])
        uncertainty = DempsterShaferUncertainty(logits)
        
        # For uniform logits: K / (K * exp(0) + K) = K / (K + K) = 0.5
        expected = 3 / (3 * 1.0 + 3)  # 3 / 6 = 0.5
        assert np.isclose(uncertainty[0], expected)

    def test_high_confidence_case(self):
        """Test with very high logits (low uncertainty case)."""
        logits = np.array([[100.0, 0.0, 0.0]])
        uncertainty = DempsterShaferUncertainty(logits)
        
        # Should be close to 0 due to large exp(100)
        assert uncertainty[0] < 0.01
        assert uncertainty[0] > 0

    def test_multiple_samples(self):
        """Test with multiple samples of varying uncertainty."""
        logits = np.array([
            [10.0, 0.0, 0.0],  # High confidence
            [0.0, 0.0, 0.0],   # Medium uncertainty
            [1.0, 1.0, 1.0]    # Medium uncertainty
        ])
        uncertainty = DempsterShaferUncertainty(logits)
        
        assert len(uncertainty) == 3
        # High confidence should have lower uncertainty
        assert uncertainty[0] < uncertainty[1]
        assert uncertainty[0] < uncertainty[2]

    def test_binary_classification(self):
        """Test with binary classification (2 classes)."""
        logits = np.array([[1.0, -1.0], [2.0, 2.0]])
        uncertainty = DempsterShaferUncertainty(logits)
        
        assert uncertainty.shape == (2,)
        assert np.all(uncertainty >= 0) and np.all(uncertainty <= 1)

    def test_many_classes(self):
        """Test with many classes."""
        num_classes = 1000
        logits = np.random.randn(5, num_classes)
        uncertainty = DempsterShaferUncertainty(logits)
        
        assert uncertainty.shape == (5,)
        assert np.all(uncertainty >= 0) and np.all(uncertainty <= 1)

    def test_edge_case_large_negative_logits(self):
        """Test with very large negative logits."""
        logits = np.array([[-100.0, -100.0, -100.0]])
        uncertainty = DempsterShaferUncertainty(logits)
        
        # Should approach 1 as exp(-100) ≈ 0
        expected = 3 / (3 * np.exp(-100.0) + 3)
        assert np.isclose(uncertainty[0], expected)
        assert uncertainty[0] > 0.99

    def test_list_input(self):
        """Test that function accepts list input."""
        logits_list = [[1.0, 2.0], [3.0, 4.0]]
        uncertainty = DempsterShaferUncertainty(logits_list)
        
        assert isinstance(uncertainty, np.ndarray)
        assert uncertainty.shape == (2,)

    def test_mathematical_properties(self):
        """Test mathematical properties of the uncertainty measure."""
        logits = np.array([[0.0, 1.0, 2.0]])
        uncertainty = DempsterShaferUncertainty(logits)
        
        # Verify the exact calculation
        num_classes = 3
        belief_mass = np.exp(0.0) + np.exp(1.0) + np.exp(2.0)
        expected = num_classes / (belief_mass + num_classes)
        
        assert np.isclose(uncertainty[0], expected, rtol=1e-10)

    def test_monotonicity(self):
        """Test that higher evidence leads to lower uncertainty."""
        # Increase evidence for first class
        logits1 = np.array([[1.0, 0.0, 0.0]])
        logits2 = np.array([[2.0, 0.0, 0.0]])
        logits3 = np.array([[3.0, 0.0, 0.0]])
        
        unc1 = DempsterShaferUncertainty(logits1)[0]
        unc2 = DempsterShaferUncertainty(logits2)[0]
        unc3 = DempsterShaferUncertainty(logits3)[0]
        
        # Higher logits should lead to lower uncertainty
        assert unc1 > unc2 > unc3

    # Error cases
    def test_invalid_dimensions(self):
        """Test error handling for invalid input dimensions."""
        with pytest.raises(ValueError, match="Expected 2D array"):
            DempsterShaferUncertainty(np.array([1, 2, 3]))
        
        with pytest.raises(ValueError, match="Expected 2D array"):
            DempsterShaferUncertainty(np.array([[[1, 2], [3, 4]]]))

    def test_nan_input(self):
        """Test error handling for NaN inputs."""
        logits = np.array([[1.0, np.nan, 2.0]])
        with pytest.raises(ValueError, match="NaN or infinite values"):
            DempsterShaferUncertainty(logits)

    def test_infinite_input(self):
        """Test error handling for infinite inputs."""
        logits = np.array([[1.0, np.inf, 2.0]])
        with pytest.raises(ValueError, match="NaN or infinite values"):
            DempsterShaferUncertainty(logits)

    def test_empty_input(self):
        """Test handling of edge case inputs."""
        logits = np.array([]).reshape(0, 3)
        uncertainty = DempsterShaferUncertainty(logits)
        assert uncertainty.shape == (0,)

    @pytest.mark.parametrize("num_classes", [2, 5, 10, 100])
    def test_different_class_numbers(self, num_classes):
        """Test with different numbers of classes."""
        logits = np.random.randn(10, num_classes)
        uncertainty = DempsterShaferUncertainty(logits)
        
        assert uncertainty.shape == (10,)
        assert np.all(uncertainty >= 0) and np.all(uncertainty <= 1)

    def test_reproducibility(self):
        """Test that results are reproducible."""
        np.random.seed(42)
        logits = np.random.randn(5, 3)
        
        unc1 = DempsterShaferUncertainty(logits)
        unc2 = DempsterShaferUncertainty(logits)
        
        np.testing.assert_array_equal(unc1, unc2)