#!/usr/bin/env python3
"""gaussian_process.py in src/sngp_core/models/sngp.

Adapted from:
https://github.com/Jmkernes/Spectral-Normalized-Gaussian-Process/blob/main/gaussian_process.py
https://github.com/google/edward2/blob/main/edward2/tensorflow/layers/gaussian_process.py
https://github.com/google/edward2/blob/main/edward2/tensorflow/layers/random_feature.py
"""


import math
import os
from functools import partial
from typing import Optional

import torch
from loguru import logger
from torch import nn

from src.models.sngp.random_fourier_features import RandomFourierFeatures

_SUPPORTED_LIKELIHOOD = ("binary_logistic", "multiclass_logistic", "poisson", "gaussian")
# _SUPPORTED_RBF_KERNEL_TYPES = ["gaussian", "laplacian"]


# TODO: Currently not used, consider refactoring to use or removing
# CustomRandomFeatureLayer is unused
class CustomRandomFeatureLayer(nn.Module):
    """
    Allows users to input custom functions to simulate a kernel.
    The idea is to approximate a kernel function K(x_i, x_j) via a decomposition
    K_ij = phi_i @ phi_j, given some nonlinearity phi_i(x). If phi is a probabilistic
    mapping with proper statistics, we can approximate K_ij for things like RBF Gaussians.

    This class allows the user to define the phi function

    Args:
        in_features: input feature dimension, x.shape[-1]
        out_featuers: size of random features phi.shape[-1]
        kernel_init: a torch.nn.init object, defaults to torch.nn.init.normal_
        bias_init: a torch.nn.init object, defaults to torch.nn.init.uniform_
        activation: a torch.nn.init object, defaults to cosine
    Returns:
        phi: a tensor of shape (..., random_feature_dimension)
    """

    def __init__(
        self,
        in_features,
        out_features,
        kernel_init=None,
        bias_init=None,
        activation=None,
    ):
        super(CustomRandomFeatureLayer, self).__init__()
        if kernel_init is None:
            kernel_init = nn.init.normal_
        if bias_init is None:
            bias_init = partial(nn.init.uniform_, a=0, b=2 * math.pi)
        if activation is None:
            activation = torch.cos

        self.out_features = out_features
        self.weight = kernel_init(
            nn.Parameter(torch.empty(in_features, out_features), requires_grad=False)
        )
        self.bias = bias_init(
            nn.Parameter(torch.empty(out_features), requires_grad=False)
        )
        self.activation = activation

    def forward(self, x):
        return self.activation(x @ self.weight + self.bias)


class RandomFeatureGaussianProcess(nn.Module):
    """Gaussian process layer using random feature approximation for uncertainty-aware predictions.

    This layer combines random feature mappings with a Gaussian process posterior to provide predictive uncertainty estimates.
    It supports flexible kernel choices, input normalization, and covariance estimation, and is adapted from the TensorFlow implementation referenced below.

    During training, the layer updates the maximum a posteriori (MAP) logits and the posterior precision matrix using minibatch statistics.
    During inference, it adjusts the MAP logits by the predictive standard deviation, approximating the posterior mean via a mean-field approach.

    Users can specify different types of random features by enabling `use_custom_features` and providing custom initializers and activations.
    For example:
        - MLP Kernel: initializer='random_normal', activation=torch.nn.ReLU
        - RBF Kernel: initializer='random_normal', activation=torch.cos

    A linear kernel can also be specified by setting `kernel_type='linear'` and `use_custom_features=True`.

    References:
        [1] Ali Rahimi and Benjamin Recht. Random Features for Large-Scale Kernel Machines. NeurIPS, 2007.
            https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf
        [2] TensorFlow Probability Gaussian Process implementation: https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/distributions/gaussian_process.py

    Attributes:
        in_features: (int) Number of input features.
        out_features: (int) Number of output features (e.g., number of classes).
        random_features: (int) Number of random features for kernel approximation.
        kernel_type: (str) Type of kernel function used.
        kernel_scale: (float) Length-scale parameter for the kernel.
        scale_random_features: (bool) Whether to scale random features by sqrt(2. / random_features).
        normalize_input: (bool) Whether to apply layer normalization to the input.
        trainable_kernel_scale: (bool) If True, kernel scale is trainable.
        use_custom_features: (bool) If True, use a custom feature mapping.
        covariance_momentum: (float) Momentum for updating the covariance estimator.
        covariance_ridge_penalty: (float) Ridge penalty for covariance regularization.
        return_covariance: (bool) If True, returns covariance estimates.
        init_stdev: (float) Standard deviation for output layer weight initialization.
        output_bias_trainable: (bool) If True, output bias is trainable.
        covariance_likelihood: (str) Likelihood type for covariance estimation.
        verbose: (bool) If True, enables verbose logging.

        The scale_random_features parameter scales Phi by 2. / sqrt(num_inducing) following [1].
        When using this GP layer as the output layer of a neural network, it is recommended to turn this scaling off to prevent it from changing the learning rate to the hidden layers.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        random_features: int = 1024,
        covariance_likelihood="gaussian",
        covariance_momentum: float = 0.999,
        covariance_ridge_penalty: float = 1e-6,
        init_stdev: float = 1e-2,
        kernel_scale: Optional[float] = None, # set based on kernel_type if None
        kernel_type: str = "gaussian",  #
        normalize_input: bool = True,
        output_bias_trainable: bool = False,
        return_covariance: bool = True,
        return_features: bool = False,
        scale_random_features=False, # double check the default value in TF/Edward2
        trainable_kernel_scale: bool = False,
        use_custom_features: bool = False,
        verbose: bool = False,
    ):
        """Initializes a Gaussian Process layer using random feature approximation.

        This constructor sets up the random feature mapping, output projection, and covariance estimator for the layer.
        It supports various kernel types, normalization, and options for trainable parameters.

        Args:
            in_features: Number of input features.
            out_features: Number of output features (e.g., number of classes for GP layer).
            random_features: Number of random Fourier features for kernel approximation.
            kernel_type: Type of kernel function to use ('gaussian', 'linear', etc.).
            kernel_scale: Length-scale parameter for the kernel function.
            scale_random_features: Whether to scale the random features.
            normalize_input: Whether to apply layer normalization to the input.
            trainable_kernel_scale: If True, kernel scale is a trainable parameter.
            use_custom_features: If True, use a custom linear feature mapping.
            covariance_momentum: Momentum for updating the covariance estimator.
            covariance_ridge_penalty: Ridge penalty for covariance regularization.
            return_covariance: If True, the layer returns covariance estimates.
            init_stdev: Standard deviation for output layer weight initialization.
            output_bias_trainable: If True, output bias is trainable.
            covariance_likelihood: Likelihood type for covariance estimation.
            verbose: If True, enables verbose logging.

        Returns:
            None.
        """
        super().__init__()
        verbose = bool(verbose or os.getenv("VERBOSE") or os.getenv("DEBUG"))
        self.verbose = verbose
        self.in_features = in_features
        self.out_features = out_features # number of classes
        self.random_features = random_features

        # set kernel type and scale
        self.kernel_type = kernel_type.lower()
        if kernel_type.lower() not in ("gaussian", "laplacian", "linear"):
            raise ValueError(f"Unsupported kernel_type: {kernel_type}")

        if kernel_scale is None:
            if kernel_type.lower() == "gaussian" and not normalize_input:
                # The length-scale parameter of the shift-invariant kernel function.
                # Heuristic: scale by sqrt(d/2) when inputs are unnormalized
                kernel_scale = math.sqrt(in_features / 2.0)
            else:
                # Edward2 default: 1.0 when inputs are normalized
                kernel_scale = 1.0

        self.normalize_input = normalize_input
        self.return_covariance = return_covariance or not self.training # always return cov if in eval mode
        self.return_features = return_features
        self.scale_random_features = scale_random_features

        logger.debug("Initializing RandomFeatureGaussianProcess layer") if self.verbose else None
        logger.debug(f"Input features: {in_features}") if self.verbose else None
        logger.debug(f"Output features: {self.out_features}") if self.verbose else None

        if normalize_input:
            logger.debug("Using LayerNorm to normalize input") if self.verbose else None
            self.input_norm = nn.LayerNorm(in_features, elementwise_affine=True)  # TF parity

        # Random feature mapping
        if use_custom_features:
            logger.debug("Using custom random feature layer") if self.verbose else None
            self.feature_layer = nn.Linear(in_features, random_features, bias=False)
            output_in = random_features
        elif kernel_type.lower() == "linear":
            logger.debug("Using linear kernel (no random features)") if self.verbose else None
            self.feature_layer = nn.Identity()
            # ensure output layer matches input features
            output_in = in_features
        else:
            logger.debug("Using RBF kernel with random features") if self.verbose else None
            self.feature_layer = RandomFourierFeatures(
                in_features,
                random_features,
                kernel_type,
                kernel_scale,
                trainable_kernel_scale,
            )
            output_in = random_features

        # Output linear projection
        self.output_layer = nn.Linear(output_in, self.out_features, bias=False)
        # Initialize output layer weights with normal distribution with small stddev
        nn.init.normal_(self.output_layer.weight, std=init_stdev)

        # Output bias
        if output_bias_trainable:
            logger.debug(f"Output bias is trainable: {output_bias_trainable}") if self.verbose else None
            self.bias = nn.Parameter(torch.zeros(self.out_features))
        else:
            self.register_buffer("bias", torch.zeros(self.out_features))

        nn.init.constant_(self.bias, 0.0)
        # Covariance estimator
        self.covariance_layer = LaplaceRandomFeatureCovariance(
            in_features=output_in,
            momentum=covariance_momentum,
            ridge_penalty=covariance_ridge_penalty,
            likelihood=covariance_likelihood,
        )

    def reset_precision(self):
        """Resets the precision matrix of the covariance estimator to its initial state.

        This method is used to clear the accumulated precision statistics in the covariance layer,
        typically before starting a new training epoch or evaluation.

        Returns:
            None.
        """
        self.covariance_layer.reset_precision()

    def forward(self, x: torch.Tensor):
        """Performs a forward pass through the Gaussian Process layer.

        This method transforms the input using random features, computes logits and covariance,
        and applies mean-field adjustment to the logits if covariance is available.

        Args:
            x: Input tensor of shape (batch_size, in_features).

        Returns:
            A dictionary containing the adjusted logits, raw logits, covariance matrix, and random features.
        """
        if self.normalize_input:
            x = self.input_norm(x)
        else:
            # Edward2 parity: scale inputs only for simple linear/custom features, not RFF
            if isinstance(self.feature_layer, nn.Linear) and not isinstance(self.feature_layer, RandomFourierFeatures):
                if hasattr(self.feature_layer, "kernel_scale") and self.feature_layer.kernel_scale is not None:
                    ell = float(self.feature_layer.kernel_scale)
                    if ell > 0:
                        x = x / math.sqrt(ell)

        Phi = self.feature_layer(x)
        if hasattr(self.feature_layer, "out_features") and self.scale_random_features:
            # Apply Rahimi–Recht scaling only once (Edward2 parity)
            num_rand_features = float(self.feature_layer.out_features)
            Phi = Phi * math.sqrt(2.0 / num_rand_features)

        logits = self.output_layer(Phi) + self.bias

        return_covariance = self.return_covariance if self.training else True

        cov_out = self.covariance_layer(Phi, logits) if return_covariance else None
        Phi_out = Phi if self.return_features else None
        # default to unadjusted return logits
        output_dict = {
            "logits": logits.clone(),
            "logits_raw": logits.clone(),
            "cov": cov_out,
            "features": Phi_out,
            "mean_field_applied": None,
            }

        if (cov_out is not None) and (not self.training):
            # Adjust logits using mean-field approximation
            output_dict["logits"] = mean_field_logits(logits=logits, covariance=cov_out)
            output_dict["mean_field_applied"] = True

        return output_dict


class LaplaceRandomFeatureCovariance(nn.Module):
    """Computes the Gaussian Process covariance using Laplace method.

    At training time, this layer updates the Gaussian process posterior using
    model features in minibatches.

    Attributes:
        momentum: (float) A discount factor used to compute the moving average for
            posterior precision matrix. Analogous to the momentum factor in batch
            normalization. If -1 then update covariance matrix using a naive sum
            without momentum, which is desirable if the goal is to compute the exact
            covariance matrix by passing through data once (say in the final epoch).

        ridge_penalty: (float) Initial Ridge penalty to weight covariance matrix.
            This value is used to stablize the eigenvalues of weight covariance
            estimate so that the matrix inverse can be computed for Cov = inv(t(X) * X
            + s * I). The ridge factor s cannot be too large since otherwise it will
            dominate the t(X) * X term and make covariance estimate not meaningful.

        likelihood: (str) The likelihood to use for computing Laplace approximation
            for the covariance matrix. Can be one of ('binary_logistic', 'poisson',
            'gaussian').
    """

    def __init__(
        self,
        in_features,
        momentum: float =0.999,
        ridge_penalty: float =1e-6,
        likelihood: str ="gaussian",
        device=None,
        dtype=None,
    ):
        if likelihood not in _SUPPORTED_LIKELIHOOD:
            raise ValueError(
                f'"likelihood" must be one of {_SUPPORTED_LIKELIHOOD}, got {likelihood}.'
            )

        super(LaplaceRandomFeatureCovariance, self).__init__()
        self.in_features = in_features
        self.ridge_penalty = ridge_penalty
        self.momentum = momentum
        self.likelihood = likelihood

        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.device = device
        self.dtype = dtype

        # TF: precision starts at zeros; ridge is injected at inversion.
        self.init_precision = torch.zeros(in_features, in_features, **self.factory_kwargs)
        self.register_buffer("_precision", torch.zeros(in_features, in_features, **self.factory_kwargs))
        self.register_buffer("_covariance", torch.zeros(in_features, in_features, **self.factory_kwargs))

        self.covariance_is_cached = False

    @property
    def precision(self):
        return self._precision

    @precision.setter
    def precision(self, val):
        self.covariance_is_cached = False
        # self._precision = self.register_buffer('_precision', val)
        self._precision = val

    @property
    def covariance(self):
        # If precision is all zeros or NaN: (s I + 0)^(-1) = (1/s) I
        if not self._precision.any() or torch.isnan(self._precision).all():
            eye = torch.eye(self.in_features, **self.factory_kwargs)
            return eye / self.ridge_penalty

        if not self.covariance_is_cached:
            # TF parity: invert (ridge * I + precision)
            eye = torch.eye(self.in_features, device=self._precision.device, dtype=self._precision.dtype)
            self._covariance = torch.linalg.inv(self.ridge_penalty * eye + self._precision)

        self.covariance_is_cached = True
        return self._covariance

    @covariance.setter
    def covariance(self, val):
        # self._covariance = self.register_buffer('_covariance', val)
        self._covariance = val

    def update_precision(self, Phi, logits):
        """
        Given the current forward pass yielding random features Phi, update the covariance matrix
        for the entire set of input data.
        """
        if self.likelihood != "gaussian":
            if logits is None:
                raise ValueError(
                    f'"logits" cannot be None when likelihood={self.likelihood}'
                )
            # if logits.shape[-1] != 1:
            #     raise ValueError(
            #         f"likelihood={self.likelihood} only supports univariate logits."
            #         f"Got logits dimension: {logits.shape[-1]}"
            #     )

        batch_size = Phi.shape[0]

        # --- Curvature multiplier p(1-p) ---
        if self.likelihood == "binary_logistic":
            prob = torch.sigmoid(logits)
            prob_multiplier = prob * (1.0 - prob)
            # get [B, 1] shape
            if prob_multiplier.ndim > 1:
                prob_multiplier = prob_multiplier.mean(dim=-1, keepdim=True)

        elif self.likelihood == "multiclass_logistic":
            # Note: this is experimental and may not work well
            logger.warning("Multiclass likelihood is experimental; use with caution.")

            # Approximate curvature upper bound: max_k p_k(1-p_k)
            probs = torch.softmax(logits, dim=-1)
            # prob_multiplier, _ = torch.max(probs * (1.0 - probs), dim=-1, keepdim=True)
            # for even tighter TF parity, consider computing the mean curvature instead of the max, which approximates a Laplace average
            prob_multiplier = torch.mean(probs * (1 - probs), dim=-1, keepdim=True)
        elif self.likelihood == "poisson":
            prob_multiplier = torch.exp(logits)
        elif self.likelihood == "gaussian":
            prob_multiplier = torch.ones(1, device=Phi.device) * 1.0
        else:
            raise ValueError(f"Invalid likelihood entered: {self.likelihood}")

        # Shape-robust multiplier: (B, 1) broadcast over features
        Phi = torch.sqrt(prob_multiplier).reshape(-1, 1) * Phi
        batch_precision = Phi.transpose(-2, -1) @ Phi

        # Update the non-batch (i.e. all data) precision matrix.
        dev = self._precision.device
        if self.momentum > 0:
            batch_precision = batch_precision / batch_size
            # IMPORTANT: assign to trigger setter (cache invalidation)
            new_precision = self.momentum * self.precision + (1 - self.momentum) * batch_precision.to(dev)
            self.precision = new_precision
        else:
            # Compute exact population-wise covariance without momentum.
            # If use this option, make sure to pass through data only once.
            self.precision = self.precision + batch_precision.to(dev)
        return self

    def reset_precision(self):
        self.precision = self.init_precision.clone()

    def compute_predictive_covariance(self, Phi):
        """Computes posterior predictive variance.

        The given testing random features Phi (B, H), we pull out the covariance matrix
        Cov (H, H) in random feature space and compute var_k = Phi @ Cov @ Phi.T to
        get an (B, B) covariance matrix of the batch with size B.

        Approximates the Gaussian process posterior using random features. Suppose
        the dataset size is N, i.e. there are N datapoints in total. Then the covariance
        would be Cov_train = (Phi_train^T @ Phi_train + lambda * 1)^{-1} = size(H, H), where lambda is
        the ridge regression penalty, and 1 is an (H, H) identity (this is self.init_cov).

        Given a testing batch of size (B', H) of random features, we wish to compute the
        covariance of this test batch given the covariance of the training data. This is done
        via:

        Cov_test = lambda * Phi_test @ Cov_train @ Phi_test^T = size(B', B')

        matrix inversion is expensive, so we cache the precision inverse result. After
        computing any forward pass with training enabled, we reset the covariance matrix
        assuming that the forward pass has made the covariance cache stale (new data came in).

        Args:
            Phi: (torch.tensor) The random feature of testing data to be used for
                computing the covariance matrix. Shape (batch_size, gp_hidden_size).

        Returns:
            (torch.tensor) Predictive covariance matrix, shape (batch_size, batch_size).
        """
        # Cov_test = s * Φ_test @ (s I + ΦᵀΦ)^(-1) @ Φ_testᵀ
        # TODO: check when to use self.ridge_penalty - Should be okay
        # DO NOT multiply by ridge_penalty here
        return Phi @ self.covariance.to(Phi.device) @ Phi.transpose(-2, -1)

    def forward(self, Phi, logits=None):
        """Minibatch updates the GP's posterior precision matrix estimate.

        Args:
            inputs: (tf.Tensor) GP random features, shape (batch_size,
                gp_hidden_size).
            logits: (tf.Tensor) Pre-activation output from the model. Needed
                for Laplace approximation under a non-Gaussian likelihood.
            training: (tf.bool) whether or not the layer is in training mode. If in
                training mode, the gp_weight covariance is updated using gp_feature.

        Returns:
            gp_stddev (tf.Tensor): GP posterior predictive variance,
                shape (batch_size, batch_size).
        """
        batch_size = Phi.shape[0]
        if self.training:
            self.update_precision(Phi=Phi, logits=logits)
            return torch.eye(batch_size, device=Phi.device)

        return self.compute_predictive_covariance(Phi=Phi)


# Note: I believe the orig impl used mean_field_factor math.pi/8
@torch.no_grad()
def mean_field_logits(
    logits: torch.Tensor,
    covariance: torch.Tensor | None = None,
    mean_field_factor: float = math.pi / 8.0,
) -> torch.Tensor:
    """
    Mean-field correction for SNGP logits:
      logits_mf = logits / sqrt(1 + mean_field_factor * v)
    where v is per-example predictive variance.

    Accepts covariance as:
      - None          -> v = 1.0 (acts like temperature scaling with sqrt(1+π/8))
      - [B]           -> v = covariance
      - [B, B]        -> v = diag(covariance)
      - [B, C]        -> (discouraged) reduce to scalar per example via mean across classes
    """
    if mean_field_factor is None or mean_field_factor < 0:
        return logits

    if not isinstance(logits, torch.Tensor):
        logger.warning(f"Expected logits to be a torch.Tensor but got {type(logits)}. Returning unmodified logits.")
        raise ValueError(f"Expected logits to be a torch.Tensor but got {type(logits)}")

    B = logits.shape[0]

    if covariance is None:
        # when covariance is None, set v=1.0 (acts like temperature scaling with sqrt(1+π/8))
        v = torch.ones(B, device=logits.device, dtype=logits.dtype)
    else:
        if covariance.dim() == 1 and covariance.shape[0] == B:
            # predictive variance per example
            v = covariance
        elif covariance.dim() == 2 and covariance.shape[0] == covariance.shape[1] == B:
            # batch predictive covariance -> take diagonal
            v = torch.diagonal(covariance, dim1=-2, dim2=-1)
        elif covariance.dim() == 2 and covariance.shape[0] == B and covariance.shape[1] == logits.shape[1]:
            # per-class variances returned (not standard for SNGP); reduce to scalar
            v = covariance.mean(dim=1)
        else:
            raise ValueError(
                f"Unsupported covariance shape {tuple(covariance.shape)} for mean-field logits."
            )

    # ensure non-negative variance
    v = v.clamp_min(1e-12)
    scale = torch.sqrt(1.0 + mean_field_factor * v).unsqueeze(-1)  # [B, 1]
    return logits / scale
