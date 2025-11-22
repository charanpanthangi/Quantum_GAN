"""Utility functions for creating simple one-dimensional datasets.

The goal of this project is to teach a Quantum Generative Adversarial
Network (QGAN) how to mimic a basic probability distribution. We use a
Gaussian distribution because it is easy to understand: most values
cluster around the mean and fewer appear farther away.
"""

from __future__ import annotations

import numpy as np


def sample_real_distribution(n_samples: int, mean: float = 0.0, std: float = 0.2) -> np.ndarray:
    """Generate samples from a simple 1D Gaussian distribution.

    Args:
        n_samples: Number of samples to draw.
        mean: Center of the Gaussian (default 0.0).
        std: Standard deviation controlling spread (default 0.2).

    Returns:
        NumPy array of shape ``(n_samples,)`` containing floating-point values.

    Notes:
        The samples are not normalized by default. Normalization to the
        ``[0, 1]`` interval is optional and can be performed by callers if
        needed. GANs learn by comparing the shape of the generated data to the
        shape of these real samples.
    """

    # Draw numbers around the mean; most will be close, few will be far away.
    samples = np.random.normal(loc=mean, scale=std, size=n_samples)
    return samples


def normalize(samples: np.ndarray) -> np.ndarray:
    """Normalize an array to the ``[0, 1]`` interval.

    Args:
        samples: Array of values to normalize.

    Returns:
        Array scaled to the ``[0, 1]`` range. If all values are equal, zeros are
        returned to avoid division-by-zero.
    """

    min_val = float(np.min(samples))
    max_val = float(np.max(samples))
    if max_val == min_val:
        # All values are identical; normalization would divide by zero.
        return np.zeros_like(samples)
    return (samples - min_val) / (max_val - min_val)
