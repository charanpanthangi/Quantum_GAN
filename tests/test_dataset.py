import numpy as np

from app.dataset import normalize, sample_real_distribution


def test_sample_shape_and_mean():
    samples = sample_real_distribution(1000, mean=0.0, std=0.2)
    assert samples.shape == (1000,)
    assert abs(np.mean(samples)) < 0.1


def test_normalize_range():
    values = np.array([2.0, 4.0, 6.0])
    normalized = normalize(values)
    assert normalized.min() == 0.0
    assert normalized.max() == 1.0
