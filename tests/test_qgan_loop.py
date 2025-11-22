import numpy as np

from app.dataset import sample_real_distribution, normalize
from app.qgan import train_qgan


def test_training_reduces_loss():
    real = normalize(sample_real_distribution(64))
    result = train_qgan(real_samples=real, n_epochs=5, batch_size=8, lr=0.05)
    # Loss should trend downward; compare first and last values.
    assert result.generator_losses[0] >= result.generator_losses[-1] or result.discriminator_losses[0] >= result.discriminator_losses[-1]
    assert result.generated_samples.shape[0] == real.shape[0]
