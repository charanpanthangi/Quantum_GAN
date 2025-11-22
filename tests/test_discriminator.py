import torch

from app.discriminator import Discriminator


def test_discriminator_forward():
    model = Discriminator()
    x = torch.rand(4, 1)
    output = model(x)
    assert output.shape == (4, 1)
    assert torch.all((output >= 0.0) & (output <= 1.0))
