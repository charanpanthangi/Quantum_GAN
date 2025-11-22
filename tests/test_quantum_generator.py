import torch

from app.quantum_generator import build_qnode, generate_samples, initialize_weights, quantum_generator


def test_generator_output_range():
    qnode = build_qnode(n_layers=1, n_qubits=1)
    weights = initialize_weights(n_layers=1, n_qubits=1)
    value = quantum_generator(weights, qnode)
    assert 0.0 <= float(value) <= 1.0


def test_generate_samples_shape():
    qnode = build_qnode(n_layers=1, n_qubits=1)
    weights = initialize_weights(n_layers=1, n_qubits=1)
    batch = generate_samples(weights, qnode, n_samples=5)
    assert batch.shape == (5, 1)
    assert torch.all((batch >= 0.0) & (batch <= 1.0))
