# Quantum GAN (QGAN) for Simple Probability Distributions

## What this project does
- A **quantum circuit generator** (built with PennyLane) creates fake samples.
- A **neural network discriminator** (built with PyTorch) guesses whether a sample is real or fake.
- Both models compete until the generated samples look like the real Gaussian distribution.

## Why Quantum Helps
- Quantum states naturally store probabilities in amplitudes.
- Superposition lets a small circuit shape distributions efficiently.
- With only a few parameters, the QGAN can match a simple classical baseline.

## How the Gaussian example works
- Real samples come from a small Gaussian distribution.
- The quantum generator tries to mimic this bell-shaped curve.
- After training, the two curves should overlap when plotted.

## Why This Repo Uses SVG Instead of PNG
> GitHub’s CODEX interface cannot preview binary files like PNG/JPG and shows
> “Binary files are not supported” warnings.  
> To avoid this, all visualizations in this project are lightweight SVG files.
> SVGs are text-based, small, and render cleanly inside GitHub and CODEX.

## Step-by-step instructions
```bash
pip install -r requirements.txt
python app/main.py
```

## What you should see
- Loss values that decrease during training.
- Real vs fake distributions becoming similar.
- SVG files created in `/examples/`.

## Future extensions
- Multi-qubit generator for richer shapes.
- Wasserstein QGAN for stabler training.
- Additional target distributions beyond a single Gaussian.

## Repository layout
- `app/` – dataset, quantum generator, discriminator, training loop, and SVG plotting utilities.
- `examples/` – sample SVG figures for quick preview.
- `notebooks/` – tutorial notebook configured for inline SVG rendering.
- `tests/` – lightweight tests to keep the demo healthy.

Enjoy exploring QGANs in a gentle, beginner-friendly way!
