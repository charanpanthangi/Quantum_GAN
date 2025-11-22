# Simple Docker image for running the QGAN demo
# We use python:3.11-slim to keep the image lightweight
FROM python:3.11-slim

# Set a working directory inside the container
WORKDIR /workspace/Quantum_GAN

# Copy dependency list first to leverage Docker layer caching
COPY requirements.txt requirements.txt

# Install required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the source code
COPY . .

# Provide a default command that runs the CLI with modest settings
CMD ["python", "app/main.py", "--n_samples", "128", "--n_epochs", "50", "--output_dir", "examples"]
