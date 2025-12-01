#!/bin/bash
# Post-create script for GitHub Codespaces / Dev Container
# This script runs after the container is created

set -e

echo "🚀 Setting up Physics-Informed Neural Networks for Printed Memristors environment..."

# Upgrade pip to latest version
echo "📦 Upgrading pip..."
python -m pip install --upgrade pip

# Install project dependencies
echo "📦 Installing project dependencies..."
pip install -r requirements.txt

# Create results directories if they don't exist
echo "📁 Ensuring results directories exist..."
mkdir -p results/results_best
mkdir -p results/results_cv
mkdir -p results/results_ablation
mkdir -p results/extended_validation
mkdir -p results/supplementary_experiments
mkdir -p results/spice_integration

# Set up matplotlib backend for headless environments
echo "⚙️ Configuring matplotlib for headless environment..."
mkdir -p ~/.config/matplotlib
echo "backend: Agg" > ~/.config/matplotlib/matplotlibrc

# Verify installation
echo "✅ Verifying installation..."
python -c "import numpy; print(f'  NumPy: {numpy.__version__}')"
python -c "import pandas; print(f'  Pandas: {pandas.__version__}')"
python -c "import matplotlib; print(f'  Matplotlib: {matplotlib.__version__}')"
python -c "import scipy; print(f'  SciPy: {scipy.__version__}')"
python -c "import tensorflow; print(f'  TensorFlow: {tensorflow.__version__}')"

echo ""
echo "✨ Environment setup complete!"
echo ""
echo "Quick start commands:"
echo "  # Train PINN model (best configuration)"
echo "  python src/run_pinn.py --mode full --full-epochs 800 --full-hidden-layers 4 --full-neurons 128 --full-learning-rate 2e-4 --full-noise-std 0.002 --full-variability 0.05 --full-max-physics-weight 0.1 --full-trainable-params ohmic_conductance --full-disable-concentration --full-seed 42 --results-dir results_best"
echo ""
echo "  # Run supplementary experiment 1"
echo "  python src/exp1_dynamic_pulse_response.py"
echo ""