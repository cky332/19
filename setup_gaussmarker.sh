#!/bin/bash
# ============================================================================
# GaussMarker (MarkDiffusion) - Anaconda Environment Setup Script
# ============================================================================
# This script sets up a complete Anaconda environment for running the
# GaussMarker watermarking algorithm from the MarkDiffusion toolkit.
#
# Prerequisites:
#   - Anaconda or Miniconda installed
#   - NVIDIA GPU with CUDA support (recommended)
#   - At least 16GB RAM, 20GB+ disk space
#
# Usage:
#   chmod +x setup_gaussmarker.sh
#   ./setup_gaussmarker.sh
# ============================================================================

set -e

ENV_NAME="gaussmarker"
PYTHON_VERSION="3.11"

echo "============================================"
echo " GaussMarker Environment Setup"
echo "============================================"

# Step 1: Create conda environment
echo "[1/5] Creating conda environment: ${ENV_NAME} (Python ${PYTHON_VERSION})..."
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

# Step 2: Activate environment
echo "[2/5] Activating environment..."
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# Step 3: Install PyTorch with CUDA support
echo "[3/5] Installing PyTorch with CUDA 12.6 support..."
pip install torch==2.6.0+cu126 torchvision==0.21.0+cu126 \
    --extra-index-url https://download.pytorch.org/whl/cu126

# Step 4: Install core dependencies
echo "[4/5] Installing MarkDiffusion core dependencies..."
pip install \
    numpy==2.2.5 \
    scipy==1.15.2 \
    scikit-learn==1.6.1 \
    Pillow==11.2.1 \
    opencv-python==4.11.0.86 \
    matplotlib==3.10.1 \
    tqdm==4.67.1 \
    ujson==5.10.0 \
    pandas \
    datasets==3.5.0 \
    diffusers==0.33.1 \
    transformers==4.47.1 \
    accelerate \
    huggingface_hub \
    sentence-transformers==4.1.0 \
    pycryptodome==3.22.0 \
    galois==0.4.5 \
    ldpc==2.3.4

# Step 5: Install the project in editable mode (from source)
echo "[5/5] Installing MarkDiffusion in editable mode..."
pip install -e ".[optional]"

echo ""
echo "============================================"
echo " Setup Complete!"
echo "============================================"
echo ""
echo "To activate the environment:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To run GaussMarker, see the demo notebook:"
echo "  jupyter notebook MarkDiffusion_demo.ipynb"
echo ""
