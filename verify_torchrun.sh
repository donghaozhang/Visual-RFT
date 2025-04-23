#!/bin/bash

# Verify torchrun with 2 GPUs for Visual-RFT
# This script activates the RFT conda environment and runs a test with torchrun

# Exit on error
set -e

echo "Activating RFT conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate RFT

# Check if the environment was activated properly
if [[ $CONDA_DEFAULT_ENV != "RFT" ]]; then
  echo "Error: Failed to activate RFT conda environment"
  exit 1
fi

# Fix Flash Attention compatibility issue
echo "Checking PyTorch version and fixing Flash Attention compatibility..."
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
echo "Current PyTorch version: $TORCH_VERSION"

echo "Reinstalling compatible Flash Attention version..."
# pip uninstall -y flash-attn
# # Fix NumPy version issue
# pip install numpy==1.24.3
# pip install torch==2.0.1 # Install a compatible PyTorch version
# pip install flash-attn==2.3.0 --no-build-isolation # Install a compatible Flash Attention version
# pip install datasets
# pip install transformers

echo "Checking CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}')"

# Check if we have at least 2 GPUs
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
if [[ $GPU_COUNT -lt 2 ]]; then
  echo "Warning: Less than 2 GPUs detected ($GPU_COUNT). This test requires 2 GPUs."
  exit 1
fi

# Verify flash-attn is working
echo "Verifying flash-attn installation..."
python flash_attn_test.py

# Run the torchrun test with 2 GPUs
echo "Running torchrun test with 2 GPUs..."
torchrun --nproc_per_node=2 torch_test.py

echo "Test completed successfully!" 
