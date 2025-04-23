#!/bin/bash
# Script to install compatible Flash Attention for PyTorch 2.5.1

set -e  # Exit on error

# Ensure conda is available and activate RFTV5 environment
echo "Activating RFTV5 conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate RFTV5

# Check if the environment was activated properly
if [[ $CONDA_DEFAULT_ENV != "RFTV5" ]]; then
  echo "Error: Failed to activate RFTV5 conda environment"
  exit 1
fi

echo "Using conda environment: RFTV5"
python -c "import sys; print(f'Python interpreter: {sys.executable}')"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Remove any existing flash-attn
echo "Removing existing Flash Attention..."
pip uninstall -y flash-attn

# Install build tools if needed
pip install ninja wheel

# Use a compatible version of Flash Attention for PyTorch 2.5.1
echo "Installing compatible Flash Attention..."
FLASH_ATTN_FORCE_BUILD=1 pip install flash-attn==2.5.5 --no-build-isolation

# Test Flash Attention
echo "Testing Flash Attention..."
python -c "
try:
    from flash_attn import flash_attn_func
    print('Flash Attention is installed correctly')
    import torch
    # Create small sample tensors
    bs, sl, h, d = 2, 16, 8, 64  # Small test case
    q = torch.randn(bs, sl, h, d, device='cuda', dtype=torch.float16)
    k = torch.randn(bs, sl, h, d, device='cuda', dtype=torch.float16)
    v = torch.randn(bs, sl, h, d, device='cuda', dtype=torch.float16)
    # Test flash attention
    out = flash_attn_func(q, k, v)
    print(f'Flash Attention test successful! Output shape: {out.shape}')
except Exception as e:
    print(f'Error with Flash Attention: {e}')
"

echo "Flash Attention installation completed!" 