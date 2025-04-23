#!/bin/bash
# Script to install required packages for Visual-RFT

set -e  # Exit on error

# Ensure conda is available and activate RFT environment
echo "Activating RFT conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate RFT

# Check if the environment was activated properly
if [[ $CONDA_DEFAULT_ENV != "RFT" ]]; then
  echo "Error: Failed to activate RFT conda environment"
  exit 1
fi

echo "Using conda environment: RFT"
python -c "import sys; print(f'Python interpreter: {sys.executable}')"

echo "Installing PyTorch 2.0.1 with CUDA support..."
# Install PyTorch 2.0.1 (known to work with flash-attn 2.3.0)
pip uninstall -y torch torchvision torchaudio flash-attn

# Install PyTorch with CUDA
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

echo "Installing Flash Attention..."
# Install compatible version of Flash Attention
pip install flash-attn==2.3.0 --no-build-isolation

echo "Installing required packages..."

# Install Hugging Face packages with compatible versions
pip install datasets
pip install transformers==4.34.0 
pip install accelerate peft 

# Install other dependencies
pip install einops timm pillow
pip install wandb tensorboardx
pip install qwen_vl_utils

# Check installations 
echo "Verifying installations..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import datasets; print(f'Datasets version: {datasets.__version__}')"

# Verify flash attention
echo "Verifying Flash Attention..."
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

echo "Installation completed successfully!" 