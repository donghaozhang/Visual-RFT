#!/bin/bash
# Script to create and configure RFTV4 conda environment

set -e  # Exit on error

echo "Creating RFTV4 conda environment from specification..."
conda env create -f conda_environment.yml

echo "Activating RFTV4 conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate RFTV4

# Verify the environment
echo "Verifying conda environment..."
python -c "import sys; print(f'Python: {sys.version}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU Count: {torch.cuda.device_count()}')"
python -c "if torch.cuda.is_available(): print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

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

echo "RFTV4 environment setup completed!" 