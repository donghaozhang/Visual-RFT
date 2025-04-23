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

echo "Checking CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}')"

# Check if we have at least 2 GPUs
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
if [[ $GPU_COUNT -lt 2 ]]; then
  echo "Warning: Less than 2 GPUs detected ($GPU_COUNT). This test requires 2 GPUs."
  exit 1
fi

echo "Running torchrun test with 2 GPUs..."
# Create a simple test script
cat > /tmp/torch_test.py << 'EOL'
import os
import torch
import torch.distributed as dist

def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Initialize the process group
    dist.init_process_group("nccl")
    
    # Get device info
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    # Get some basic info
    print(f"Running on node: {os.uname().nodename}")
    print(f"Rank: {rank}, Local Rank: {local_rank}, World Size: {world_size}")
    print(f"Device: {device}, Device name: {torch.cuda.get_device_name(local_rank)}")
    
    # Create a tensor on the device
    x = torch.randn(10, 10).to(device)
    print(f"Tensor created on {device}: shape={x.shape}")
    
    # Synchronize all processes
    dist.barrier()
    
    if rank == 0:
        print("\nTorchrun with 2 GPUs is working correctly!")

if __name__ == "__main__":
    main()
EOL

# Run the test with torchrun
torchrun --nproc_per_node=2 /tmp/torch_test.py

echo "Test completed successfully!" 

# pip install flash-attn==2.5.3 --no-build-isolation
