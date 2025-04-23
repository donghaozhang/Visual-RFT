#!/bin/bash
# Script to download the Qwen2-VL model for Visual-RFT training

set -e  # Exit on error

# Color formatting for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}============================================${NC}"
echo -e "${YELLOW}     Visual-RFT Model Downloader           ${NC}"
echo -e "${YELLOW}============================================${NC}"

# Path configurations - adjust as needed
MODEL_REPO="Qwen/Qwen2-VL-2B-Instruct"  # Use actual Hugging Face repo ID
ROOT_DIR="/home/ubuntu/petertest/new_start/Visual-RFT"
CHECKPOINT_PATH="${ROOT_DIR}/share_models/Qwen2-VL-2B-Instruct"

# Determine the Python path from conda
if [ -n "$CONDA_PREFIX" ]; then
    PYTHON_PATH="$CONDA_PREFIX/bin/python"
else
    PYTHON_PATH=$(which python)
fi

echo -e "${YELLOW}Using Python: ${GREEN}$PYTHON_PATH${NC}"

# Check if conda environment is active
if [[ "$CONDA_DEFAULT_ENV" != "rftv3" ]]; then
    echo -e "${YELLOW}Activating rftv3 conda environment...${NC}"
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate rftv3
    
    # Check if activation was successful
    if [[ "$CONDA_DEFAULT_ENV" != "rftv3" ]]; then
        echo -e "${RED}Error: Failed to activate rftv3 conda environment${NC}"
        echo -e "${YELLOW}Make sure the environment exists using: conda env list${NC}"
        exit 1
    fi
fi

# Step 1: Create model directory if it doesn't exist
echo -e "${YELLOW}Creating model directory...${NC}"
mkdir -p "$CHECKPOINT_PATH"

# Step 2: Check for required packages
echo -e "${YELLOW}Checking required packages...${NC}"
$PYTHON_PATH -c "
import sys
try:
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoProcessor, AutoModel
    print('All required packages are installed.')
except ImportError as e:
    print(f'Error: {e}')
    print('Please install the required packages with: pip install transformers torch')
    sys.exit(1)
"

# Step 3: Download the model using the Hugging Face transformers library
echo -e "${YELLOW}Downloading model from ${GREEN}$MODEL_REPO${YELLOW} to ${GREEN}$CHECKPOINT_PATH${YELLOW}...${NC}"
$PYTHON_PATH -c "
import os
import sys
from transformers import AutoTokenizer, AutoProcessor, AutoModel

# Path settings
model_repo = '$MODEL_REPO'
checkpoint_path = '$CHECKPOINT_PATH'

print(f'Downloading model from {model_repo} to {checkpoint_path}...')

try:
    # Download the full model with weights
    print('Downloading model weights (this may take a while)...')
    model = AutoModel.from_pretrained(
        model_repo, 
        trust_remote_code=True,
        device_map='auto',  # Use all available GPUs
        torch_dtype='auto'  # Use the appropriate dtype
    )

    # Save the model to disk
    print('Saving model weights to disk...')
    model.save_pretrained(checkpoint_path)
    print(f'Model weights saved to {checkpoint_path}')

    # Download tokenizer
    print('Downloading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(model_repo, trust_remote_code=True)
    tokenizer.save_pretrained(checkpoint_path)
    print('Tokenizer saved')

    # Download processor config (important for vision models)
    print('Downloading processor configuration...')
    processor = AutoProcessor.from_pretrained(model_repo, trust_remote_code=True)
    processor.save_pretrained(checkpoint_path)
    print('Processor configuration saved')

    print('Model files downloaded successfully!')
    print(f'Check the directory: {checkpoint_path}')
    
except Exception as e:
    print(f'Error: {e}')
    sys.exit(1)
"

echo -e "${GREEN}=========================================================${NC}"
echo -e "${GREEN}Model download completed!${NC}"
echo -e "${GREEN}The model is now available at: ${YELLOW}$CHECKPOINT_PATH${NC}"
echo -e "${GREEN}You can now run the training script with:${NC}"
echo -e "${YELLOW}bash src/scripts/train_2gpu_fixed.sh${NC}"
echo -e "${GREEN}=========================================================${NC}" 