#!/bin/bash
# Script to download and prepare the ViRFT_COCO_base65 dataset for training

set -e  # Exit on error

# Color formatting for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}============================================${NC}"
echo -e "${YELLOW}     Visual-RFT Dataset Downloader         ${NC}"
echo -e "${YELLOW}============================================${NC}"

# Path configurations - adjust as needed
DATASET_REPO="laolao77/ViRFT_COCO_base65"
ROOT_DIR="/home/ubuntu/petertest/new_start/Visual-RFT"
DOWNLOAD_DIR="$ROOT_DIR/shared_data/ViRFT_COCO_base65_hf"
CONVERT_DIR="$ROOT_DIR/share_data/base65cate_6k_think"  # This matches the path in the training script

# Determine the Python path
if [ -n "$CONDA_PREFIX" ]; then
    PYTHON_PATH="$CONDA_PREFIX/bin/python"
    HF_CLI="$CONDA_PREFIX/bin/huggingface-cli"
else
    PYTHON_PATH=$(which python)
    HF_CLI=$(which huggingface-cli)
fi

echo -e "${YELLOW}Using Python: ${GREEN}$PYTHON_PATH${NC}"
echo -e "${YELLOW}Using Hugging Face CLI: ${GREEN}$HF_CLI${NC}"

# Step 1: Create directories if they don't exist
echo -e "${YELLOW}Creating directories...${NC}"
mkdir -p "$DOWNLOAD_DIR"
mkdir -p "$CONVERT_DIR"

# Step 2: Check for the huggingface-cli tool
if ! command -v $HF_CLI &> /dev/null; then
    echo -e "${RED}huggingface-cli not found. Installing...${NC}"
    pip install huggingface_hub[cli]
    HF_CLI=$(which huggingface-cli)
fi

# Step 3: Download the dataset using huggingface-cli
echo -e "${YELLOW}Downloading dataset from $DATASET_REPO...${NC}"
if [ -d "$DOWNLOAD_DIR/data" ] && [ "$(ls -A $DOWNLOAD_DIR/data/*.parquet 2>/dev/null | wc -l)" -ge 7 ]; then
    echo -e "${GREEN}Dataset already exists at $DOWNLOAD_DIR, skipping download.${NC}"
else
    echo -e "${YELLOW}Downloading dataset to $DOWNLOAD_DIR...${NC}"
    $HF_CLI download $DATASET_REPO --repo-type dataset --local-dir $DOWNLOAD_DIR
fi

# Step 4: Convert the dataset to Arrow format
echo -e "${YELLOW}Converting dataset to Arrow format...${NC}"
$PYTHON_PATH -c "
import os
from datasets import load_dataset, DatasetDict

# Path settings
source_path = '$DOWNLOAD_DIR/data'
output_path = '$CONVERT_DIR'

print(f'Converting dataset from {source_path} to {output_path}...')

# Create output directory
os.makedirs(output_path, exist_ok=True)

# Load the Parquet files
parquet_files = [os.path.join(source_path, f) for f in os.listdir(source_path) if f.endswith('.parquet')]
print(f'Found {len(parquet_files)} Parquet files')

# Load dataset
dataset = load_dataset('parquet', data_files=parquet_files)
print(f'Successfully loaded dataset: {dataset}')

# Create DatasetDict
dataset_dict = DatasetDict({'train': dataset['train']})
print(f'Created DatasetDict: {dataset_dict}')

# Save in Arrow format
dataset_dict.save_to_disk(output_path)
print(f'Dataset saved to {output_path} in Arrow format')

# Verify
loaded_dataset = DatasetDict.load_from_disk(output_path)
print(f'Successfully verified dataset with {len(loaded_dataset[\"train\"])} examples')
"

echo -e "${GREEN}=========================================================${NC}"
echo -e "${GREEN}Dataset download and conversion completed successfully!${NC}"
echo -e "${GREEN}The dataset is now available at: ${NC}"
echo -e "${YELLOW}$CONVERT_DIR${NC}"
echo -e "${GREEN}This path matches the one used in the training scripts.${NC}"
echo -e "${GREEN}You can now run the training script with:${NC}"
echo -e "${YELLOW}bash src/scripts/train_2gpu.sh${NC}"
echo -e "${GREEN}=========================================================${NC}" 