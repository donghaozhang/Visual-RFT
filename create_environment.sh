#!/bin/bash

# Set color codes for output formatting
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Define environment name
ENV_NAME="rftv3"

echo -e "${YELLOW}Creating conda environment '${ENV_NAME}' from environment.yml...${NC}"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Conda is not installed or not in PATH. Please install conda first.${NC}"
    exit 1
fi

# Check if environment already exists
if conda info --envs | grep -q "${ENV_NAME}"; then
    echo -e "${YELLOW}Environment '${ENV_NAME}' already exists. Do you want to replace it? (y/n)${NC}"
    read -r answer
    if [[ "$answer" != "y" ]]; then
        echo -e "${YELLOW}Operation canceled.${NC}"
        exit 0
    fi
    
    echo -e "${YELLOW}Removing existing '${ENV_NAME}' environment...${NC}"
    conda env remove -n "${ENV_NAME}" -y
fi

# Create new environment from YAML file
echo -e "${YELLOW}Creating new environment from environment.yml...${NC}"
conda env create -f environment.yml

# Activate the environment
echo -e "${YELLOW}Activating '${ENV_NAME}' environment...${NC}"
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

# Install flash-attn with the correct flag
echo -e "${YELLOW}Installing flash-attn with no-build-isolation flag...${NC}"
pip uninstall -y flash-attn
pip install flash-attn==2.5.5 --no-build-isolation

# Install remaining packages that might be missing
echo -e "${YELLOW}Installing any missing packages...${NC}"
pip install accelerate tensorboardx wandb qwen-vl-utils

# Verify installation using the print_versions.py script
if [ -f "print_versions.py" ]; then
    echo -e "${YELLOW}Verifying installation...${NC}"
    python print_versions.py
fi

echo -e "${GREEN}Environment setup complete!${NC}"
echo -e "${YELLOW}To activate this environment, use: ${GREEN}conda activate ${ENV_NAME}${NC}" 