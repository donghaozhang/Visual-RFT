#!/bin/bash

# Set color codes for output formatting
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Define source and target environment names
SOURCE_ENV="rftv3"
TARGET_ENV="rftv3_backup"

echo -e "${YELLOW}Cloning conda environment '${SOURCE_ENV}' to '${TARGET_ENV}'...${NC}"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Conda is not installed or not in PATH. Please install conda first.${NC}"
    exit 1
fi

# Check if source environment exists
if ! conda info --envs | grep -q "${SOURCE_ENV}"; then
    echo -e "${RED}Source environment '${SOURCE_ENV}' does not exist!${NC}"
    exit 1
fi

# Check if target environment already exists
if conda info --envs | grep -q "${TARGET_ENV}"; then
    echo -e "${YELLOW}Target environment '${TARGET_ENV}' already exists. Do you want to replace it? (y/n)${NC}"
    read -r answer
    if [[ "$answer" != "y" ]]; then
        echo -e "${YELLOW}Operation canceled.${NC}"
        exit 0
    fi
    
    echo -e "${YELLOW}Removing existing '${TARGET_ENV}' environment...${NC}"
    conda env remove -n "${TARGET_ENV}" -y
fi

# Clone environment using conda create command
echo -e "${YELLOW}Cloning environment using conda...${NC}"
conda create --name "${TARGET_ENV}" --clone "${SOURCE_ENV}" -y

# Install flash-attn with the correct flag
echo -e "${YELLOW}Installing flash-attn with no-build-isolation flag in the new environment...${NC}"
conda run -n "${TARGET_ENV}" pip uninstall -y flash-attn
conda run -n "${TARGET_ENV}" pip install flash-attn==2.5.5 --no-build-isolation

echo -e "${GREEN}Environment cloning complete!${NC}"
echo -e "${YELLOW}Original environment: ${GREEN}${SOURCE_ENV}${NC}"
echo -e "${YELLOW}Cloned environment: ${GREEN}${TARGET_ENV}${NC}"
echo -e "${YELLOW}To activate the cloned environment, use: ${GREEN}conda activate ${TARGET_ENV}${NC}" 