#!/bin/bash

# Create and activate the conda environment
echo "Creating conda environment from environment.yml..."
conda env create -f environment.yml
conda activate visual-rft

# Uninstall and reinstall flash-attn with the proper flags
echo "Installing flash-attn with no-build-isolation flag..."
pip uninstall -y flash-attn
pip install flash-attn==2.5.5 --no-build-isolation

# Install the package in development mode
echo "Installing package in development mode..."
cd src/virft
pip install -e ".[dev]"

echo "Environment setup complete!" 