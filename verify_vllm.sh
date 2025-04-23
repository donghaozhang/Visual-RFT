#!/bin/bash

# Set color codes for output formatting
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Verifying Visual-RFT environment...${NC}"

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [[ "$PYTHON_VERSION" == "3.10" ]]; then
  echo -e "${GREEN}✓ Python version is 3.10${NC}"
else
  echo -e "${RED}✗ Python version is ${PYTHON_VERSION}, but expected 3.10${NC}"
fi

# Check PyTorch version
echo -e "${YELLOW}Checking PyTorch installation...${NC}"
if python -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null; then
  TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
  if [[ "$TORCH_VERSION" == "2.5.1"* ]]; then
    echo -e "${GREEN}✓ PyTorch 2.5.1 is installed.${NC}"
  else
    echo -e "${YELLOW}! PyTorch version is ${TORCH_VERSION}, but expected 2.5.1${NC}"
  fi
else
  echo -e "${RED}✗ PyTorch is not installed.${NC}"
fi

# Check if vLLM is installed
echo -e "${YELLOW}Verifying vLLM installation...${NC}"
if python -c "import vllm" 2>/dev/null; then
  echo -e "${GREEN}✓ vLLM is installed.${NC}"
  
  # Get vLLM version
  VERSION=$(python -c "import vllm; print(vllm.__version__)")
  echo -e "${GREEN}✓ vLLM version: ${VERSION}${NC}"
  
  # Check if version matches the expected version
  if [[ "$VERSION" == "0.7.2" ]]; then
    echo -e "${GREEN}✓ Correct vLLM version (0.7.2) is installed.${NC}"
  else
    echo -e "${YELLOW}! vLLM version is ${VERSION}, but expected 0.7.2.${NC}"
  fi

  # Check if vLLM can be initialized
  echo -e "${YELLOW}Testing vLLM initialization...${NC}"
  if python -c "from vllm import LLM; print('vLLM can be initialized successfully')" 2>/dev/null; then
    echo -e "${GREEN}✓ vLLM can be initialized.${NC}"
  else
    echo -e "${RED}✗ Failed to initialize vLLM. This might indicate a CUDA or other dependency issue.${NC}"
  fi
else
  echo -e "${RED}✗ vLLM is not installed. Please install it with:${NC}"
  echo -e "pip install vllm==0.7.2"
fi

# Check transformers version
echo -e "${YELLOW}Checking transformers version...${NC}"
if python -c "import transformers; print(f'Transformers {transformers.__version__}')" 2>/dev/null; then
  TRANSFORMERS_VERSION=$(python -c "import transformers; print(transformers.__version__)")
  if [[ "$TRANSFORMERS_VERSION" == "4.51.3" ]]; then
    echo -e "${GREEN}✓ Transformers 4.51.3 is installed.${NC}"
  else
    echo -e "${YELLOW}! Transformers version is ${TRANSFORMERS_VERSION}, but expected 4.51.3${NC}"
  fi
else
  echo -e "${RED}✗ Transformers is not installed.${NC}"
fi

# Check if flash-attn is installed
echo -e "${YELLOW}Checking flash-attention installation...${NC}"
if python -c "import flash_attn" 2>/dev/null; then
  echo -e "${GREEN}✓ flash-attn is installed.${NC}"
  FLASH_ATTN_VERSION=$(python -c "import flash_attn; print(flash_attn.__version__)")
  echo -e "${GREEN}✓ flash-attn version: ${FLASH_ATTN_VERSION}${NC}"
else
  echo -e "${RED}✗ flash-attn is not installed.${NC}"
fi

# Check xformers installation
echo -e "${YELLOW}Checking xformers installation...${NC}"
if python -c "import xformers; print(f'xformers is installed')" 2>/dev/null; then
  echo -e "${GREEN}✓ xformers is installed.${NC}"
  XFORMERS_VERSION=$(python -c "import xformers; print(getattr(xformers, '__version__', 'unknown'))")
  echo -e "${GREEN}✓ xformers version: ${XFORMERS_VERSION}${NC}"
else
  echo -e "${RED}✗ xformers is not installed.${NC}"
fi

# Check CUDA availability - Fixed syntax error
echo -e "${YELLOW}Checking CUDA availability...${NC}"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]" && echo -e "${GREEN}✓ CUDA check completed.${NC}" || echo -e "${RED}✗ CUDA check failed.${NC}"

echo -e "${YELLOW}Verification complete.${NC}" 