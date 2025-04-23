#!/bin/bash
# Training script optimized for 2 NVIDIA RTX A6000 GPUs (48GB each)
set -e  # Exit on error

# Ensure conda is available and activate rftv3 environment
echo "Activating rftv3 conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate rftv3

# Check if the environment was activated properly
if [[ $CONDA_DEFAULT_ENV != "rftv3" ]]; then
  echo "Error: Failed to activate rftv3 conda environment"
  exit 1
fi

echo "Using conda environment: rftv3"
python -c "import sys; print(f'Python interpreter: {sys.executable}')"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

export DEBUG_MODE="true"
export LOG_PATH="./debug_log_2b_GRPO_2gpu.txt"

# Use absolute paths instead of relative paths
ROOT_DIR="/home/ubuntu/petertest/new_start/Visual-RFT"
export DATA_PATH="${ROOT_DIR}/share_data/base65cate_6k_think"
export CKPT_PATH="${ROOT_DIR}/share_models/Qwen2-VL-2B-Instruct"
export SAVE_PATH="${ROOT_DIR}/share_models/Qwen2-VL-2B-Instruct_GRPO_2gpu"
export DS_CONFIG="${ROOT_DIR}/src/virft/local_scripts/zero3.json"

# Make sure the model and output directories exist
if [ ! -d "${CKPT_PATH}" ]; then
  echo "Error: Model path ${CKPT_PATH} does not exist"
  echo "Please download the model or update the path"
  exit 1
fi

# Make sure the output directory exists
mkdir -p "${SAVE_PATH}"

# Make sure the dataset exists
if [ ! -f "${DATA_PATH}/dataset_dict.json" ]; then
  echo "Error: Dataset not found at ${DATA_PATH}"
  echo "Please run the download_dataset.sh script first"
  exit 1
fi

echo "Starting training with 2 GPUs..."
echo "- Model path: ${CKPT_PATH}"
echo "- Dataset path: ${DATA_PATH}"
echo "- Output path: ${SAVE_PATH}"
echo "- DeepSpeed config: ${DS_CONFIG}"

# Run training with 2 GPUs
torchrun --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=12345 \
    src/virft/src/open_r1/grpo.py \
    --output_dir "${SAVE_PATH}" \
    --model_name_or_path "${CKPT_PATH}" \
    --dataset_name "${DATA_PATH}" \
    --deepspeed "${DS_CONFIG}" \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 3 \
    --run_name Qwen2-VL-2B_GRPO_2gpu \
    --save_steps 100 \
    --save_only_model true \
    --num_generations 8 \
    --trust_remote_code true

echo "Training completed successfully!" 