#!/bin/bash
# Training script optimized for 2 NVIDIA RTX A6000 GPUs (48GB each)
set -e  # Exit on error

# Ensure conda is available and activate RFTV4 environment
echo "Activating rft conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate rft

# Check if the environment was activated properly
if [[ $CONDA_DEFAULT_ENV != "rft" ]]; then
  echo "Error: Failed to activate rft conda environment"
  exit 1
fi

echo "Using conda environment: rft"
python -c "import sys; print(f'Python interpreter: {sys.executable}')"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
export HF_HOME=/mnt/data/jiaxing.zjx/cache/huggingface/
export HF_DATASETS_CACHE=/home/ubuntu/visualrft/Visual-RFT/hf_cache
export DEBUG_MODE="true"
export LOG_PATH="./debug_log_2b_GRPO_2gpu.txt"

# Dataset and model paths - update these as needed
export DATA_PATH=./share_data/ViRFT_COCO_base65
export CKPT_PATH=./share_models/Qwen2-VL-2B-Instruct
export SAVE_PATH=./share_models/Qwen2-VL-2B-Instruct_GRPO_2gpu

# Make sure the output directory exists
mkdir -p ${SAVE_PATH}

echo "Starting training with 1 GPU..."
# Run training with 2 GPUs
torchrun --nproc_per_node=1 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=12345 \
    src/virft/src/open_r1/grpo.py \
    --output_dir ${SAVE_PATH} \
    --model_name_or_path ${CKPT_PATH} \
    --dataset_name ${DATA_PATH} \
    --deepspeed src/virft/local_scripts/zero3.json \
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
    --num_generations 8

echo "Training completed successfully!" 