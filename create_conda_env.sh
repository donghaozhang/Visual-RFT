#!/bin/bash
# Script to create and configure RFTV4 conda environment

set -e  # Exit on error
# source /home/ubuntu/petertest/miniconda/bin/activate
# conda create -n rftv2 python=3.10 -y
# conda activate rftv3
# pip3 install torch torchvision torchaudio
echo "Creating RFTV6 conda environment from specification..."
conda env create -f conda_environment.yml

echo "Activating RFTV6 conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate RFTV6

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

# Successfully installed MarkupSafe-3.0.2 aiohappyeyeballs-2.6.1 aiohttp-3.11.18 aiohttp_cors-0.8.1 aiosignal-1.3.2 airportsdata-20250224 annotated-types-0.7.0 anyio-4.9.0 astor-0.8.1 async-timeout-5.0.1 attrs-25.3.0 blake3-1.0.4 cachetools-5.5.2 certifi-2025.1.31 charset-normalizer-3.4.1 click-8.1.8 cloudpickle-3.1.1 colorful-0.5.6 compressed-tensors-0.9.1 depyf-0.18.0 dill-0.4.0 diskcache-5.6.3 distlib-0.3.9 distro-1.9.0 exceptiongroup-1.2.2 fastapi-0.115.12 filelock-3.18.0 frozenlist-1.6.0 fsspec-2025.3.2 gguf-0.10.0 google-api-core-2.24.2 google-auth-2.39.0 googleapis-common-protos-1.70.0 grpcio-1.71.0 h11-0.14.0 httpcore-1.0.8 httptools-0.6.4 httpx-0.28.1 huggingface-hub-0.30.2 idna-3.10 importlib_metadata-8.6.1 interegular-0.3.3 jinja2-3.1.6 jiter-0.9.0 jsonschema-4.23.0 jsonschema-specifications-2025.4.1 lark-1.2.2 lm-format-enforcer-0.10.11 mistral_common-1.5.4 mpmath-1.3.0 msgpack-1.1.0 msgspec-0.19.0 multidict-6.4.3 nest_asyncio-1.6.0 networkx-3.4.2 numpy-1.26.4 nvidia-cublas-cu12-12.4.5.8 nvidia-cuda-cupti-cu12-12.4.127 nvidia-cuda-nvrtc-cu12-12.4.127 nvidia-cuda-runtime-cu12-12.4.127 nvidia-cudnn-cu12-9.1.0.70 nvidia-cufft-cu12-11.2.1.3 nvidia-curand-cu12-10.3.5.147 nvidia-cusolver-cu12-11.6.1.9 nvidia-cusparse-cu12-12.3.1.170 nvidia-ml-py-12.570.86 nvidia-nccl-cu12-2.21.5 nvidia-nvjitlink-cu12-12.4.127 nvidia-nvtx-cu12-12.4.127 openai-1.75.0 opencensus-0.11.4 opencensus-context-0.1.3 opencv-python-headless-4.11.0.86 outlines-0.1.11 outlines_core-0.1.26 packaging-25.0 partial-json-parser-0.2.1.1.post5 pillow-11.2.1 platformdirs-4.3.7 prometheus-fastapi-instrumentator-7.1.0 prometheus_client-0.21.1 propcache-0.3.1 proto-plus-1.26.1 protobuf-6.30.2 psutil-7.0.0 py-cpuinfo-9.0.0 py-spy-0.4.0 pyasn1-0.6.1 pyasn1-modules-0.4.2 pycountry-24.6.1 pydantic-2.11.3 pydantic-core-2.33.1 python-dotenv-1.1.0 pyyaml-6.0.2 pyzmq-26.4.0 ray-2.44.1 referencing-0.36.2 regex-2024.11.6 requests-2.32.3 rpds-py-0.24.0 rsa-4.9.1 safetensors-0.5.3 sentencepiece-0.2.0 six-1.17.0 smart_open-7.1.0 sniffio-1.3.1 starlette-0.46.2 sympy-1.13.1 tiktoken-0.9.0 tokenizers-0.21.1 torch-2.5.1 torchaudio-2.5.1 torchvision-0.20.1 tqdm-4.67.1 transformers-4.51.3 triton-3.1.0 typing-inspection-0.4.0 typing_extensions-4.13.2 urllib3-2.4.0 uvicorn-0.34.2 uvloop-0.21.0 virtualenv-20.30.0 vllm-0.7.2 watchfiles-1.0.5 websockets-15.0.1 wrapt-1.17.2 xformers-0.0.28.post3 xgrammar-0.1.18 yarl-1.20.0 zipp-3.21.0