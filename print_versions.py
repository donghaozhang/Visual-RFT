#!/usr/bin/env python3
"""
Script to print current versions of key packages for RFT.
"""
import importlib
import sys
import subprocess

def get_version(package):
    try:
        if package == "pytorch-cuda":
            # Special case for CUDA version
            import torch
            return torch.version.cuda
        
        # Try to import and get version
        module = importlib.import_module(package.replace('-', '_'))
        if hasattr(module, '__version__'):
            return module.__version__
        elif hasattr(module, 'version'):
            return module.version
        else:
            return "Unknown version format"
    except ImportError:
        return "Not installed"
    except Exception as e:
        return f"Error: {str(e)}"

def get_pip_version(package):
    """Get version using pip"""
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'show', package],
            capture_output=True, text=True, check=False
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if line.startswith('Version:'):
                    return line.split(':', 1)[1].strip()
        return "Not found via pip"
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    # Core packages
    packages = [
        "python",
        "torch",
        "torchvision",
        "torchaudio",
        "pytorch-cuda",
        "flash_attn",
        "vllm",
        "transformers",
        "accelerate",
        "triton",
        "xformers",
        "deepspeed",
        "einops",
        "numpy",
        "wandb",
        "tensorboardx",
        "qwen_vl_utils"
    ]
    
    # Print Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"Python: {python_version}")
    
    # Print versions for remaining packages
    for package in packages[1:]:
        if package == "python":
            continue
            
        version = get_version(package)
        if version == "Not installed":
            # Try using pip as fallback
            pip_version = get_pip_version(package)
            if pip_version != "Not found via pip":
                version = pip_version + " (pip)"
        
        friendly_name = package.replace('_', '-')
        print(f"{friendly_name}: {version}")
    
    # Print CUDA availability
    try:
        import torch
        print(f"\nCUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"GPU count: {device_count}")
            for i in range(device_count):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    except Exception as e:
        print(f"Error checking CUDA: {e}")