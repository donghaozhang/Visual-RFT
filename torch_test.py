import os
import torch
import torch.distributed as dist

def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Initialize the process group
    dist.init_process_group("nccl")
    
    # Get device info
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    # Get some basic info
    print(f"Running on node: {os.uname().nodename}")
    print(f"Rank: {rank}, Local Rank: {local_rank}, World Size: {world_size}")
    print(f"Device: {device}, Device name: {torch.cuda.get_device_name(local_rank)}")
    
    # Create a tensor on the device
    x = torch.randn(10, 10).to(device)
    print(f"Tensor created on {device}: shape={x.shape}")
    
    # Synchronize all processes
    dist.barrier()
    
    if rank == 0:
        print("\nTorchrun with 2 GPUs is working correctly!")

if __name__ == "__main__":
    main() 