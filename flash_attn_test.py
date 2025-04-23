import torch

def test_flash_attention():
    try:
        from flash_attn import flash_attn_func
        print('Flash Attention is installed correctly')
        
        # Create sample tensors to test flash attention - using fp16
        batch_size, seq_len, n_heads, d_head = 2, 1024, 8, 64
        q = torch.randn(batch_size, seq_len, n_heads, d_head, device='cuda', dtype=torch.float16)
        k = torch.randn(batch_size, seq_len, n_heads, d_head, device='cuda', dtype=torch.float16)
        v = torch.randn(batch_size, seq_len, n_heads, d_head, device='cuda', dtype=torch.float16)
        
        # Run flash attention
        out = flash_attn_func(q, k, v)
        print(f'Flash Attention test successful! Output shape: {out.shape}')
        print(f'Flash Attention version information: {flash_attn_func.__module__}')
        return True
    except ImportError as e:
        print(f'Error importing flash_attn: {e}')
        return False
    except Exception as e:
        print(f'Error running flash_attn test: {e}')
        return False

if __name__ == "__main__":
    test_flash_attention() 