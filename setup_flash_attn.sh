#!/bin/bash
# Script to create and set up a conda environment for FlashAttention

set -e  # Exit on error

echo "Creating flash-attn conda environment..."
conda env create -f conda_flash_attn.yml

echo "Activating flash-attn environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate flash-attn

# Verify PyTorch and CUDA
echo "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}'); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"

# Test Flash Attention
echo "Testing Flash Attention..."
python -c "
try:
    from flash_attn import flash_attn_func
    print('Flash Attention is installed correctly')
    
    # Create sample tensors to test flash attention - using fp16
    import torch
    batch_size, seq_len, n_heads, d_head = 2, 1024, 8, 64
    q = torch.randn(batch_size, seq_len, n_heads, d_head, device='cuda', dtype=torch.float16)
    k = torch.randn(batch_size, seq_len, n_heads, d_head, device='cuda', dtype=torch.float16)
    v = torch.randn(batch_size, seq_len, n_heads, d_head, device='cuda', dtype=torch.float16)
    
    # Run flash attention
    out = flash_attn_func(q, k, v)
    print(f'Flash Attention test successful! Output shape: {out.shape}')
    
    # Try with causal mask (important for decoder-only models)
    causal_out = flash_attn_func(q, k, v, causal=True)
    print(f'Causal Flash Attention test successful! Output shape: {causal_out.shape}')
    
    # Display version info
    import flash_attn
    print(f'Flash Attention version: {flash_attn.__version__}')
except Exception as e:
    print(f'Error with Flash Attention: {e}')
"

echo "Creating a sample FlashAttention implementation..."
cat > flash_attn_example.py << 'EOL'
import torch
import torch.nn as nn
from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
from flash_attn.bert_padding import pad_input, unpad_input

# Simple Flash Attention module
class FlashAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_rate=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout_rate = dropout_rate
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x, attention_mask=None, causal=False):
        batch_size, seq_len, _ = x.shape
        
        # Project to q, k, v
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Handle attention mask if provided
        if attention_mask is not None:
            # Convert mask from [batch_size, seq_len] to [batch_size, 1, 1, seq_len]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # 1 = keep, 0 = mask out
            cu_seqlens_q = torch.zeros(batch_size + 1, device=x.device, dtype=torch.int32)
            cu_seqlens_k = torch.zeros(batch_size + 1, device=x.device, dtype=torch.int32)
            
            for i in range(batch_size):
                cu_seqlens_q[i + 1] = cu_seqlens_q[i] + attention_mask[i].sum()
                cu_seqlens_k[i + 1] = cu_seqlens_k[i] + attention_mask[i].sum()
            
            # Use flash_attn_varlen_func for variable length sequences
            from flash_attn import flash_attn_varlen_func
            x_unpad, indices, cu_seqlens, max_seqlen = unpad_input(x, attention_mask.squeeze())
            q_unpad = self.q_proj(x_unpad).view(-1, self.num_heads, self.head_dim)
            k_unpad = self.k_proj(x_unpad).view(-1, self.num_heads, self.head_dim)
            v_unpad = self.v_proj(x_unpad).view(-1, self.num_heads, self.head_dim)
            
            output_unpad = flash_attn_varlen_func(
                q_unpad, k_unpad, v_unpad, 
                cu_seqlens, cu_seqlens, 
                max_seqlen, max_seqlen,
                dropout_p=self.dropout_rate, 
                causal=causal
            )
            
            output = pad_input(output_unpad, indices, batch_size, seq_len)
            output = self.out_proj(output)
            
        else:
            # Use standard flash_attn_func
            output = flash_attn_func(
                q, k, v,
                dropout_p=self.dropout_rate,
                causal=causal
            )
            
            # Reshape and project back
            output = output.contiguous().view(batch_size, seq_len, self.hidden_size)
            output = self.out_proj(output)
            
        return output

print("Example model created successfully!")
EOL

echo "Environment setup completed!"
echo "To activate: conda activate flash-attn" 