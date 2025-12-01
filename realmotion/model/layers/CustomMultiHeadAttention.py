import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CustomMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, add_bias_kv=True, batch_first=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.batch_first = batch_first
        self.dropout = dropout
        
        # Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=add_bias_kv)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=add_bias_kv)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=add_bias_kv)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=add_bias_kv)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        # Initialize like PyTorch's MultiheadAttention
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.)
            nn.init.constant_(self.k_proj.bias, 0.)
            nn.init.constant_(self.v_proj.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
    
    
    def forward(self, query, key, value, key_padding_mask=None, need_weights=False, 
                attn_mask=None, average_attn_weights=True):
        """
        Args:
            query: (L, N, E) if batch_first=False else (N, L, E)
            key: (S, N, E) if batch_first=False else (N, S, E)
            value: (S, N, E) if batch_first=False else (N, S, E)
            key_padding_mask: (N, S) or (N * num_heads, 1, S)
            attn_mask: (L, S) or (N * num_heads, L, S)
        Returns:
            attn_output: (L, N, E) if batch_first=False else (N, L, E)
            attn_weights: (N, L, S) or (N * num_heads, L, S) if need_weights
        """
        # Handle batch_first
        if self.batch_first:
            query = query.transpose(0, 1)  # (N, L, E) -> (L, N, E)
            key = key.transpose(0, 1)      # (N, S, E) -> (S, N, E)
            value = value.transpose(0, 1)  # (N, S, E) -> (S, N, E)
        
        # Store original shapes
        L, N, E = query.shape
        S, N_k, E_v = key.shape
        assert N == N_k, "batch size mismatch between query and key"
        assert E == E_v, "embed_dim mismatch"
        
        # Project Q, K, V
        q = self.q_proj(query)  # (L, N, E)
        k = self.k_proj(key)    # (S, N, E)
        v = self.v_proj(value)  # (S, N, E)
        
        # Reshape for multi-head attention: (L, N, E) -> (L, N, num_heads, head_dim) -> (L, N * num_heads, head_dim)
        q = q.reshape(L, N, self.num_heads, self.head_dim).transpose(1, 2)  # (L, num_heads, N, head_dim)
        k = k.reshape(S, N, self.num_heads, self.head_dim).transpose(1, 2)  # (S, num_heads, N, head_dim)
        v = v.reshape(S, N, self.num_heads, self.head_dim).transpose(1, 2)  # (S, num_heads, N, head_dim)
        
        # Transpose for batched matmul: (num_heads, N, L, head_dim)
        q = q.transpose(0, 2)  # (num_heads, N, L, head_dim)
        k = k.transpose(0, 2)  # (num_heads, N, S, head_dim)
        v = v.transpose(0, 2)  # (num_heads, N, S, head_dim)
        
        # Reshape for efficient computation
        q = q.reshape(-1, L, self.head_dim)  # (num_heads * N, L, head_dim)
        k = k.reshape(-1, S, self.head_dim)  # (num_heads * N, S, head_dim)
        v = v.reshape(-1, S, self.head_dim)  # (num_heads * N, S, head_dim)
        
        # Compute attention scores
        attn_scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)  # (num_heads * N, L, S)
        
        # Apply attention mask if provided
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                # (L, S) -> (1, L, S) -> (num_heads * N, L, S)
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                # (N * num_heads, L, S) or (N, L, S)
                if attn_mask.size(0) != N * self.num_heads:
                    # (N, L, S) -> (N, 1, L, S) -> (N, num_heads, L, S) -> (num_heads * N, L, S)
                    attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1).reshape(N * self.num_heads, L, S)
            
            # Apply mask (mask where True means mask out)
            attn_scores = attn_scores.masked_fill(attn_mask, float('-inf'))
        
        # Apply key padding mask if provided
        if key_padding_mask is not None:
            if key_padding_mask.dim() == 2:
                # (N, S) -> (N, 1, S) -> (N, num_heads, 1, S) -> (num_heads * N, 1, S)
                key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(1)
                key_padding_mask = key_padding_mask.repeat(1, self.num_heads, 1, 1)
                key_padding_mask = key_padding_mask.reshape(N * self.num_heads, 1, S)
            elif key_padding_mask.dim() == 3:
                # (num_heads * N, 1, S) or (N, 1, S)
                if key_padding_mask.size(0) != N * self.num_heads:
                    key_padding_mask = key_padding_mask.repeat(self.num_heads, 1, 1)
            
            # Apply mask (mask where True means mask out)
            attn_scores = attn_scores.masked_fill(key_padding_mask, float('-inf'))
        
        # Apply softmax
        attn_weights = F.softmax(attn_scores, dim=-1)  # (num_heads * N, L, S)
        
        # Apply dropout
        if self.dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # Compute attention output
        attn_output = torch.bmm(attn_weights, v)  # (num_heads * N, L, head_dim)
        
        # Reshape back: (num_heads * N, L, head_dim) -> (num_heads, N, L, head_dim) -> (L, num_heads, N, head_dim)
        attn_output = attn_output.reshape(self.num_heads, N, L, self.head_dim)
        attn_output = attn_output.transpose(0, 2)  # (L, num_heads, N, head_dim)
        
        # Combine heads: (L, num_heads, N, head_dim) -> (L, N, E)
        attn_output = attn_output.transpose(1, 2).reshape(L, N, E)  # (L, num_heads, N, head_dim) -> (L, N, num_heads, head_dim) -> (L, N, E)
        
        # Project output
        attn_output = self.out_proj(attn_output)  # (L, N, E)
        
        # Handle batch_first
        if self.batch_first:
            attn_output = attn_output.transpose(0, 1)  # (L, N, E) -> (N, L, E)
        
        # Prepare attention weights for return
        if need_weights:
            # Reshape attn_weights: (num_heads * N, L, S) -> (N, num_heads, L, S)
            attn_weights = attn_weights.reshape(N, self.num_heads, L, S)
            
            if average_attn_weights:
                # Average over heads: (N, L, S)
                attn_weights = attn_weights.mean(dim=1)
            
            return attn_output, attn_weights
        else:
            return attn_output, None