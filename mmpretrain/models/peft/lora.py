from ..builder import PEFT
from .base import BasePEFT
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch import Tensor
import regex as re
import math
class LoRAMultiheadAttention(nn.Module):
    def __init__(self,origin_MHA, rank, scale):
        super().__init__()
        self.__dict__['in_proj_weight'] = origin_MHA.in_proj_weight
        self.__dict__['in_proj_bias'] = origin_MHA.in_proj_bias
        self.__dict__['out_proj'] = origin_MHA.out_proj
        d_model = origin_MHA.embed_dim
        self.d_model = d_model
        self.n_head = origin_MHA.num_heads
        self._ft_k_proj_a = nn.Parameter(torch.empty(rank, d_model))
        self._ft_k_proj_b = nn.Parameter(torch.empty(rank, d_model))
        self._ft_v_proj_a = nn.Parameter(torch.empty(rank, d_model))
        self._ft_v_proj_b = nn.Parameter(torch.empty(rank, d_model))
        self.rank = rank
        self.scale = scale
        nn.init.kaiming_uniform_(self._ft_k_proj_a, a=math.sqrt(6))
        nn.init.kaiming_uniform_(self._ft_v_proj_a, a=math.sqrt(6))
        nn.init.zeros_(self._ft_k_proj_b) 
        nn.init.zeros_(self._ft_v_proj_b) 
    
    def forward(self, query, key, value, need_weights, attn_mask):
        q,k,v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        tgt_len, bsz, embed_dim = q.size()
        ############ lora ############
        k += self.scale / self.rank * query @ self._ft_k_proj_a.t() @ self._ft_k_proj_b
        v += self.scale / self.rank * query @ self._ft_v_proj_a.t() @ self._ft_v_proj_b

        ############ lora ############
        num_heads = self.n_head
        head_dim = embed_dim // num_heads
        q = q.view(tgt_len, bsz, num_heads, head_dim).transpose(0, 2).reshape(-1, tgt_len, head_dim)  
        k = k.view(tgt_len, bsz, num_heads, head_dim).transpose(0, 2).reshape(-1, tgt_len, head_dim) 
        v = v.view(tgt_len, bsz, num_heads, head_dim).transpose(0, 2).reshape(-1, tgt_len, head_dim) 
        # (num_heads, batch_size, seq_len, head_dim)

        # Compute self-attention scores and apply mask if provided.
        scores = torch.bmm(q / math.sqrt(head_dim), k.transpose(-2, -1))
        # (num_heads * batch_size, tgt_len_q, tgt_len_k)

        # Apply softmax to get attention weights.
        attn_output = F.softmax(scores, dim=-1)
        # Apply attention weights to values to get context vectors.
        attn_output = torch.matmul(attn_output, v)
        # (num_heads * batch_size, tgt_len_q, embed_dim)

        # Concatenate and project context vectors back to the original dimensionality.
        attn_output = attn_output.contiguous().view(num_heads, bsz, tgt_len, head_dim).transpose(0, 2)
        attn_output = attn_output.reshape(tgt_len, bsz, embed_dim)
        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)
        # Return attention output and attention weights if requested.
        return attn_output, None
       
@PEFT.register_module()
class LoRA(BasePEFT):
    def __init__(self, rank, scale, pattern, **kwargs):
        super(LoRA, self).__init__(**kwargs)
        self.rank = rank
        self.scale = scale
        self.pattern = re.compile(pattern)
    
    def add_parameter(self, module, child, path):
        if type(child) == nn.MultiheadAttention and bool(self.pattern.search(path)):
            module.lora_MHA = LoRAMultiheadAttention(child, self.rank, self.scale)           
            
    def change_forward(self, module, child, path):
        if type(child) == nn.MultiheadAttention and bool(self.pattern.search(path)):
            lora_MHA = getattr(module, 'lora_MHA')
            child.forward = lora_MHA.forward
