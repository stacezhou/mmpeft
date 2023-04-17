from ..builder import PEFT
import math
class BasePEFT:
    def __init__(self, freeze_other_params=True):
        self.freeze_other_params = freeze_other_params

    def __call__(self, module):
        if self.freeze_other_params:
            for param in module.parameters():
                param.requires_grad = False

        self._recur_add_parameter(module)
        self._recur_change_forward(module)
        self.count_train_params(module)
        return module

    def count_train_params(self, module):
        params_to_update = []
        for param in module.parameters():
            if param.requires_grad:
                params_to_update.append(param)
        from mmengine.logging import MMLogger
        logger = MMLogger.get_current_instance()
        total_params = sum(p.numel() for p in module.parameters())
        logger.info(f"Training params count: {total_params}")
        module._num_trainable_params = total_params
    
    def add_parameter(self, module, child, path):
        # Add parameters to the module
        raise NotImplementedError
    
    def change_forward(self, module, child, path):
        # Change the forward function of the module
        raise NotImplementedError

    def _recur_add_parameter(self, module, path='.'):
        # Recursively add peft parameters to the module
        for name, child in list(module.named_children()):
            child_path = path + '.' + name
            self._recur_add_parameter(child, child_path)
            self.add_parameter(module, child, child_path)
            
    def _recur_change_forward(self, module, path='.'):
        for name, child in module.named_children():
            child_path = path + '.' + name
            self._recur_change_forward(child, child_path)
            self.change_forward(module, child, child_path)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch import Tensor
import regex as re

@PEFT.register_module()
class LoRA(BasePEFT):
    def __init__(self, rank, scale, pattern, **kwargs):
        super(LoRA, self).__init__(**kwargs)
        self.rank = rank
        self.scale = scale
        self.pattern = re.compile(pattern)
    
    def add_parameter(self, module, child, path):
        if type(module) == nn.MultiheadAttention and bool(self.pattern.search(path)):
            if module._qkv_same_embed_dim:
                dim_q = dim_k = module.embed_dim
            else:
                dim_q = module.q_proj_weight.size(1)
                dim_k = module.k_proj_weight.size(1)
            
            module.Wqa = nn.Parameter(torch.empty(self.rank, dim_q))
            module.Wqb = nn.Parameter(torch.empty(self.rank, dim_q))
            module.Wka = nn.Parameter(torch.empty(self.rank, dim_k))
            module.Wkb = nn.Parameter(torch.empty(self.rank, dim_k))
            module.scale = self.scale
            nn.init.kaiming_uniform_(module.Wqa, a=math.sqrt(5))
            nn.init.zeros_(module.Wqb)
            nn.init.kaiming_uniform_(module.Wka, a=math.sqrt(5))
            nn.init.zeros_(module.Wkb)

    def change_forward(self, module, child, path):
        if type(module) == nn.MultiheadAttention and bool(self.pattern.search(path)):
            module.forward = lora_MHA_forward.__get__(module, nn.MultiheadAttention)


def lora_MHA_forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True, attn_mask: Optional[Tensor] = None,
            average_attn_weights: bool = True) -> Tuple[Tensor, Optional[Tensor]]:
    ### LoRA: Low-Rank Attention
    in_proj_weight = self.in_proj_weight
    delta_q = self.Wqa.t() @ self.Wqb * self.scale / self.Wqa.size(0)
    delta_k = self.Wka.t() @ self.Wkb * self.scale / self.Wka.size(0)
    if not self._qkv_same_embed_dim:
        q_proj_weight=self.q_proj_weight + delta_q
        k_proj_weight=self.k_proj_weight + delta_k
    else:
        dim = delta_q.size(1)
        in_proj_weight = torch.cat([
            in_proj_weight[:dim, :] + delta_q,
            in_proj_weight[dim:2*dim, :] + delta_k,
            in_proj_weight[2*dim:, :]
        ]) 
    ### LoRA: Low-Rank Attention
    is_batched = query.dim() == 3
    if self.batch_first and is_batched:
        query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

    if not self._qkv_same_embed_dim:
        attn_output, attn_output_weights = F.multi_head_attention_forward(
            query, key, value, self.embed_dim, self.num_heads,
            in_proj_weight, self.in_proj_bias,
            self.bias_k, self.bias_v, self.add_zero_attn,
            self.dropout, self.out_proj.weight, self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask, need_weights=need_weights,
            attn_mask=attn_mask, use_separate_proj_weight=True,
            q_proj_weight=q_proj_weight, k_proj_weight=k_proj_weight,
            v_proj_weight=self.v_proj_weight, average_attn_weights=average_attn_weights)
    else:
        attn_output, attn_output_weights = F.multi_head_attention_forward(
            query, key, value, self.embed_dim, self.num_heads,
            in_proj_weight, self.in_proj_bias,
            self.bias_k, self.bias_v, self.add_zero_attn,
            self.dropout, self.out_proj.weight, self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask, need_weights=need_weights,
            attn_mask=attn_mask, average_attn_weights=average_attn_weights)
    if self.batch_first and is_batched:
        return attn_output.transpose(1, 0), attn_output_weights
    else:
        return attn_output, attn_output_weights