from __future__ import annotations
from collections.abc import Iterable
from pathlib import Path
from typing import BinaryIO
import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from src.pa1 import (
    AdamW, Embedding, GroupedQuerySelfAttention, Linear, RMSNorm,
    RotaryPositionalEmbedding, SwiGLU, TransformerBlock, TransformerLM,
    cross_entropy, get_batch, get_lr_cosine_schedule, gradient_clipping,
    load_checkpoint, load_token_array, save_checkpoint,
    scaled_dot_product_attention, silu, softmax,
)

def run_load_token_array(path): return load_token_array(path)
def run_get_batch(dataset,batch_size,sequence_length,device,generator): return get_batch(dataset,batch_size,sequence_length,device,generator)
def run_linear(d_in,d_out,weights,in_features):
    m=Linear(d_in,d_out,device=weights.device,dtype=weights.dtype); m.weight.data.copy_(weights); return m(in_features)
def run_embedding(vocab_size,d_model,weights,token_ids):
    m=Embedding(vocab_size,d_model,device=weights.device,dtype=weights.dtype); m.weight.data.copy_(weights); return m(token_ids)
def run_rmsnorm(d_model,norm_eps,weights,in_features):
    m=RMSNorm(d_model,norm_eps,device=weights.device,dtype=weights.dtype); m.weight.data.copy_(weights); return m(in_features)
def run_silu(in_features): return silu(in_features)
def run_swiglu(d_model,d_ff,gate_weight,down_weight,up_weight,in_features):
    m=SwiGLU(d_model,d_ff,device=gate_weight.device,dtype=gate_weight.dtype)
    with torch.no_grad(): m.gate.weight.copy_(gate_weight); m.down.weight.copy_(down_weight); m.up.weight.copy_(up_weight)
    return m(in_features)
def run_rope(head_dim,rope_theta,context_length,in_query_or_key,token_positions):
    return RotaryPositionalEmbedding(rope_theta,head_dim,context_length,device=in_query_or_key.device)(in_query_or_key, token_positions)
def run_softmax(in_features,dim): return softmax(in_features,dim)
def run_scaled_dot_product_attention(queries,keys,values,mask=None): return scaled_dot_product_attention(queries,keys,values,mask)
def run_grouped_query_self_attention(d_model,n_q_heads,n_kv_heads,context_length,rope_theta,q_proj_weight,k_proj_weight,v_proj_weight,output_proj_weight,in_features,token_positions=None):
    m=GroupedQuerySelfAttention(d_model,n_q_heads,n_kv_heads,context_length,rope_theta,device=q_proj_weight.device,dtype=q_proj_weight.dtype)
    with torch.no_grad():
        m.q_proj.weight.copy_(q_proj_weight);m.k_proj.weight.copy_(k_proj_weight);m.v_proj.weight.copy_(v_proj_weight);m.out_proj.weight.copy_(output_proj_weight)
    return m(in_features, token_positions=token_positions)
def run_transformer_block(d_model,n_q_heads,n_kv_heads,d_ff,context_length,rope_theta,weights,in_features,token_positions=None,norm_eps=1e-5):
    first=next(iter(weights.values())); m=TransformerBlock(d_model,n_q_heads,n_kv_heads,d_ff,context_length,rope_theta,norm_eps=norm_eps,device=first.device,dtype=first.dtype);m.load_state_dict(weights,strict=True);return m(in_features,token_positions=token_positions)
def run_transformer_lm(vocab_size,context_length,d_model,num_layers,n_q_heads,n_kv_heads,d_ff,rope_theta,weights,token_ids,token_positions=None,norm_eps=1e-5):
    first=next(iter(weights.values()));m=TransformerLM(vocab_size,context_length,d_model,num_layers,n_q_heads,n_kv_heads,d_ff,rope_theta,norm_eps=norm_eps,device=first.device,dtype=first.dtype);m.load_state_dict(weights,strict=True);return m(token_ids,token_positions=token_positions)
def get_transformer_lm(vocab_size,context_length,d_model,num_layers,n_q_heads,n_kv_heads,d_ff,rope_theta,*,norm_eps=1e-5,device=None,dtype=None): return TransformerLM(vocab_size,context_length,d_model,num_layers,n_q_heads,n_kv_heads,d_ff,rope_theta,norm_eps=norm_eps,device=device,dtype=dtype)
def run_cross_entropy(logits,targets): return cross_entropy(logits,targets)
def get_adamw_cls(): return AdamW
def run_get_lr_cosine_schedule(step,learning_rate_max,learning_rate_min,warmup_steps,cosine_steps): return get_lr_cosine_schedule(step,learning_rate_max,learning_rate_min,warmup_steps,cosine_steps)
def run_gradient_clipping(parameters,max_l2_norm): return gradient_clipping(parameters,max_l2_norm)
def run_save_checkpoint(model,optimizer,next_step,train_generator,val_generator,out): return save_checkpoint(model,optimizer,next_step,train_generator,val_generator,out)
def run_load_checkpoint(src,model,optimizer,train_generator,val_generator): return load_checkpoint(src,model,optimizer,train_generator,val_generator)
