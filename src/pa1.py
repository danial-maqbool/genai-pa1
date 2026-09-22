from __future__ import annotations

import csv
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Iterable

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.optim import Optimizer


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(in_features, int) or isinstance(in_features, bool) or in_features <= 0:
            raise ValueError("in_features must be a positive integer")
        if not isinstance(out_features, int) or isinstance(out_features, bool) or out_features <= 0:
            raise ValueError("out_features must be a positive integer")
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected final dimension {self.in_features}, got {x.shape[-1]}")
        return x @ self.weight.transpose(-1, -2)


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(num_embeddings, int) or isinstance(num_embeddings, bool) or num_embeddings <= 0:
            raise ValueError("num_embeddings must be a positive integer")
        if not isinstance(embedding_dim, int) or isinstance(embedding_dim, bool) or embedding_dim <= 0:
            raise ValueError("embedding_dim must be a positive integer")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        if token_ids.dtype not in _INTEGER_DTYPES:
            raise TypeError("token_ids must have an integer dtype")
        return self.weight[token_ids.long()]


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        norm_eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(d_model, int) or isinstance(d_model, bool) or d_model <= 0:
            raise ValueError("d_model must be a positive integer")
        if norm_eps < 0:
            raise ValueError("norm_eps must be nonnegative")
        self.d_model = d_model
        self.norm_eps = float(norm_eps)
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        if not x.is_floating_point():
            raise TypeError("RMSNorm input must be floating point")
        if x.shape[-1] != self.d_model:
            raise ValueError(f"expected final dimension {self.d_model}, got {x.shape[-1]}")
        in_dtype = x.dtype
        if in_dtype in (torch.float16, torch.bfloat16):
            work = x.to(torch.float32)
            gain = self.weight.to(torch.float32)
        else:
            work = x
            gain = self.weight.to(work.dtype)
        inv_rms = torch.rsqrt(work.square().mean(dim=-1, keepdim=True) + self.norm_eps)
        return (work * inv_rms * gain).to(in_dtype)


def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.gate = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.up = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.down = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(silu(self.gate(x)) * self.up(x))


_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}
if hasattr(torch, "uint16"):
    _INTEGER_DTYPES.add(torch.uint16)
if hasattr(torch, "uint32"):
    _INTEGER_DTYPES.add(torch.uint32)
if hasattr(torch, "uint64"):
    _INTEGER_DTYPES.add(torch.uint64)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        rope_theta: float,
        head_dim: int,
        context_length: int,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(head_dim, int) or isinstance(head_dim, bool) or head_dim <= 0:
            raise ValueError("head_dim must be a positive integer")
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for adjacent-pair RoPE")
        if not isinstance(context_length, int) or isinstance(context_length, bool) or context_length <= 0:
            raise ValueError("context_length must be a positive integer")
        if not isinstance(rope_theta, (int, float)) or rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
        self.rope_theta = float(rope_theta)
        self.head_dim = head_dim
        self.context_length = context_length

        pair_coordinates = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = self.rope_theta ** (-pair_coordinates / head_dim)
        positions = torch.arange(context_length, dtype=torch.float32, device=device)
        angles = positions[:, None] * inv_freq[None, :]
        self.register_buffer("cos_cache", angles.cos(), persistent=False)
        self.register_buffer("sin_cache", angles.sin(), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        if x.ndim < 2:
            raise ValueError("RoPE input must have sequence and head_dim dimensions")
        if not x.is_floating_point():
            raise TypeError("RoPE input must be floating point")
        if x.shape[-1] != self.head_dim:
            raise ValueError(f"input head_dim is {x.shape[-1]}, expected head_dim={self.head_dim}")
        if token_positions.ndim < 1:
            raise ValueError("token_positions must have a sequence dimension")
        if token_positions.dtype not in _INTEGER_DTYPES:
            raise TypeError("token_positions must have an integer dtype")
        sequence_length = x.shape[-2]
        if token_positions.shape[-1] != sequence_length:
            raise ValueError("token_positions final dimension must match sequence length")

        x_batch_dims = x.ndim - 2
        pos_batch_dims = token_positions.ndim - 1
        if pos_batch_dims > x_batch_dims:
            raise ValueError("token_positions has too many leading dimensions")
        for pos_size, x_size in zip(token_positions.shape[:-1], x.shape[:pos_batch_dims]):
            if pos_size not in (1, x_size) and x_size != 1:
                raise ValueError("token_positions leading dimensions are not broadcastable")

        if token_positions.numel() > 0:
            pos_min = int(token_positions.min().item())
            pos_max = int(token_positions.max().item())
            if pos_min < 0 or pos_max >= self.context_length:
                raise ValueError(f"token_positions contains a position outside [0, {self.context_length})")

        indices = token_positions.to(device=self.cos_cache.device, dtype=torch.long)
        cos = self.cos_cache[indices]
        sin = self.sin_cache[indices]
        extra_singletons = (1,) * (x_batch_dims - pos_batch_dims)
        cache_shape = (*token_positions.shape[:-1], *extra_singletons, sequence_length, self.head_dim // 2)
        cos = cos.reshape(cache_shape)
        sin = sin.reshape(cache_shape)

        even = x[..., 0::2]
        odd = x[..., 1::2]
        out_even = even * cos - odd * sin
        out_odd = even * sin + odd * cos
        return torch.stack((out_even, out_odd), dim=-1).flatten(-2).to(x.dtype)


def softmax(x: Tensor, dim: int) -> Tensor:
    maximum = x.max(dim=dim, keepdim=True).values
    exp_shifted = torch.exp(x - maximum)
    return exp_shifted / exp_shifted.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    queries: Tensor,
    keys: Tensor,
    values: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    if queries.ndim < 2 or keys.ndim < 2 or values.ndim < 2:
        raise ValueError("queries, keys, and values must each have at least two dimensions")
    if queries.shape[-1] != keys.shape[-1]:
        raise ValueError("query and key feature dimensions must match")
    if keys.shape[-2] != values.shape[-2]:
        raise ValueError("key and value sequence lengths must match")
    if queries.shape[-1] <= 0 or keys.shape[-2] <= 0:
        raise ValueError("attention feature and key sequence dimensions must be nonempty")

    scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(queries.shape[-1])
    if mask is not None:
        if mask.dtype != torch.bool:
            raise TypeError("attention mask must be boolean")
        if mask.ndim < 2 or mask.shape[-2:] != scores.shape[-2:]:
            raise ValueError("attention mask final dimensions must match query/key sequence dimensions")
        try:
            broadcast_shape = torch.broadcast_shapes(scores.shape, mask.shape)
        except RuntimeError as exc:
            raise ValueError("attention mask is not broadcastable to attention scores") from exc
        if tuple(broadcast_shape) != tuple(scores.shape):
            raise ValueError("attention mask may not expand the attention score batch dimensions")
        expanded_mask = torch.broadcast_to(mask, scores.shape)
        if (~expanded_mask).all(dim=-1).any().item():
            raise ValueError("attention mask leaves a query with no permitted key positions")
        scores = scores.masked_fill(~expanded_mask, -torch.inf)

    probabilities = softmax(scores, dim=-1)
    try:
        return torch.matmul(probabilities, values)
    except RuntimeError as exc:
        raise ValueError("attention leading dimensions are not broadcastable") from exc


class GroupedQuerySelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_q_heads: int,
        n_kv_heads: int,
        context_length: int,
        rope_theta: float,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        for name, value in (("d_model", d_model), ("n_q_heads", n_q_heads), ("n_kv_heads", n_kv_heads)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if d_model % n_q_heads != 0:
            raise ValueError("d_model must be divisible by n_q_heads")
        if n_q_heads % n_kv_heads != 0:
            raise ValueError("n_q_heads must be divisible by n_kv_heads")
        self.d_model = d_model
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.context_length = context_length
        self.head_dim = d_model // n_q_heads
        self.group_size = n_q_heads // n_kv_heads

        self.q_proj = Linear(d_model, n_q_heads * self.head_dim, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, n_kv_heads * self.head_dim, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, n_kv_heads * self.head_dim, device=device, dtype=dtype)
        self.out_proj = Linear(n_q_heads * self.head_dim, d_model, device=device, dtype=dtype)
        self.rope = RotaryPositionalEmbedding(rope_theta, self.head_dim, context_length, device=device)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError("attention input must have shape [batch, sequence, d_model]")
        batch_size, sequence_length, _ = x.shape
        if sequence_length < 1 or sequence_length > self.context_length:
            raise ValueError("attention sequence length is outside the supported context window")
        if token_positions is None:
            token_positions = torch.arange(sequence_length, device=x.device, dtype=torch.long)

        q = self.q_proj(x).reshape(
            batch_size, sequence_length, self.n_kv_heads, self.group_size, self.head_dim
        ).permute(0, 2, 3, 1, 4)
        k = self.k_proj(x).reshape(
            batch_size, sequence_length, self.n_kv_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(
            batch_size, sequence_length, self.n_kv_heads, self.head_dim
        ).permute(0, 2, 1, 3)

        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)
        causal = torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=x.device).tril()
        attended = scaled_dot_product_attention(q, k.unsqueeze(2), v.unsqueeze(2), causal)
        merged = attended.permute(0, 3, 1, 2, 4).reshape(batch_size, sequence_length, self.d_model)
        return self.out_proj(merged)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_q_heads: int,
        n_kv_heads: int,
        d_ff: int,
        context_length: int,
        rope_theta: float,
        norm_eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.attention = GroupedQuerySelfAttention(
            d_model, n_q_heads, n_kv_heads, context_length, rope_theta, device=device, dtype=dtype
        )
        self.attention_norm = RMSNorm(d_model, norm_eps, device=device, dtype=dtype)
        self.ffn_norm = RMSNorm(d_model, norm_eps, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        x = x + self.attention(self.attention_norm(x), token_positions=token_positions)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        n_q_heads: int,
        n_kv_heads: int,
        d_ff: int,
        rope_theta: float,
        norm_eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(num_layers, int) or isinstance(num_layers, bool) or num_layers <= 0:
            raise ValueError("num_layers must be a positive integer")
        if not isinstance(context_length, int) or isinstance(context_length, bool) or context_length <= 0:
            raise ValueError("context_length must be a positive integer")
        self.context_length = context_length
        self.token_embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    n_q_heads,
                    n_kv_heads,
                    d_ff,
                    context_length,
                    rope_theta,
                    norm_eps=norm_eps,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(d_model, norm_eps, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: Tensor, token_positions: Tensor | None = None) -> Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        if token_ids.dtype not in _INTEGER_DTYPES:
            raise TypeError("token_ids must have an integer dtype")
        sequence_length = token_ids.shape[1]
        if sequence_length < 1 or sequence_length > self.context_length:
            raise ValueError("sequence length must satisfy 1 <= sequence_length <= context_length")
        if token_positions is not None and token_positions.shape[-1] != sequence_length:
            raise ValueError("token_positions final dimension must match sequence length")
        x = self.token_embedding(token_ids)
        for block in self.blocks:
            x = block(x, token_positions=token_positions)
        return self.lm_head(self.final_norm(x))


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    if logits.ndim < 1 or logits.shape[-1] <= 0:
        raise ValueError("logits must have a nonempty vocabulary dimension")
    if tuple(targets.shape) != tuple(logits.shape[:-1]):
        raise ValueError("targets must match all leading logits dimensions")
    if targets.dtype not in _INTEGER_DTYPES:
        raise TypeError("targets must have an integer dtype")
    if targets.numel() == 0:
        raise ValueError("cross_entropy requires at least one target")
    vocab_size = logits.shape[-1]
    if int(targets.min().item()) < 0 or int(targets.max().item()) >= vocab_size:
        raise ValueError("target class is outside the vocabulary")
    target_logits = logits.gather(-1, targets.long().unsqueeze(-1)).squeeze(-1)
    losses = torch.logsumexp(logits, dim=-1) - target_logits
    return losses.mean()


class AdamW(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self._validate_hparams(lr, betas, eps, weight_decay)
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        for group in self.param_groups:
            self._validate_group(group)

    @staticmethod
    def _validate_hparams(lr, betas, eps, weight_decay) -> None:
        if not isinstance(lr, (int, float)) or lr < 0:
            raise ValueError("lr must be nonnegative")
        if not isinstance(eps, (int, float)) or eps < 0:
            raise ValueError("eps must be nonnegative")
        if not isinstance(weight_decay, (int, float)) or weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative")
        if not isinstance(betas, (tuple, list)) or len(betas) != 2:
            raise ValueError("betas must contain exactly two values")
        beta1, beta2 = betas
        if not isinstance(beta1, (int, float)) or not 0 <= beta1 < 1:
            raise ValueError("beta1 must satisfy 0 <= beta1 < 1")
        if not isinstance(beta2, (int, float)) or not 0 <= beta2 < 1:
            raise ValueError("beta2 must satisfy 0 <= beta2 < 1")

    @classmethod
    def _validate_group(cls, group) -> None:
        cls._validate_hparams(group["lr"], group["betas"], group["eps"], group["weight_decay"])

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        with torch.no_grad():
            for group in self.param_groups:
                self._validate_group(group)
                lr = float(group["lr"])
                beta1, beta2 = group["betas"]
                eps = float(group["eps"])
                weight_decay = float(group["weight_decay"])
                for parameter in group["params"]:
                    gradient = parameter.grad
                    if gradient is None:
                        continue
                    if gradient.is_sparse:
                        raise RuntimeError("AdamW does not support sparse gradients")
                    state = self.state[parameter]
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(parameter)
                        state["exp_avg_sq"] = torch.zeros_like(parameter)
                    state["step"] += 1
                    step = state["step"]
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
                    bias_correction1 = 1.0 - beta1**step
                    bias_correction2 = 1.0 - beta2**step
                    parameter.mul_(1.0 - lr * weight_decay)
                    denom = exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
                    denom.add_(eps)
                    parameter.addcdiv_(exp_avg, denom, value=-(lr / bias_correction1))
        return loss


def get_lr_cosine_schedule(
    step: int,
    learning_rate_max: float,
    learning_rate_min: float,
    warmup_steps: int,
    cosine_steps: int,
) -> float:
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("step must be a nonnegative integer")
    if not isinstance(warmup_steps, int) or isinstance(warmup_steps, bool) or warmup_steps < 0:
        raise ValueError("warmup_steps must be a nonnegative integer")
    if not isinstance(cosine_steps, int) or isinstance(cosine_steps, bool):
        raise ValueError("cosine_steps must be an integer")
    if warmup_steps >= cosine_steps:
        raise ValueError("schedule requires warmup_steps < cosine_steps")
    if learning_rate_min < 0 or learning_rate_max < 0 or learning_rate_min > learning_rate_max:
        raise ValueError("learning rates must satisfy 0 <= min <= max")
    if warmup_steps > 0 and step < warmup_steps:
        return float((step / warmup_steps) * learning_rate_max)
    if step <= cosine_steps:
        progress = (step - warmup_steps) / (cosine_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(learning_rate_min + cosine * (learning_rate_max - learning_rate_min))
    return float(learning_rate_min)


def gradient_clipping(parameters: Iterable[nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> float:
    if not isinstance(max_l2_norm, (int, float)) or max_l2_norm <= 0:
        raise ValueError("max_l2_norm must be positive")
    gradients: list[Tensor] = []
    squared_sum = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        if parameter.grad.is_sparse:
            raise RuntimeError("gradient clipping requires dense gradients")
        gradient = parameter.grad
        gradients.append(gradient)
        squared_sum += float(gradient.detach().double().square().sum().item())
    norm = math.sqrt(squared_sum)
    if norm > max_l2_norm:
        scale = float(max_l2_norm) / (norm + eps)
        for gradient in gradients:
            gradient.mul_(scale)
    return float(norm)


def load_token_array(path: str | os.PathLike) -> np.memmap:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise ValueError(f"not a file: {path}")
    size = path.stat().st_size
    if size % 2 != 0:
        raise ValueError("token stream has an odd byte length")
    if size == 0:
        raise ValueError("token stream is empty")
    return np.memmap(path, mode="r", dtype=np.dtype("<u2"))


def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    sequence_length: int,
    device: str | torch.device,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not isinstance(sequence_length, int) or isinstance(sequence_length, bool) or sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if getattr(dataset, "ndim", None) != 1:
        raise ValueError("dataset must be one-dimensional")
    if len(dataset) < sequence_length + 1:
        raise ValueError("dataset does not contain one complete next-token window")
    high = len(dataset) - sequence_length
    starts = torch.randint(0, high, (batch_size,), generator=generator)
    windows = np.stack(
        [np.asarray(dataset[start : start + sequence_length + 1], dtype=np.int64) for start in starts.tolist()],
        axis=0,
    )
    batch = torch.from_numpy(windows).to(device=device, dtype=torch.long)
    return batch[:, :-1], batch[:, 1:]


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    next_step: int,
    train_generator: torch.Generator,
    val_generator: torch.Generator,
    out: str | os.PathLike | BinaryIO,
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "next_step": int(next_step),
        "train_generator": train_generator.get_state(),
        "val_generator": val_generator.get_state(),
    }
    torch.save(payload, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO,
    model: nn.Module,
    optimizer: Optimizer,
    train_generator: torch.Generator,
    val_generator: torch.Generator,
) -> int:
    checkpoint = torch.load(src, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    train_generator.set_state(checkpoint["train_generator"])
    val_generator.set_state(checkpoint["val_generator"])
    return int(checkpoint["next_step"])


def evaluate_validation(
    model: nn.Module,
    validation_tokens: np.ndarray,
    *,
    batch_size: int,
    sequence_length: int,
    num_batches: int,
    device: str | torch.device,
    generator: torch.Generator,
) -> float:
    if num_batches <= 0:
        raise ValueError("num_batches must be positive")
    was_training = model.training
    model.eval()
    total = 0.0
    try:
        with torch.inference_mode():
            for _ in range(num_batches):
                x, y = get_batch(validation_tokens, batch_size, sequence_length, device, generator)
                total += float(cross_entropy(model(x), y).item())
    finally:
        model.train(was_training)
    return total / num_batches


def generate(
    model: nn.Module,
    prompt_ids: Tensor,
    max_new_tokens: int,
    context_length: int,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eot_token_id: int | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    if prompt_ids.ndim != 1 or prompt_ids.numel() == 0 or prompt_ids.dtype != torch.long:
        raise ValueError("prompt_ids must be a non-empty one-dimensional torch.long tensor")
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a nonnegative integer")
    if not isinstance(context_length, int) or isinstance(context_length, bool) or context_length <= 0:
        raise ValueError("context_length must be positive")
    if not hasattr(model, "context_length") or context_length != model.context_length:
        raise ValueError("context_length must equal model.context_length")
    if not isinstance(temperature, (int, float)) or temperature <= 0:
        raise ValueError("temperature must be positive")
    if not isinstance(top_p, (int, float)) or not 0 < top_p <= 1:
        raise ValueError("top_p must satisfy 0 < top_p <= 1")
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = prompt_ids.device
    if prompt_ids.device != model_device:
        raise ValueError("prompt_ids must be on the model device")
    if max_new_tokens == 0:
        return prompt_ids.clone()

    was_training = model.training
    model.eval()
    sequence = prompt_ids.clone()
    try:
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                cropped = sequence[-context_length:]
                logits = model(cropped.unsqueeze(0))[0, -1]
                probabilities = softmax(logits / float(temperature), dim=-1)
                sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
                cumulative = sorted_probabilities.cumsum(dim=-1)
                keep = torch.ones_like(sorted_probabilities, dtype=torch.bool)
                if keep.numel() > 1:
                    keep[1:] = cumulative[:-1] < float(top_p)
                filtered = sorted_probabilities * keep
                filtered = filtered / filtered.sum()
                sampled_rank = torch.multinomial(filtered, 1, generator=generator)
                next_token = sorted_indices[sampled_rank]
                sequence = torch.cat((sequence, next_token.to(sequence.device)))
                if eot_token_id is not None and int(next_token.item()) == int(eot_token_id):
                    break
    finally:
        model.train(was_training)
    return sequence


@dataclass
class TrainingConfig:
    batch_size: int = 16
    sequence_length: int = 256
    gradient_accumulation_steps: int = 16
    num_steps: int = 10_000
    learning_rate_max: float = 3e-4
    learning_rate_min: float = 3e-5
    warmup_steps: int = 200
    cosine_steps: int = 9_999
    betas: tuple[float, float] = (0.9, 0.95)
    adam_eps: float = 1e-8
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    eval_interval: int = 200
    log_interval: int = 20
    checkpoint_interval: int = 500
    num_validation_batches: int = 20
    train_seed: int = 12345
    val_seed: int = 54321


def train_model(
    model: nn.Module,
    train_tokens: np.ndarray,
    validation_tokens: np.ndarray,
    *,
    device: str | torch.device,
    config: TrainingConfig,
    checkpoint_path: str | os.PathLike | None = None,
    resume: bool = False,
    metrics_path: str | os.PathLike | None = None,
) -> tuple[Optimizer, list[dict[str, float | int | None]]]:
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate_max,
        betas=config.betas,
        eps=config.adam_eps,
        weight_decay=config.weight_decay,
    )
    train_generator = torch.Generator().manual_seed(config.train_seed)
    val_generator = torch.Generator().manual_seed(config.val_seed)
    next_step = 0
    if resume:
        if checkpoint_path is None or not Path(checkpoint_path).exists():
            raise FileNotFoundError("resume=True requires an existing checkpoint_path")
        next_step = load_checkpoint(checkpoint_path, model, optimizer, train_generator, val_generator)

    history: list[dict[str, float | int | None]] = []
    metrics_file = Path(metrics_path) if metrics_path is not None else None
    fieldnames = ["completed_steps", "train_loss", "validation_loss", "learning_rate", "grad_norm"]

    for step in range(next_step, config.num_steps):
        model.train()
        lr = get_lr_cosine_schedule(
            step,
            config.learning_rate_max,
            config.learning_rate_min,
            config.warmup_steps,
            config.cosine_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad()
        train_loss = 0.0
        for _ in range(config.gradient_accumulation_steps):
            x, y = get_batch(
                train_tokens,
                config.batch_size,
                config.sequence_length,
                device,
                train_generator,
            )
            microbatch_loss = cross_entropy(model(x), y)
            (microbatch_loss / config.gradient_accumulation_steps).backward()
            train_loss += float(microbatch_loss.detach().item())
        train_loss /= config.gradient_accumulation_steps
        grad_norm = gradient_clipping(model.parameters(), config.max_grad_norm)
        optimizer.step()

        completed_steps = step + 1
        final_step = completed_steps == config.num_steps
        should_validate = final_step or completed_steps % config.eval_interval == 0
        should_log = should_validate or completed_steps % config.log_interval == 0
        should_checkpoint = final_step or completed_steps % config.checkpoint_interval == 0
        validation_loss = None
        if should_validate:
            validation_loss = evaluate_validation(
                model,
                validation_tokens,
                batch_size=config.batch_size,
                sequence_length=config.sequence_length,
                num_batches=config.num_validation_batches,
                device=device,
                generator=val_generator,
            )
        if should_log:
            row = {
                "completed_steps": completed_steps,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": lr,
                "grad_norm": grad_norm,
            }
            history.append(row)
            print(
                f"step={completed_steps:5d} train={train_loss:.4f} "
                f"val={validation_loss if validation_loss is not None else float('nan'):.4f} "
                f"lr={lr:.3e} grad_norm={grad_norm:.3f}"
            )
            if metrics_file is not None:
                metrics_file.parent.mkdir(parents=True, exist_ok=True)
                file_exists = metrics_file.exists() and metrics_file.stat().st_size > 0
                with metrics_file.open("a", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(row)
        if should_checkpoint and checkpoint_path is not None:
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                model,
                optimizer,
                completed_steps,
                train_generator,
                val_generator,
                checkpoint_path,
            )
    return optimizer, history


def export_final_model(model: nn.Module, path: str | os.PathLike = "final_model.pt") -> dict[str, Tensor]:
    state = {
        name: tensor.detach().cpu().to(torch.float16) if tensor.is_floating_point() else tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
    }
    torch.save(state, path)
    return state
