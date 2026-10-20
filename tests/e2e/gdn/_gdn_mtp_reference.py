# SPDX-License-Identifier: Apache-2.0
"""Independent eager PyTorch recurrence, without vLLM, Triton or native ops.

The state stays FP32 within a request. Candidate states and pre-normalization
outputs are cast at the storage boundaries used by the unfused GDN path.
"""

import torch
import torch.nn.functional as F


@torch.inference_mode()
def gdn_mtp_reference(
    *,
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    state: torch.Tensor,
    output_gate: torch.Tensor,
    norm_weight: torch.Tensor,
    scale: float,
    norm_eps: float,
    output_gate_activation: str,
) -> torch.Tensor:
    """Return gated output and update each positive destination state slot."""
    _, value_heads, value_dim, key_dim = state.shape
    key_heads = (mixed_qkv.shape[1] - value_heads * value_dim) // (2 * key_dim)
    q, k, v = mixed_qkv.float().split(
        [key_heads * key_dim, key_heads * key_dim, value_heads * value_dim], dim=-1
    )
    q = q.reshape(-1, key_heads, key_dim)
    k = k.reshape(-1, key_heads, key_dim)
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * scale
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    q = q.repeat_interleave(value_heads // key_heads, dim=1)
    k = k.repeat_interleave(value_heads // key_heads, dim=1)
    v = v.reshape(-1, value_heads, value_dim)
    decay = torch.exp(-A_log.float().exp() * F.softplus(a.float() + dt_bias.float()))
    beta = b.float().sigmoid()
    if output_gate_activation == "silu":
        gate = F.silu(output_gate.float())
    elif output_gate_activation == "sigmoid":
        gate = output_gate.float().sigmoid()
    else:
        raise ValueError(f"Unknown output gate: {output_gate_activation}")

    output = torch.zeros_like(output_gate)
    offsets = cu_seqlens.tolist()
    slots = state_indices.tolist()
    for request, accepted in enumerate(num_accepted_tokens.tolist()):
        start, end = offsets[request : request + 2]
        if start == end or not 1 <= accepted <= len(slots[request]):
            continue
        source = slots[request][accepted - 1]
        if source <= 0:
            continue
        h = state[source].float().clone()
        for step, token in enumerate(range(start, end)):
            h *= decay[token, :, None, None]
            projection = (h * k[token, :, None, :]).sum(-1)
            delta = (v[token] - projection) * beta[token, :, None]
            h += delta[:, :, None] * k[token, :, None, :]
            raw = (h * q[token, :, None, :]).sum(-1).to(mixed_qkv.dtype).float()
            norm = raw * torch.rsqrt(raw.square().mean(-1, keepdim=True) + norm_eps)
            output[token] = (norm * norm_weight.float() * gate[token]).to(output.dtype)
            destination = slots[request][step]
            if destination > 0:
                state[destination].copy_(h)
    return output
