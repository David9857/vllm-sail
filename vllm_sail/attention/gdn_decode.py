# SPDX-License-Identifier: Apache-2.0
"""GDN decode policy; optional device libraries are queried only by callers."""

from __future__ import annotations

from dataclasses import dataclass

from vllm_sail import envs


@dataclass(frozen=True)
class GDNDecodeBackend:
    requested: str
    active: str
    reason: str | None = None


def resolve_gdn_decode_backend(
    requested: str,
    *,
    use_pla: bool,
    pla_reason: str | None,
    cuda_reason: str | None,
) -> GDNDecodeBackend:
    """Resolve static capabilities, respecting an explicit backend choice.

    Auto retains the existing PLA preference when the legacy switch permits
    it, then tries the fused native implementation, then Triton. Per-batch
    constraints remain the responsibility of the selected execution path.
    """
    if requested not in envs.GDN_DECODE_KERNELS:
        raise ValueError(f"Invalid GDN decode kernel: {requested}")
    if requested == "auto":
        if use_pla and pla_reason is None:
            return GDNDecodeBackend(requested, "pla")
        if cuda_reason is None:
            return GDNDecodeBackend(requested, "cuda", pla_reason)
        reason = "; ".join(
            (pla_reason or "PLA disabled by VLLM_SAIL_USE_PLA", cuda_reason)
        )
        return GDNDecodeBackend(requested, "triton", reason)
    reason = {"pla": pla_reason, "cuda": cuda_reason, "triton": None}[requested]
    if reason is not None:
        raise ValueError(
            f"VLLM_GDN_DECODE_KERNEL={requested} is not supported: {reason}"
        )
    return GDNDecodeBackend(requested, requested)


def pla_decode_enabled() -> bool:
    """An explicit decode choice overrides the legacy PLA switch for GDN only."""
    requested = envs.VLLM_GDN_DECODE_KERNEL
    return requested == "pla" or (requested == "auto" and envs.VLLM_SAIL_USE_PLA)


def has_fused_gdn_kernel() -> bool:
    """A CUDA dispatch registration is required; a schema or stub is insufficient.

    PPU uses torch's CUDA dispatch key. Unsupported SAIL stubs register only a
    CompositeExplicitAutograd implementation, so they cannot pass this check.
    """
    import torch

    try:
        return torch._C._dispatch_has_kernel_for_dispatch_key(
            "_C::fused_gdn_decode_post_conv_mtp", "CUDA"
        )
    except RuntimeError:
        return False
