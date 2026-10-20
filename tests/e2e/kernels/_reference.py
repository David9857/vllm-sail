# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Golden references and tolerance table for the category-3 PPU kernel suite.

Everything in this module is plain PyTorch: no ``vllm_sail`` import, no PPU
extension and no ``torch.ops.vllm`` lookup appears here, so the helpers stay
valid oracles for the kernels under test.  All references accumulate in
``float32`` and return ``float32`` unless documented otherwise; callers decide
how the comparison is rounded.

The five numeric test modules (``test_deepgemm_numeric``,
``test_acext_moe_numeric``, ``test_pla_numeric``, ``test_flash_attn_numeric``
and ``test_flash_mla_numeric``) import their tolerances from :data:`TOL` so
that per-dtype and per-chip bands are declared exactly once.
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# Tolerance table
# ---------------------------------------------------------------------------
#
# ``(atol, rtol)`` pairs keyed by quantisation/precision family.  The values
# follow PART 4 of the CI plan: bf16/fp16 golden paths use 2e-2, the w8a8 int8
# paths use 3e-2 (one extra bit of head-room for the per-token rounding) and
# the fp8/mxfp4 paths use 8e-2.  ``fp8_relaxed`` mirrors the band already used
# by ``tests/e2e/tier_b`` for E4M3FN comparisons, and ``*_decode`` widens the
# linear-attention band because a recurrent kernel accumulates error over the
# whole sequence instead of over a single reduction.
#
# ``attention_fp8`` is wider than both.  An E4M3FN GEMM only rounds its two
# operands, but an E4M3FN FlashAttention kernel additionally rounds the softmax
# probabilities ``P`` to E4M3FN before ``P @ V``, and no fp32 golden can
# reproduce that without re-implementing the tiling.  The resulting floor was
# measured on ZW-890P (8, 9) with FA3 at s512/h16/kv8/d128, s256/h16/kv1/d128
# and s128/h16/kv8/d128: worst absolute deviation 0.0681 (4 of 1048576 elements
# above the 0.03125 ``fp8_relaxed`` floor, every one at ``|expected| < 0.05``
# where the signed sum cancels), while the very same wrapper on bf16 operands
# deviates by only 0.0094.  0.075 keeps ~10% head-room over that measurement
# without letting a structural error through.
TOL: dict[str, tuple[float, float]] = {
    "fp32": (1e-4, 1e-4),
    "fp16": (1e-2, 1e-2),
    "bf16": (2e-2, 2e-2),
    "int8": (3e-2, 3e-2),
    "w4a8": (3e-2, 3e-2),
    "fp8": (8e-2, 8e-2),
    "fp8_relaxed": (0.03125, 0.15),
    "attention_fp8": (0.075, 0.15),
    "mxfp4": (8e-2, 8e-2),
    "attention": (2e-2, 2e-2),
    "linear_attn": (2e-2, 2e-2),
    "linear_attn_decode": (5e-2, 5e-2),
    "mla": (2e-2, 2e-2),
    "mla_fp8_cache": (0.063, 0.15),
    "packed_mla_cache": (2e-2, 2e-2),
}

# Per-chip widenings.  Both PPU chips run the bf16/int8 paths, so only the
# paths that are physically chip-specific need an override: the 810E packed
# E4M3FN MLA cache reorders the KV accumulation relative to the 890P path and
# therefore needs roughly twice the bf16 band.
#
# ``("bf16", "cap80")`` covers the ACEXT fused MoE on ZW-810E.  Its expert
# GEMMs accumulate the ``K`` reduction in bf16 rather than fp32, which leaves a
# reproducible noise floor on top of the ordinary bf16 rounding band.  Measured
# on swu113 over the six ``MOE_CASES`` of ``test_acext_moe_numeric.py`` the
# worst deviation was 3.268x the shared ``TOL["bf16"]`` band, so 3.5 keeps ~7%
# head-room over that measurement while still rejecting a structural error (a
# wrong expert, a wrong reduction axis or a dropped routing weight moves the
# output by tens of band-widths, not three).  The shared ``TOL["bf16"]`` band
# is deliberately left untouched: it is keyed without a chip dimension and
# widening it would silently relax every bf16 path on ZW-890P too, which
# accumulates in fp32 and passes at factor 1.0.
CHIP_TOLERANCE_FACTOR: dict[tuple[str, str], float] = {
    ("bf16", "cap80"): 3.5,
    ("mla", "cap80"): 1.0,
    ("mla_fp8_cache", "cap80"): 2.0,
    ("mxfp4", "cap89"): 1.0,
    ("fp8", "cap89"): 1.0,
}


def tolerance(kind: str, chip: str | None = None) -> tuple[float, float]:
    """Return ``(atol, rtol)`` for ``kind``, optionally widened for ``chip``.

    ``chip`` is the marker name (``"cap80"`` / ``"cap89"``) or ``None`` when
    the case runs on both PPU chips.
    """
    if kind not in TOL:
        raise KeyError(
            f"unknown tolerance kind {kind!r}; expected one of {sorted(TOL)}"
        )
    atol, rtol = TOL[kind]
    factor = CHIP_TOLERANCE_FACTOR.get((kind, chip or ""), 1.0)
    return atol * factor, rtol * factor


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    kind: str = "bf16",
    chip: str | None = None,
    *,
    atol: float | None = None,
    rtol: float | None = None,
    msg: str = "",
) -> None:
    """Compare ``actual`` against the fp32 ``expected`` using the shared band."""
    default_atol, default_rtol = tolerance(kind, chip)
    torch.testing.assert_close(
        actual.detach().to(torch.float32).cpu(),
        expected.detach().to(torch.float32).cpu(),
        atol=default_atol if atol is None else atol,
        rtol=default_rtol if rtol is None else rtol,
        msg=msg or None,
    )


def assert_ulp(actual: torch.Tensor, expected: torch.Tensor, max_ulp: int = 1) -> None:
    """Allow only rounding-boundary differences, including values near zero.

    Mirrors ``_assert_ulp`` in ``tests/e2e/test_deepseek_v4_native.py`` so the
    fp8/mxfp4 cases can be checked in storage units instead of relying on a
    relative band that is meaningless for subnormals.
    """
    if actual.dtype != expected.dtype:
        raise AssertionError(f"dtype mismatch: {actual.dtype} vs {expected.dtype}")
    storage = torch.int8 if actual.element_size() == 1 else torch.int16
    sign = 1 << (actual.element_size() * 8 - 1)
    a = actual.detach().cpu().contiguous().view(storage).to(torch.int32)
    b = expected.detach().cpu().contiguous().view(storage).to(torch.int32)
    a = torch.where(a < 0, -sign - a, a)
    b = torch.where(b < 0, -sign - b, b)
    distance = int((a - b).abs().max().item())
    if distance > max_ulp:
        raise AssertionError(f"maximum storage ULP distance: {distance} > {max_ulp}")


# ---------------------------------------------------------------------------
# Quantisation helpers (pure PyTorch)
# ---------------------------------------------------------------------------

FP8_E4M3_MAX = 448.0
INT8_MAX = 127.0
MXFP4_BLOCK = 32
MXFP4_MAX = 6.0
E8M0_BIAS = 127
# E2M1 magnitude table indexed by the low three bits of a nibble.
E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def quant_per_tensor_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-tensor fp8; returns ``(q, dequant_scale[1])``."""
    dtype = torch.float8_e4m3fn
    amax = x.detach().to(torch.float32).abs().amax().clamp(min=1e-4)
    scale = (amax / FP8_E4M3_MAX).reshape(1)
    q = (x.to(torch.float32) / scale).to(dtype)
    return q, scale


def quant_per_token_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-token fp8; returns ``(q[M, K], scale[M, 1])``."""
    dtype = torch.float8_e4m3fn
    xf = x.detach().to(torch.float32)
    amax = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
    scale = amax / FP8_E4M3_MAX
    return (xf / scale).to(dtype), scale


def quant_block_fp8(
    x: torch.Tensor, block: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-wise fp8; returns ``(q[M, K], scale[M // block, K // block])``."""
    dtype = torch.float8_e4m3fn
    xf = x.detach().to(torch.float32)
    m, k = xf.shape
    if m % block or k % block:
        raise ValueError(f"block fp8 needs dims divisible by {block}, got {(m, k)}")
    view = xf.view(m // block, block, k // block, block)
    amax = view.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-4)
    scale = amax / FP8_E4M3_MAX
    q = (view / scale).to(dtype).view(m, k)
    return q, scale.view(m // block, k // block)


def quant_per_tensor_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-tensor int8; returns ``(q, dequant_scale[1])``."""
    xf = x.detach().to(torch.float32)
    scale = (xf.abs().amax().clamp(min=1e-4) / INT8_MAX).reshape(1)
    q = torch.round(xf / scale).clamp(-INT8_MAX - 1, INT8_MAX).to(torch.int8)
    return q, scale


def quant_per_token_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-token int8; returns ``(q[M, K], scale[M, 1])``.

    Matches ``per_token_group_quant_int8(..., use_rounding=True)`` which is the
    activation quantiser the PPU ScaledMM kernels are fed with.
    """
    xf = x.detach().to(torch.float32)
    scale = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4) / INT8_MAX
    q = torch.round(xf / scale).clamp(-INT8_MAX - 1, INT8_MAX).to(torch.int8)
    return q, scale


def quant_per_channel_int8(
    x: torch.Tensor, dim: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel int8 weight quantisation.

    ``x`` is ``[N, K]`` and ``dim=0`` selects the output channel, giving the
    ``[N, 1]`` scale layout compressed-tensors stores on disk.

    This helper is 2-D only: it reduces over *every* axis except ``dim``, so
    handing it a ``[E, N, K]`` expert stack with ``dim=1`` would collapse the
    expert axis as well and return a scale that no longer identifies a channel.
    Use :func:`quant_per_channel_int8_3d` for the ``[E, N, K]`` MoE layouts.
    """
    assert x.ndim == 2, (
        f"quant_per_channel_int8 is the 2-D [N, K] weight quantiser, got "
        f"ndim={x.ndim}; use quant_per_channel_int8_3d for [E, N, K] stacks"
    )
    xf = x.detach().to(torch.float32)
    reduce_dims = tuple(d for d in range(xf.ndim) if d != dim)
    amax = xf.abs().amax(dim=reduce_dims, keepdim=True).clamp(min=1e-4)
    scale = amax / INT8_MAX
    q = torch.round(xf / scale).clamp(-INT8_MAX - 1, INT8_MAX).to(torch.int8)
    return q, scale


def quant_per_channel_int8_3d(
    x: torch.Tensor, dim: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel int8 weight quantisation for an ``[E, N, K]`` stack.

    ``dim`` selects the output-channel axis (``1`` for the ``[E, N, K]`` layout
    the ACEXT fused MoE kernels take) and the reduction runs over ``K`` *only*,
    so the expert axis survives and the returned scale is ``[E, N, 1]`` - one
    independent dequant scale per expert output channel, which is what
    ``int8_w8a8_moe_quant_config`` and the ``[E, N, 1]`` on-disk layout expect.

    Keeping this separate from :func:`quant_per_channel_int8` preserves that
    helper's established 2-D contract for the deepgemm and perf-lane cases
    instead of overloading one function with two different reduction rules.
    """
    if x.ndim != 3:
        raise ValueError(f"expected an [E, N, K] stack, got shape {tuple(x.shape)}")
    reduce_dims = tuple(d for d in range(3) if d != 0 and d != dim)
    xf = x.detach().to(torch.float32)
    amax = xf.abs().amax(dim=reduce_dims, keepdim=True).clamp(min=1e-4)
    scale = amax / INT8_MAX
    q = torch.round(xf / scale).clamp(-INT8_MAX - 1, INT8_MAX).to(torch.int8)
    return q, scale


def expand_blocks(
    scale: torch.Tensor,
    block: int | tuple[int, int],
    rows: int,
    cols: int,
) -> torch.Tensor:
    """Expand a blocked scale grid to a dense ``[rows, cols]`` scale grid.

    ``block`` is ``(row_block, col_block)``; a bare ``int`` means a square
    block.  A per-token activation scale cast with ``block_size=[1, 128]`` is
    already ``[rows, cols // 128]``, i.e. ``row_block == 1``: repeating it
    along dim 0 would hand every row somebody else's scale.  Both axes are
    sliced back to the unpadded extent because ``per_block_cast_to_fp8`` rounds
    the scale grid up to a whole block.
    """
    row_block, col_block = (block, block) if isinstance(block, int) else block
    return (
        scale.repeat_interleave(row_block, dim=0)[:rows]
        .repeat_interleave(col_block, dim=1)[:, :cols]
        .to(torch.float32)
    )


def dequant_int8(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.to(torch.float32) * scale.to(torch.float32)


def dequant_fp8(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.to(torch.float32) * scale.to(torch.float32)


# --- MXFP4 -----------------------------------------------------------------


def decode_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """Unpack ``uint8[..., L]`` into ``float32[..., 2 * L]``.

    The production encoder stores the even element in the low nibble
    (``evens | (odds << 4)``), so the low nibble is expanded first.
    """
    table = torch.tensor(E2M1_MAGNITUDES, dtype=torch.float32, device=packed.device)
    codes = torch.stack([packed & 0x0F, (packed >> 4) & 0x0F], dim=-1).to(torch.int64)
    magnitudes = table[codes & 0x07]
    signs = torch.where((codes & 0x08) != 0, -1.0, 1.0).to(magnitudes.dtype)
    return (magnitudes * signs).reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def decode_e8m0(code: torch.Tensor) -> torch.Tensor:
    """Turn E8M0 exponent bytes into ``float32`` power-of-two dequant scales."""
    exponent = code.to(torch.int32) - E8M0_BIAS
    return torch.pow(2.0, exponent.to(torch.float32))


def decode_e8m0_uint16_col_major(scale: torch.Tensor) -> torch.Tensor:
    """Unpack ``uint16[..., S // 2]`` into per-block scales ``[..., S]``.

    ``downcast_to_mxfp4`` with ``ScaleFormat.UINT16_COL_MAJOR`` stores
    ``lo_e8m0 | (hi_e8m0 << 8)``, i.e. block ``2 * p`` in the low byte and
    block ``2 * p + 1`` in the high byte.
    """
    raw = scale.to(torch.int32)
    lo = (raw & 0xFF).to(torch.uint8)
    hi = ((raw >> 8) & 0xFF).to(torch.uint8)
    pairs = torch.stack([lo, hi], dim=-1).reshape(*scale.shape[:-1], -1)
    return decode_e8m0(pairs)


def dequantize_mxfp4(
    packed: torch.Tensor, scale: torch.Tensor, *, uint16_scale: bool = True
) -> torch.Tensor:
    """Reconstruct the fp32 values encoded by ``downcast_to_mxfp4``.

    ``packed`` is ``uint8[..., K // 2]`` and ``scale`` covers ``K // 32``
    blocks along the same (last) axis.
    """
    values = decode_e2m1(packed)
    if uint16_scale:
        scales = decode_e8m0_uint16_col_major(scale)
    else:
        scales = decode_e8m0(scale.to(torch.uint8))
    k = values.shape[-1]
    if scales.shape[-1] * MXFP4_BLOCK != k:
        raise ValueError(
            f"mxfp4 scale covers {scales.shape[-1] * MXFP4_BLOCK} elements, got {k}"
        )
    return values * scales.repeat_interleave(MXFP4_BLOCK, dim=-1)


def pack_int4_along_last(x: torch.Tensor) -> torch.Tensor:
    """Pack ``int[..., L]`` values in ``[-8, 7]`` into ``int8[..., L // 2]``.

    Element ``2 * i`` lands in the low nibble and ``2 * i + 1`` in the high
    nibble, matching the ACEXT w4a8 weight convention.
    """
    values = x.to(torch.int32)
    if bool((values < -8).any()) or bool((values > 7).any()):
        raise ValueError("int4 values must live in [-8, 7]")
    low = values[..., 0::2] & 0x0F
    high = values[..., 1::2] & 0x0F
    return (low | (high << 4)).to(torch.int8)


def unpack_int4_along_last(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`pack_int4_along_last`, returning signed ``int32``."""
    raw = packed.to(torch.int32)
    low = raw & 0x0F
    high = (raw >> 4) & 0x0F
    low = torch.where(low > 7, low - 16, low)
    high = torch.where(high > 7, high - 16, high)
    return torch.stack([low, high], dim=-1).reshape(*packed.shape[:-1], -1)


# ---------------------------------------------------------------------------
# GEMM references
# ---------------------------------------------------------------------------


def ref_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """``a @ b.T`` in fp32 with ``a`` ``[M, K]`` and ``b`` ``[N, K]``."""
    out = a.to(torch.float32) @ b.to(torch.float32).t()
    if bias is not None:
        out = out + bias.to(torch.float32)
    return out


def ref_grouped_gemm(
    a: torch.Tensor,
    w: torch.Tensor,
    expert_ids: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Row-contiguous grouped GEMM: row ``m`` uses expert ``expert_ids[m]``.

    ``a`` is ``[M, K]``, ``w`` is ``[E, N, K]`` and the result is ``[M, N]``.
    """
    m = a.shape[0]
    out = a.new_zeros((m, w.shape[1]), dtype=torch.float32)
    ids = expert_ids.to(torch.int64).reshape(-1)
    if ids.numel() != m:
        raise ValueError(f"expert_ids has {ids.numel()} entries for M={m}")
    for expert in range(w.shape[0]):
        rows = ids == expert
        if not bool(rows.any()):
            continue
        selected = a[rows].to(torch.float32) @ w[expert].to(torch.float32).t()
        if bias is not None:
            selected = selected + bias[expert].to(torch.float32)
        out[rows] = selected
    return out


# ---------------------------------------------------------------------------
# MoE reference
# ---------------------------------------------------------------------------


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """vLLM's ``SiluAndMul``: split the last dim in half, ``silu(a) * b``."""
    half = x.shape[-1] // 2
    gate, up = x[..., :half], x[..., half:]
    return torch.nn.functional.silu(gate) * up


def ref_moe(
    hidden: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dense per-expert reference for a SiLU-gated fused MoE layer.

    ``w1`` is ``[E, 2 * I, H]`` (gate rows first, up rows second) and ``w2`` is
    ``[E, H, I]``.  Optional scales are per-output-channel ``[E, N, 1]``
    tensors applied to the int8/int4 weights before the fp32 matmul.
    """
    tokens, hidden_size = hidden.shape
    num_experts, two_i, _ = w1.shape
    out = hidden.new_zeros((tokens, hidden_size), dtype=torch.float32)
    x = hidden.to(torch.float32)

    w1f = w1.to(torch.float32)
    w2f = w2.to(torch.float32)
    if w1_scale is not None:
        w1f = w1f * w1_scale.to(torch.float32).reshape(num_experts, two_i, 1)
    if w2_scale is not None:
        w2f = w2f * w2_scale.to(torch.float32).reshape(num_experts, hidden_size, 1)

    top_k = topk_ids.shape[1]
    for slot in range(top_k):
        experts = topk_ids[:, slot].to(torch.int64)
        weights = topk_weights[:, slot].to(torch.float32).unsqueeze(-1)
        for expert in range(num_experts):
            rows = experts == expert
            if not bool(rows.any()):
                continue
            gate_up = x[rows] @ w1f[expert].t()
            if w1_bias is not None:
                gate_up = gate_up + w1_bias[expert].to(torch.float32)
            activated = silu_and_mul(gate_up)
            down = activated @ w2f[expert].t()
            if w2_bias is not None:
                down = down + w2_bias[expert].to(torch.float32)
            out[rows] += weights[rows] * down
    return out


def ref_moe_from_packed_w4(
    hidden: torch.Tensor,
    w1_packed: torch.Tensor,
    w2_packed: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> torch.Tensor:
    """Reference for the ACEXT w4a8 path (N-major, K-packed int4 weights).

    ``w1_packed`` is the raw buffer declared by
    ``W4AInt8MoEMethod.create_weights``: ``[E, 2I, H // 2]``, where the two
    int4 weights sharing a byte are adjacent along the *input* (K) axis.
    ``w2_packed`` is ``[E, H, I // 2]``.  Per-output-channel scales are
    ``[E, N, 1]`` with ``N == 2 * I`` and ``N == H``.

    ``process_weights_after_loading`` only relabels those buffers (``view`` to
    ``[E, H, I]`` / ``[E, I, H // 2]``) and never moves data, so the golden is
    built in the physical order the weight loader produced.
    """
    w1 = unpack_int4_along_last(w1_packed)  # [E, 2I, H]
    w2 = unpack_int4_along_last(w2_packed)  # [E, H, I]
    return ref_moe(
        hidden,
        w1,
        w2,
        topk_weights,
        topk_ids,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )


# ---------------------------------------------------------------------------
# Attention references
# ---------------------------------------------------------------------------


def _causal_mask(
    query_len: int,
    key_len: int,
    device: torch.device,
    *,
    row_start: int = 0,
    row_count: int | None = None,
    head_axis: bool = False,
) -> torch.Tensor:
    """Bottom-right aligned causal mask (FlashAttention convention).

    ``query_len``/``key_len`` are the lengths of the *whole* attention problem:
    the bottom-right alignment ``offset = key_len - query_len`` is a property of
    that problem, not of the block being masked, so a caller walking the query
    axis in chunks must still pass the totals.  ``row_start``/``row_count``
    select the query rows of the current chunk and keep them at their absolute
    positions, which is what makes the blocked golden equal the single-pass one
    row for row.

    ``head_axis`` inserts a singleton head dimension for references whose score
    layout is ``[B, S, H, K]`` (MLA) instead of ``[B, H, S, K]``.
    """
    if row_count is None:
        row_count = max(query_len - row_start, 0)
    offset = key_len - query_len
    rows = (
        torch.arange(row_start, row_start + row_count, device=device)
        .unsqueeze(1)
        .to(torch.int64)
    )
    cols = torch.arange(key_len, device=device).unsqueeze(0).to(torch.int64)
    mask = cols > rows + offset
    return mask.unsqueeze(1) if head_axis else mask


def _chunk_step(query_len: int, query_chunk: int | None) -> int:
    """Resolve the query-blocking stride, clamped to ``[1, query_len]``."""
    if query_chunk is None or query_chunk <= 0 or query_chunk >= query_len:
        return max(query_len, 1)
    return query_chunk


def ref_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    scale: float | None = None,
    query_chunk: int | None = None,
) -> torch.Tensor:
    """Batched ``softmax(Q K^T / sqrt(d)) V`` in fp32.

    ``q`` is ``[B, H, Sq, D]`` and ``k``/``v`` are ``[B, Hk, Sk, D]``; grouped
    query attention is expanded by repeating each KV head ``H // Hk`` times.

    ``query_chunk`` blocks the score matrix along the query axis.  A dense
    ``H=64, Sq=Sk=4096`` case would otherwise materialise a ``4.3 GiB`` fp32
    score tensor; blocking keeps the golden runnable on a single PPU device
    while leaving the arithmetic bit-identical (each row of the softmax is
    independent).
    """
    batch, heads, query_len, head_dim = q.shape
    kv_heads = k.shape[1]
    key_len = k.shape[2]
    if heads % kv_heads:
        raise ValueError(f"H={heads} is not a multiple of Hk={kv_heads}")
    repeat = heads // kv_heads
    if scale is None:
        scale = head_dim**-0.5

    kf = k.to(torch.float32).repeat_interleave(repeat, dim=1)
    vf = v.to(torch.float32).repeat_interleave(repeat, dim=1)
    kt = kf.transpose(-1, -2)

    step = _chunk_step(query_len, query_chunk)
    chunks = []
    for start in range(0, query_len, step):
        end = min(start + step, query_len)
        scores = (q[:, :, start:end].to(torch.float32) @ kt) * scale
        if causal:
            mask = _causal_mask(
                query_len,
                key_len,
                scores.device,
                row_start=start,
                row_count=end - start,
            )
            scores = scores.masked_fill(mask, float("-inf"))
        chunks.append(torch.softmax(scores, dim=-1) @ vf)
    if len(chunks) == 1:
        return chunks[0]
    return torch.cat(chunks, dim=2)


def ref_attention_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    scale: float | None = None,
    query_chunk: int | None = None,
) -> torch.Tensor:
    """``logsumexp`` companion of :func:`ref_attention`, laid out ``[B, H, Sq]``.

    FlashAttention reports the natural logarithm of the softmax denominator,
    which is what the PPU FA2/FA3 wrappers return as ``softmax_lse`` (there in
    ``[H, total_q]`` varlen layout).
    """
    batch, heads, query_len, head_dim = q.shape
    kv_heads = k.shape[1]
    key_len = k.shape[2]
    if heads % kv_heads:
        raise ValueError(f"H={heads} is not a multiple of Hk={kv_heads}")
    repeat = heads // kv_heads
    if scale is None:
        scale = head_dim**-0.5

    kt = k.to(torch.float32).repeat_interleave(repeat, dim=1).transpose(-1, -2)
    step = _chunk_step(query_len, query_chunk)
    chunks = []
    for start in range(0, query_len, step):
        end = min(start + step, query_len)
        scores = (q[:, :, start:end].to(torch.float32) @ kt) * scale
        if causal:
            mask = _causal_mask(
                query_len,
                key_len,
                scores.device,
                row_start=start,
                row_count=end - start,
            )
            scores = scores.masked_fill(mask, float("-inf"))
        chunks.append(torch.logsumexp(scores, dim=-1))
    return chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=2)


def ref_varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    causal: bool = False,
    scale: float | None = None,
    query_chunk: int | None = None,
) -> torch.Tensor:
    """Flattened varlen reference: ``q`` ``[total_q, H, D]`` -> same layout."""
    head_dim = q.shape[-1]
    if scale is None:
        scale = head_dim**-0.5
    cu_q = cu_seqlens_q.detach().cpu().to(torch.int64).tolist()
    cu_k = cu_seqlens_k.detach().cpu().to(torch.int64).tolist()
    if len(cu_q) != len(cu_k):
        raise ValueError("cu_seqlens_q and cu_seqlens_k must have equal length")
    chunks = []
    for index in range(len(cu_q) - 1):
        q_slice = q[cu_q[index] : cu_q[index + 1]].transpose(0, 1).unsqueeze(0)
        k_slice = k[cu_k[index] : cu_k[index + 1]].transpose(0, 1).unsqueeze(0)
        v_slice = v[cu_k[index] : cu_k[index + 1]].transpose(0, 1).unsqueeze(0)
        out = ref_attention(
            q_slice,
            k_slice,
            v_slice,
            causal=causal,
            scale=scale,
            query_chunk=query_chunk,
        )
        chunks.append(out.squeeze(0).transpose(0, 1))
    if not chunks:
        return q.new_zeros(q.shape, dtype=torch.float32)
    return torch.cat(chunks, dim=0)


# ---------------------------------------------------------------------------
# MLA reference
# ---------------------------------------------------------------------------


def ref_mla(
    q: torch.Tensor,
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    *,
    causal: bool = True,
    scale: float | None = None,
    query_chunk: int | None = None,
) -> torch.Tensor:
    """Absorbed-weight MLA decode/prefill reference.

    ``q`` is ``[B, Sq, H, 576]`` (``512`` latent + ``64`` RoPE), ``kv_c`` is
    ``[B, Sk, 512]`` and ``k_pe`` is ``[B, Sk, 1, 64]``.  The output is the
    ``[B, Sq, H, 512]`` latent projection, matching ``head_dim_v=512``.
    """
    _, _, heads, head_dim = q.shape
    latent = kv_c.shape[-1]
    rope = head_dim - latent
    if rope <= 0:
        raise ValueError(f"q head_dim {head_dim} must exceed kv_lora_rank {latent}")
    if scale is None:
        scale = head_dim**-0.5

    kv = kv_c.to(torch.float32)
    pe = k_pe.to(torch.float32).squeeze(2)
    query_len = q.shape[1]
    key_len = kv.shape[1]

    step = _chunk_step(query_len, query_chunk)
    chunks = []
    for start in range(0, query_len, step):
        end = min(start + step, query_len)
        block = q[:, start:end]
        scores = torch.einsum(
            "bshd,bkd->bshk", block[..., :latent].to(torch.float32), kv
        )
        scores = scores + torch.einsum(
            "bshd,bkd->bshk", block[..., latent:].to(torch.float32), pe
        )
        scores = scores * scale
        if causal:
            # ``scores`` is ``[B, chunk, H, Sk]``, so the mask needs a
            # singleton head axis to broadcast over ``H``.
            mask = _causal_mask(
                query_len,
                key_len,
                scores.device,
                row_start=start,
                row_count=end - start,
                head_axis=True,
            )
            scores = scores.masked_fill(mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        chunks.append(torch.einsum("bshk,bkd->bshd", probs, kv))
    if len(chunks) == 1:
        return chunks[0]
    return torch.cat(chunks, dim=1)


def ref_mla_lse(
    q: torch.Tensor,
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    *,
    causal: bool = True,
    scale: float | None = None,
    query_chunk: int | None = None,
) -> torch.Tensor:
    """``logsumexp`` companion of :func:`ref_mla`, laid out ``[B, H, Sq]``."""
    _, _, heads, head_dim = q.shape
    latent = kv_c.shape[-1]
    if scale is None:
        scale = head_dim**-0.5
    kv = kv_c.to(torch.float32)
    pe = k_pe.to(torch.float32).squeeze(2)
    query_len = q.shape[1]
    key_len = kv.shape[1]

    step = _chunk_step(query_len, query_chunk)
    chunks = []
    for start in range(0, query_len, step):
        end = min(start + step, query_len)
        block = q[:, start:end]
        scores = torch.einsum(
            "bshd,bkd->bshk", block[..., :latent].to(torch.float32), kv
        )
        scores = scores + torch.einsum(
            "bshd,bkd->bshk", block[..., latent:].to(torch.float32), pe
        )
        scores = scores * scale
        if causal:
            mask = _causal_mask(
                query_len,
                key_len,
                scores.device,
                row_start=start,
                row_count=end - start,
                head_axis=True,
            )
            scores = scores.masked_fill(mask, float("-inf"))
        chunks.append(torch.logsumexp(scores, dim=-1))
    merged = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=1)
    return merged.permute(0, 2, 1).contiguous()


def _paged_mla_tokens(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    latent: int,
    index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather request ``index`` into dense ``(kv_c, k_pe)`` fp32 operands."""
    block_size = int(kv_cache.shape[1])
    flat = kv_cache.to(torch.float32).reshape(-1, kv_cache.shape[-1])
    length = int(seq_lens[index].item())
    if length <= 0:
        raise ValueError(f"request {index} has non-positive seq_len {length}")
    positions = torch.arange(length, device=flat.device)
    physical = block_table[index].to(torch.int64)[positions // block_size]
    rows = physical * block_size + positions % block_size
    tokens = flat[rows]
    return tokens[:, :latent].unsqueeze(0), (
        tokens[:, latent:].unsqueeze(0).unsqueeze(2)
    )


def ref_mla_paged(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    latent: int = 512,
    causal: bool = True,
    scale: float | None = None,
    query_chunk: int | None = None,
) -> torch.Tensor:
    """Per-request MLA golden over a paged packed cache.

    ``q`` is ``[B, Sq, H, latent + rope]`` and ``kv_cache`` is either
    ``[num_blocks, block_size, D]`` or the FlashMLA ``[..., 1, D]`` view; the
    result is ``[B, Sq, H, latent]`` in fp32.  Requests are decoded one at a
    time so each keeps its own key length and bottom-right causal alignment.
    """
    batch, query_len, heads, _ = q.shape
    out = q.new_zeros((batch, query_len, heads, latent), dtype=torch.float32)
    for index in range(batch):
        kv_c, k_pe = _paged_mla_tokens(kv_cache, block_table, seq_lens, latent, index)
        out[index] = ref_mla(
            q[index : index + 1],
            kv_c,
            k_pe,
            causal=causal,
            scale=scale,
            query_chunk=query_chunk,
        )[0]
    return out


def ref_mla_paged_lse(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    latent: int = 512,
    causal: bool = True,
    scale: float | None = None,
    query_chunk: int | None = None,
) -> torch.Tensor:
    """``logsumexp`` companion of :func:`ref_mla_paged`, laid out ``[B, H, Sq]``."""
    batch, query_len, heads, _ = q.shape
    out = q.new_zeros((batch, heads, query_len), dtype=torch.float32)
    for index in range(batch):
        kv_c, k_pe = _paged_mla_tokens(kv_cache, block_table, seq_lens, latent, index)
        out[index] = ref_mla_lse(
            q[index : index + 1],
            kv_c,
            k_pe,
            causal=causal,
            scale=scale,
            query_chunk=query_chunk,
        )[0]
    return out


# ---------------------------------------------------------------------------
# Linear attention (GDN / KDA) reference
# ---------------------------------------------------------------------------


def _softplus(x: torch.Tensor, beta: float, threshold: float) -> torch.Tensor:
    """Triton's ``where(beta * x <= threshold, log1p(exp(beta * x)) / beta, x)``."""
    scaled = beta * x
    safe = torch.clamp(scaled, max=threshold)
    return torch.where(scaled <= threshold, torch.log1p(torch.exp(safe)) / beta, x)


def _delta_rule_step(
    h: torch.Tensor,
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    gate: torch.Tensor,
    beta_t: torch.Tensor,
    scale: float,
    *,
    per_channel_gate: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One timestep of the gated delta rule; ``h`` is ``[Hv, V, K]``.

    Direct transcription of ``fused_sigmoid_gating_delta_rule_update_kernel``:
    decay the state, subtract the current prediction, scale by beta, apply the
    rank-one update and read out with the scaled query.
    """
    q_t = q_t * scale
    if per_channel_gate:
        h = h * gate.unsqueeze(1)
    else:
        h = h * gate.reshape(h.shape[0], 1, 1)
    delta = (h * k_t.unsqueeze(1)).sum(-1)
    v_t = (v_t - delta) * beta_t.unsqueeze(-1)
    h = h + v_t.unsqueeze(-1) * k_t.unsqueeze(1)
    return h, (h * q_t.unsqueeze(1)).sum(-1)


def ref_linear_attn(
    a_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    beta: float = 1.0,
    threshold: float = 20.0,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    state_indices: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    use_qk_l2norm: bool = False,
    is_kda: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Golden recurrence for the fused sigmoid-gating delta rule.

    Inputs mirror ``fused_sigmoid_gating_delta_rule_update`` and the upstream
    Triton kernel's pointer arithmetic:

    * ``q``/``k`` are ``[B, T, H, K]``, ``v`` is ``[B, T, Hv, V]``;
    * ``a`` is ``[B * T, Hv]`` (GDN) or ``[B * T, Hv, K]`` (KDA) and ``b`` is
      ``[B * T, Hv]`` -- both indexed by the *global* token position;
    * ``a_log`` is ``[Hv]`` for both variants, ``dt_bias`` is ``[Hv]`` (GDN) or
      ``[Hv, K]`` (KDA);
    * ``initial_state`` is the ``[num_slots, Hv, V, K]`` fp32 pool and
      ``state_indices`` maps a sequence to its slot.  Slot ``<= 0`` is vLLM's
      reserved ``NULL_BLOCK_ID``: the kernel returns early, so neither the
      output nor the state is touched for that sequence.

    Returns ``(o, final_state)`` with ``o`` shaped like ``v`` (fp32) and
    ``final_state`` shaped like ``initial_state``.
    """
    batch, seq_len, heads, head_k = q.shape
    head_v = v.shape[2]
    head_v_dim = v.shape[3]
    if head_v % heads:
        raise ValueError(f"Hv={head_v} is not a multiple of H={heads}")
    repeat = head_v // heads
    if scale is None:
        scale = head_k**-0.5

    varlen = cu_seqlens is not None
    if varlen:
        if batch != 1:
            raise ValueError(f"cu_seqlens requires a flattened batch, got B={batch}")
        boundaries = cu_seqlens.detach().cpu().to(torch.int64).tolist()
    else:
        boundaries = [seq * seq_len for seq in range(batch + 1)]
    num_seqs = len(boundaries) - 1

    state = (
        initial_state.to(torch.float32).clone() if initial_state is not None else None
    )
    slots = (
        state_indices.detach().cpu().to(torch.int64).tolist()
        if state_indices is not None
        else list(range(num_seqs))
    )
    if len(slots) != num_seqs:
        raise ValueError(
            f"state_indices has {len(slots)} entries for {num_seqs} sequences"
        )

    out = q.new_zeros((batch, seq_len, head_v, head_v_dim), dtype=torch.float32)
    a_log_f = (
        a_log.to(torch.float32).reshape(-1, 1) if is_kda else a_log.to(torch.float32)
    )
    dt_bias_f = dt_bias.to(torch.float32)

    for seq in range(num_seqs):
        start, end = boundaries[seq], boundaries[seq + 1]
        if end <= start:
            continue
        slot = slots[seq]
        if state is not None and slot <= 0:
            continue  # NULL_BLOCK_ID: the kernel leaves output and state alone
        h = (
            state[slot].clone()
            if state is not None
            else q.new_zeros((head_v, head_v_dim, head_k), dtype=torch.float32)
        )
        batch_row = 0 if varlen else seq
        for step in range(start, end):
            tok_row = step if varlen else step - start
            q_t = q[batch_row, tok_row].to(torch.float32)
            k_t = k[batch_row, tok_row].to(torch.float32)
            v_t = v[batch_row, tok_row].to(torch.float32)
            b_t = b[step].to(torch.float32)
            a_t = a[step].to(torch.float32)

            if use_qk_l2norm:
                q_t = q_t * torch.rsqrt((q_t * q_t).sum(-1, keepdim=True) + 1e-6)
                k_t = k_t * torch.rsqrt((k_t * k_t).sum(-1, keepdim=True) + 1e-6)
            q_t = q_t.repeat_interleave(repeat, dim=0)
            k_t = k_t.repeat_interleave(repeat, dim=0)

            gate = torch.exp(
                -torch.exp(a_log_f) * _softplus(a_t + dt_bias_f, beta, threshold)
            )
            h, o_t = _delta_rule_step(
                h,
                q_t,
                k_t,
                v_t,
                gate,
                torch.sigmoid(b_t),
                scale,
                per_channel_gate=is_kda,
            )
            out[batch_row, tok_row] = o_t

        if state is not None:
            state[slot] = h

    return out, state


def ref_chunk_linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    use_qk_l2norm: bool = False,
    is_kda: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Golden for the GDN/KDA *prefill* chunk kernels (``forward_pla``).

    Unlike :func:`ref_linear_attn` the gate is supplied directly in log space:
    ``g`` is ``[1, T, Hv]`` (GDN) or ``[1, T, Hv, K]`` (KDA) and ``beta`` is
    ``[1, T, Hv]`` already passed through the activation the caller uses.
    ``use_qk_l2norm`` mirrors ``use_qk_l2norm_in_kernel``: the PLA prefill
    backend asserts it is ``False`` and expects pre-normalised q/k, while the
    Triton/FLA path normalises inside the kernel.  ``initial_state`` is the
    vLLM ``[N, Hv, V, K]`` fp32 pool (``forward_pla`` transposes it to the
    ``[N, Hv, K, V]`` layout FlashQLA wants and transposes the result back),
    so a zero pool keeps the golden layout-agnostic.
    """
    _, total_tokens, heads, head_k = q.shape
    head_v = v.shape[2]
    head_v_dim = v.shape[3]
    repeat = head_v // heads
    if scale is None:
        scale = head_k**-0.5

    if cu_seqlens is not None:
        boundaries = cu_seqlens.detach().cpu().to(torch.int64).tolist()
    else:
        boundaries = [0, total_tokens]

    state = (
        initial_state.to(torch.float32).clone() if initial_state is not None else None
    )
    out = q.new_zeros((1, total_tokens, head_v, head_v_dim), dtype=torch.float32)

    for seq in range(len(boundaries) - 1):
        start, end = boundaries[seq], boundaries[seq + 1]
        if end <= start:
            continue
        h = (
            state[seq].clone()
            if state is not None
            else q.new_zeros((head_v, head_v_dim, head_k), dtype=torch.float32)
        )
        for step in range(start, end):
            q_t = q[0, step].to(torch.float32)
            k_t = k[0, step].to(torch.float32)
            if use_qk_l2norm:
                q_t = q_t * torch.rsqrt((q_t * q_t).sum(-1, keepdim=True) + 1e-6)
                k_t = k_t * torch.rsqrt((k_t * k_t).sum(-1, keepdim=True) + 1e-6)
            q_t = q_t.repeat_interleave(repeat, dim=0)
            k_t = k_t.repeat_interleave(repeat, dim=0)
            v_t = v[0, step].to(torch.float32)
            gate = torch.exp(g[0, step].to(torch.float32))
            h, o_t = _delta_rule_step(
                h,
                q_t,
                k_t,
                v_t,
                gate,
                beta[0, step].to(torch.float32),
                scale,
                per_channel_gate=is_kda,
            )
            out[0, step] = o_t
        if state is not None:
            state[seq] = h

    return out, state


# ---------------------------------------------------------------------------
# Packed MLA cache (810E software E4M3FN) helpers
# ---------------------------------------------------------------------------
#
# ``vllm_sail/models/deepseek_v4/ops/cache.py`` ships an SM80 gather kernel
# that decodes the packed DeepSeek-V4 K cache entirely in software, because
# Triton refuses to compile ``tl.float8e4nv`` on capability (8, 0).
# ``register_ops()`` exposes it as ``torch.ops.vllm.ppu_deepseek_v4_dequant_``
# ``gather`` and ``patch/enhancement/models/deepseek_v4_cache.py`` routes the
# upstream ``dequantize_and_gather_k_cache`` entry point to it exactly when
# ``is_ppu() and is_device_capability((8, 0))``.
#
# The helpers below reproduce the byte layout documented by upstream
# ``vllm/models/deepseek_v4/common/ops/cache_utils.py`` so a test can build a
# cache without running the compressor-dependent PPU encoder, and can decode
# the *same bytes* in fp32 as the golden.

PACKED_MLA_FP8_DIM = 448
PACKED_MLA_ROPE_DIM = 64
PACKED_MLA_HEAD_DIM = PACKED_MLA_FP8_DIM + PACKED_MLA_ROPE_DIM
PACKED_MLA_SCALE_DIM = 8
PACKED_MLA_QUANT_BLOCK = 64
PACKED_MLA_N_QUANT_BLOCKS = PACKED_MLA_FP8_DIM // PACKED_MLA_QUANT_BLOCK
PACKED_MLA_TOKEN_BYTES = PACKED_MLA_FP8_DIM + PACKED_MLA_ROPE_DIM * 2
# Per-block layout: ``block_size`` token rows of 576 bytes, then ``block_size``
# scale rows of 8 bytes.
PACKED_MLA_BLOCK_BYTES = PACKED_MLA_TOKEN_BYTES + PACKED_MLA_SCALE_DIM


def packed_mla_block_stride(block_size: int, *, pad: bool = True) -> int:
    """Bytes per physical cache block.

    Upstream pads each block to a multiple of the 576-byte token row; the
    kernel only ever reads ``k_cache.stride(0)`` so both variants are valid.
    """
    raw = block_size * PACKED_MLA_BLOCK_BYTES
    if not pad:
        return raw
    unit = PACKED_MLA_TOKEN_BYTES
    return ((raw + unit - 1) // unit) * unit


def encode_packed_mla_tokens(k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """UE8M0 + E4M3FN encode ``k`` ``[T, 512]``.

    Returns ``(data, scales)`` where ``data`` is ``[T, 576]`` uint8 (``448``
    E4M3FN bytes followed by ``128`` bf16 RoPE bytes) and ``scales`` is
    ``[T, 8]`` uint8 (``7`` biased exponents plus the documented padding byte).
    The exponent rule is upstream's ``ceil(log2(max(amax, 1e-4) / 448))``
    biased by ``127`` and clamped to ``[0, 255]``.
    """
    if k.dim() != 2 or k.shape[1] != PACKED_MLA_HEAD_DIM:
        raise ValueError(f"expected [T, {PACKED_MLA_HEAD_DIM}], got {tuple(k.shape)}")
    tokens = k.shape[0]
    x = k.to(torch.bfloat16).to(torch.float32).cpu()

    nope = x[:, :PACKED_MLA_FP8_DIM].reshape(
        tokens, PACKED_MLA_N_QUANT_BLOCKS, PACKED_MLA_QUANT_BLOCK
    )
    amax = nope.abs().amax(dim=-1).clamp_min(1e-4)
    exponent = torch.ceil(torch.log2(amax / FP8_E4M3_MAX))
    inv_scale = torch.exp2(-exponent).unsqueeze(-1)
    scaled = torch.clamp(nope * inv_scale, -FP8_E4M3_MAX, FP8_E4M3_MAX)
    fp8_bytes = (
        scaled.to(torch.float8_e4m3fn)
        .view(torch.uint8)
        .reshape(tokens, PACKED_MLA_FP8_DIM)
    )
    encoded = (exponent + float(E8M0_BIAS)).clamp(0.0, 255.0).to(torch.uint8)
    padding = encoded.new_zeros(
        (tokens, PACKED_MLA_SCALE_DIM - PACKED_MLA_N_QUANT_BLOCKS)
    )
    scales = torch.cat([encoded, padding], dim=1)
    rope = (
        x[:, PACKED_MLA_FP8_DIM:]
        .contiguous()
        .to(torch.bfloat16)
        .view(torch.uint8)
        .reshape(tokens, PACKED_MLA_ROPE_DIM * 2)
    )
    return torch.cat([fp8_bytes, rope], dim=1), scales


def build_packed_mla_cache(
    k: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_blocks: int,
    block_size: int,
    *,
    pad: bool = True,
) -> torch.Tensor:
    """Scatter ``k`` ``[T, 512]`` into a uint8 ``[num_blocks, stride]`` cache.

    ``slot_mapping`` holds absolute ``block * block_size + offset`` slots and
    must be non-negative; the returned tensor lives on ``k.device`` and is
    contiguous so ``stride(0)`` equals :func:`packed_mla_block_stride`.
    """
    stride = packed_mla_block_stride(block_size, pad=pad)
    data, scales = encode_packed_mla_tokens(k)
    slots = slot_mapping.detach().cpu().to(torch.int64)
    if bool((slots < 0).any()):
        raise ValueError("slot_mapping must be non-negative; padding is unsupported")

    cache = torch.zeros((num_blocks * stride,), dtype=torch.uint8)
    block_of = slots // block_size
    pos_of = slots % block_size
    token_base = block_of * stride + pos_of * PACKED_MLA_TOKEN_BYTES
    rows = token_base.unsqueeze(1) + torch.arange(PACKED_MLA_TOKEN_BYTES)
    cache.index_copy_(0, rows.reshape(-1), data.reshape(-1))
    scale_base = (
        block_of * stride
        + block_size * PACKED_MLA_TOKEN_BYTES
        + pos_of * PACKED_MLA_SCALE_DIM
    )
    scale_rows = scale_base.unsqueeze(1) + torch.arange(PACKED_MLA_SCALE_DIM)
    cache.index_copy_(0, scale_rows.reshape(-1), scales.reshape(-1))
    return cache.view(num_blocks, stride).to(k.device)


def ref_packed_mla_gather(
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
    max_rows: int,
) -> torch.Tensor:
    """fp32 golden for ``ppu_deepseek_v4_dequant_gather``.

    Mirrors the SM80 kernel exactly: ``gather_len`` defaults to ``seq_len``,
    the window starts at ``seq_len - gather_len``, each gathered token lands in
    output row ``offset + i`` and untouched rows stay zero.  Returns
    ``[num_reqs, max_rows, 512]`` fp32 with the ``448`` dequantised latent
    entries followed by the ``64`` bf16 RoPE entries passed through verbatim.
    """
    stride = int(k_cache.stride(0))
    cache = k_cache.detach().cpu().contiguous().view(torch.uint8).reshape(-1)
    lens = seq_lens.detach().cpu().to(torch.int64).tolist()
    glens = (
        lens
        if gather_lens is None
        else gather_lens.detach().cpu().to(torch.int64).tolist()
    )
    table = block_table.detach().cpu().to(torch.int64)
    out = torch.zeros((len(lens), max_rows, PACKED_MLA_HEAD_DIM), dtype=torch.float32)
    for req, seq_len in enumerate(lens):
        gather_len = glens[req]
        for i in range(gather_len):
            pos = seq_len - gather_len + i
            physical = int(table[req, pos // block_size].item())
            pos_in_block = pos % block_size
            block_base = physical * stride
            token_base = block_base + pos_in_block * PACKED_MLA_TOKEN_BYTES

            fp8_bytes = cache[token_base : token_base + PACKED_MLA_FP8_DIM]
            values = fp8_bytes.view(torch.float8_e4m3fn).to(torch.float32)
            scale_base = (
                block_base
                + block_size * PACKED_MLA_TOKEN_BYTES
                + pos_in_block * PACKED_MLA_SCALE_DIM
            )
            encoded = cache[scale_base : scale_base + PACKED_MLA_N_QUANT_BLOCKS]
            scale = torch.exp2(encoded.to(torch.float32) - float(E8M0_BIAS))
            latent = (
                values.reshape(PACKED_MLA_N_QUANT_BLOCKS, PACKED_MLA_QUANT_BLOCK)
                * scale.unsqueeze(-1)
            ).reshape(-1)

            rope_base = token_base + PACKED_MLA_FP8_DIM
            rope = (
                cache[rope_base : rope_base + PACKED_MLA_ROPE_DIM * 2]
                .view(torch.bfloat16)
                .to(torch.float32)
            )
            out[req, offset + i] = torch.cat([latent, rope])
    return out
