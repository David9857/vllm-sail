# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numeric E2E oracles for the PPU FlashAttention kernels (CI plan PART 4).

PPU does not build vLLM's ``_vllm_fa2_C`` / ``_vllm_fa3_C`` extensions.  It
ships its own ``flash_attn`` (FA2) and ``flash_attn_3._C`` (FA3) wheels and
rebinds ``vllm.vllm_flash_attn`` to ``vllm_sail.attention.flash_attn`` through
``flash_attn_shim.install()``.  This module exercises that implementation on a
real device; ``tests/ut/test_flash_attn_shim.py`` only covers the import layer.

Both PPU chips run FlashAttention, so the numeric cases carry ``ppu``.  The
only chip-specific behaviour is FP8: ``fa_utils.flash_attn_supports_kv_cache_``
``dtype`` returns ``fa_version == 3 and current_platform.supports_fp8()`` on
PPU, which is false on 810E and true on 890P, so the FP8 descale case is marked
``cap89``.

Every numeric case compares against the fp32 golden in
``tests/e2e/kernels/_reference.py`` (``ref.ref_varlen_attention`` /
``ref.ref_attention_lse``) with the shared ``attention`` band (2e-2).  The
``native_kernels`` fixture asserts both wheels and both ``torch.ops``
namespaces are present, so a missing operator fails the run instead of
skipping it.
"""

from __future__ import annotations

import importlib
import sys

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("requires a real PPU/CUDA device", allow_module_level=True)
pytest.importorskip("vllm")

import vllm_sail  # noqa: E402

vllm_sail.register_out_of_tree()

from vllm.platforms import current_platform  # noqa: E402

if not current_platform.is_ppu():
    pytest.skip("requires a PPU platform (is_ppu() is False)", allow_module_level=True)

from tests.e2e.kernels import _reference as ref  # noqa: E402
from vllm_sail.attention.flash_attn import _kernels as fa  # noqa: E402
from vllm_sail.attention.flash_attn.flash_attn_interface import (  # noqa: E402
    sparse_attn_func,
    sparse_attn_varlen_func,
)

pytestmark = pytest.mark.ppu

DEVICE = torch.device("cuda")

_NO_FA2 = (
    "the PPU `flash_attn` (FA2) wheel is mandatory for this suite: install the "
    "FlashAttention wheel shipped with the PPU SDK and check its shared-library "
    "dependencies"
)
_NO_FA3 = (
    "the PPU `flash_attn_3._C` (FA3) wheel is mandatory for this suite: install "
    "the FlashAttention wheel shipped with the PPU SDK and check its "
    "shared-library dependencies"
)

# PART 4 asks for both dispatch branches to be measured, so every numeric case
# is run twice, once per wheel.
FA_VERSIONS = [
    pytest.param(2, id="fa2"),
    pytest.param(3, id="fa3"),
]

# PART 4 cross product: causal x seq_len x num_heads x head_dim x GQA kv_heads
# (2 x 3 x 3 x 2 x 2 = 72).  The full product spends almost all of its time in
# the fp32 golden rather than in the kernel, so the sweep below keeps every
# level of every factor present while pairing them into 11 cases.
ATTN_CASES = [
    pytest.param(False, 128, 16, 8, 64, id="full-s128-h16-kv8-d64"),
    pytest.param(True, 128, 16, 1, 128, id="causal-s128-h16-mqa-d128"),
    pytest.param(False, 128, 32, 8, 128, id="full-s128-h32-kv8-d128"),
    pytest.param(True, 128, 64, 1, 64, id="causal-s128-h64-mqa-d64"),
    pytest.param(False, 1024, 16, 1, 128, id="full-s1024-h16-mqa-d128"),
    pytest.param(True, 1024, 32, 8, 64, id="causal-s1024-h32-kv8-d64"),
    pytest.param(False, 1024, 64, 8, 128, id="full-s1024-h64-kv8-d128"),
    pytest.param(True, 1024, 64, 1, 64, id="causal-s1024-h64-mqa-d64"),
    pytest.param(False, 4096, 32, 1, 64, id="full-s4096-h32-mqa-d64"),
    pytest.param(True, 4096, 16, 8, 128, id="causal-s4096-h16-kv8-d128"),
    pytest.param(True, 4096, 64, 8, 64, id="causal-s4096-h64-kv8-d64"),
]

# Cheaper subset reused by the LSE, parity and out-buffer cases.
SMALL_CASES = [
    pytest.param(False, 128, 16, 8, 64, id="full-s128-h16-kv8-d64"),
    pytest.param(True, 128, 32, 1, 128, id="causal-s128-h32-mqa-d128"),
    pytest.param(True, 1024, 32, 8, 64, id="causal-s1024-h32-kv8-d64"),
    pytest.param(False, 1024, 64, 8, 128, id="full-s1024-h64-kv8-d128"),
]

# Ragged self-attention batch: mixed lengths exercise the cu_seqlens walk and
# the bottom-right causal alignment on sequences that are not block aligned.
RAGGED_SEQ_LENS = (128, 17, 512, 1)

# The fp32 golden materialises ``[B, H, chunk, Sk]``; 64 rows keeps a
# ``H=64, Sk=4096`` case at ~67 MiB instead of the 4.3 GiB dense score matrix.
QUERY_CHUNK = 64


@pytest.fixture(scope="module", autouse=True)
def native_kernels():
    """Load the native extension strictly; a missing wheel must fail."""
    from vllm_sail.attention.flash_attn_shim import install as install_fa_shim
    from vllm_sail.native import install
    from vllm_sail.native.extensions import import_kernels

    import_kernels(strict=True)
    install()
    install_fa_shim()

    assert fa.FA2_AVAILABLE, f"{_NO_FA2}: {fa.FA2_UNAVAILABLE_REASON}"
    assert fa.FA3_AVAILABLE, f"{_NO_FA3}: {fa.FA3_UNAVAILABLE_REASON}"
    assert fa.is_fa_version_supported(2), fa.fa_version_unsupported_reason(2)
    assert fa.is_fa_version_supported(3), fa.fa_version_unsupported_reason(3)
    # The wheels register their ops under flash_attn / flash_attn_3, never under
    # vLLM's own _vllm_fa2_C / _vllm_fa3_C namespaces.
    assert (
        getattr(torch.ops.flash_attn, "_flash_attn_varlen_forward", None) is not None
    ), _NO_FA2
    assert getattr(torch.ops.flash_attn_3, "fwd", None) is not None, _NO_FA3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generator(seed: int = 20240517) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(seed)


def _randn(*shape: int, seed: int, dtype=torch.bfloat16) -> torch.Tensor:
    return torch.randn(*shape, generator=_generator(seed), device=DEVICE, dtype=dtype)


def _cumsum(lengths, dtype=torch.int32) -> torch.Tensor:
    """``[0, l0, l0 + l1, ...]`` on device, in the dtype FA expects."""
    total = [0]
    for length in lengths:
        total.append(total[-1] + int(length))
    return torch.tensor(total, dtype=dtype, device=DEVICE)


def _self_attention_inputs(seq_len, num_heads, kv_heads, head_dim, seed):
    """Single-sequence varlen operands plus matching ``cu_seqlens``."""
    q = _randn(seq_len, num_heads, head_dim, seed=seed)
    k = _randn(seq_len, kv_heads, head_dim, seed=seed + 1)
    v = _randn(seq_len, kv_heads, head_dim, seed=seed + 2)
    cu = _cumsum([seq_len])
    return q, k, v, cu


def _run_varlen(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    *,
    causal,
    fa_version,
    softmax_scale=None,
    **kwargs,
):
    """Call the PPU wrapper with the max lengths derived from ``cu_seqlens``."""
    max_q = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item())
    max_k = int((cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item())
    return fa.flash_attn_varlen_func(
        q,
        k,
        v,
        max_q,
        cu_seqlens_q,
        max_k,
        cu_seqlens_k=cu_seqlens_k,
        softmax_scale=softmax_scale,
        causal=causal,
        fa_version=fa_version,
        **kwargs,
    )


def _golden_varlen(q, k, v, cu_seqlens, *, causal, softmax_scale=None):
    """fp32 golden in the wrapper's flattened ``[total, H, D]`` layout."""
    return ref.ref_varlen_attention(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        causal=causal,
        scale=softmax_scale,
        query_chunk=QUERY_CHUNK,
    )


def _gather_paged(cache, block_table, seq_len, block_size, index):
    """Densify ``[length, kv_heads, D]`` for request ``index``."""
    positions = torch.arange(seq_len, device=cache.device)
    physical = block_table[index].to(torch.int64)[positions // block_size]
    rows = physical * block_size + positions % block_size
    return cache.reshape(-1, *cache.shape[2:])[rows]


# ---------------------------------------------------------------------------
# Availability / dispatch contracts (no heavy compute)
# ---------------------------------------------------------------------------


def test_fa_version_probe_matches_wheel_availability():
    """The public probe reports exactly what the private wheels support."""
    assert fa.is_fa_version_supported(2) == fa.FA2_AVAILABLE
    assert fa.is_fa_version_supported(3) == fa.FA3_AVAILABLE
    assert fa.fa_version_unsupported_reason(2) is None
    assert fa.fa_version_unsupported_reason(3) is None
    # FA4 needs NVIDIA CuteDSL; PPU exposes the compile hook as None.
    assert fa.compile_flash_attn_varlen_func_from_specs is None

    interface = importlib.import_module(
        "vllm_sail.attention.flash_attn.flash_attn_interface"
    )
    assert interface.is_fa_version_supported(4) is False
    assert "FA4" in interface.fa_version_unsupported_reason(4)
    with pytest.raises(AssertionError, match="Unsupported FA version"):
        fa._require_fa_version(4)


def test_shim_rebinds_the_upstream_import_path():
    """``vllm.vllm_flash_attn`` must resolve to the PPU package, not upstream."""
    package = importlib.import_module("vllm.vllm_flash_attn")
    interface = importlib.import_module("vllm.vllm_flash_attn.flash_attn_interface")
    assert package is importlib.import_module("vllm_sail.attention.flash_attn")
    assert interface is importlib.import_module(
        "vllm_sail.attention.flash_attn.flash_attn_interface"
    )
    assert sys.modules["vllm"].vllm_flash_attn is package
    assert package.flash_attn_varlen_func is interface.flash_attn_varlen_func


def test_ppu_prefers_fa3_when_both_wheels_are_present():
    """``fa_utils.get_flash_attn_version`` is patched to prefer FA3 on PPU."""
    from vllm.v1.attention.backends import fa_utils

    assert fa_utils.get_flash_attn_version() == 3
    assert fa_utils.get_flash_attn_version(requires_alibi=True) == 2


def test_fp8_kv_cache_support_follows_the_chip():
    """FP8 KV needs FA3 *and* hardware FP8, i.e. 890P only."""
    from vllm.v1.attention.backends import fa_utils

    assert fa_utils.flash_attn_supports_kv_cache_dtype("fp8_e5m2") is False
    expected = bool(current_platform.supports_fp8())
    assert fa_utils.flash_attn_supports_kv_cache_dtype("fp8_e4m3") is expected


def test_sparse_attention_entry_points_are_rejected():
    """PPU FA ships no vertical/slash sparse kernel; the stubs must raise.

    ``raise RuntimeError`` is the first statement of both bodies, so the sparse
    metadata is never inspected and the contract holds for any operands.
    """
    q, k, v, _ = _self_attention_inputs(64, 8, 8, 64, seed=7)
    with pytest.raises(RuntimeError, match="do not support sparse_attn"):
        sparse_attn_func(q, k, v, None, None, None, None)
    cu = _cumsum([64])
    with pytest.raises(RuntimeError, match="do not support sparse_attn"):
        sparse_attn_varlen_func(q, k, v, None, None, None, None, cu, cu, 64, 64)


def test_fa2_rejects_fa3_only_arguments():
    """FA2 must fail loudly instead of silently dropping FA3-only features."""
    q, k, v, cu = _self_attention_inputs(64, 8, 8, 64, seed=11)
    common = dict(cu_seqlens_k=cu, causal=False, fa_version=2)

    with pytest.raises(NotImplementedError, match="num_splits"):
        fa.flash_attn_varlen_func(q, k, v, 64, cu, 64, num_splits=2, **common)

    aux = torch.empty((1, 8, 64), dtype=torch.float32, device=DEVICE)
    with pytest.raises(NotImplementedError, match="s_aux"):
        fa.flash_attn_varlen_func(q, k, v, 64, cu, 64, s_aux=aux, **common)

    descale = torch.ones((1, 8), dtype=torch.float32, device=DEVICE)
    with pytest.raises(NotImplementedError, match="scheduler_metadata"):
        fa.flash_attn_varlen_func(
            q,
            k,
            v,
            64,
            cu,
            64,
            scheduler_metadata=torch.zeros((1, 8), dtype=torch.int32, device=DEVICE),
            q_descale=descale,
            k_descale=descale,
            v_descale=descale,
            **common,
        )


def test_fa3_rejects_alibi_slopes():
    """The FA3 branch asserts ALiBi away rather than ignoring it."""
    q, k, v, cu = _self_attention_inputs(64, 8, 8, 64, seed=13)
    slopes = torch.full((8,), 0.5, dtype=torch.float32, device=DEVICE)
    with pytest.raises(AssertionError, match="Alibi is not supported in FA3"):
        fa.flash_attn_varlen_func(
            q,
            k,
            v,
            64,
            cu,
            64,
            cu_seqlens_k=cu,
            alibi_slopes=slopes,
            fa_version=3,
        )


def test_wrapper_requires_key_length_metadata():
    """``cu_seqlens_k`` and ``seqused_k`` are mutually exclusive but mandatory."""
    q, k, v, cu = _self_attention_inputs(64, 8, 8, 64, seed=17)
    with pytest.raises(AssertionError, match="must be provided"):
        fa.flash_attn_varlen_func(q, k, v, 64, cu, 64, fa_version=2)
    used = torch.tensor([64], dtype=torch.int32, device=DEVICE)
    with pytest.raises(AssertionError, match="cannot be provided at the same time"):
        fa.flash_attn_varlen_func(
            q, k, v, 64, cu, 64, cu_seqlens_k=cu, seqused_k=used, fa_version=2
        )
    with pytest.raises(AssertionError, match="seqused_k must be provided"):
        fa.flash_attn_varlen_func(
            q,
            k,
            v,
            64,
            cu,
            64,
            cu_seqlens_k=cu,
            block_table=torch.zeros((1, 1), dtype=torch.int32, device=DEVICE),
            fa_version=2,
        )


# ---------------------------------------------------------------------------
# Numeric cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fa_version", FA_VERSIONS)
@pytest.mark.parametrize("causal,seq_len,num_heads,kv_heads,head_dim", ATTN_CASES)
def test_varlen_matches_golden(
    fa_version, causal, seq_len, num_heads, kv_heads, head_dim
):
    """The PART 4 shape sweep, compared against the fp32 softmax golden."""
    q, k, v, cu = _self_attention_inputs(
        seq_len, num_heads, kv_heads, head_dim, seed=1000 + seq_len + num_heads
    )
    out = _run_varlen(q, k, v, cu, cu, causal=causal, fa_version=fa_version)
    expected = _golden_varlen(q, k, v, cu, causal=causal)

    assert tuple(out.shape) == (seq_len, num_heads, head_dim)
    assert out.dtype == torch.bfloat16
    ref.assert_close(out, expected, "attention")


@pytest.mark.parametrize("fa_version", FA_VERSIONS)
@pytest.mark.parametrize(
    "causal", [pytest.param(False, id="full"), pytest.param(True, id="causal")]
)
def test_varlen_ragged_batch_matches_golden(fa_version, causal):
    """A ragged batch must keep each sequence's mask inside its own slice."""
    cu = _cumsum(RAGGED_SEQ_LENS)
    total = int(cu[-1].item())
    num_heads, kv_heads, head_dim = 16, 8, 64
    q = _randn(total, num_heads, head_dim, seed=2001)
    k = _randn(total, kv_heads, head_dim, seed=2002)
    v = _randn(total, kv_heads, head_dim, seed=2003)

    out = _run_varlen(q, k, v, cu, cu, causal=causal, fa_version=fa_version)
    expected = _golden_varlen(q, k, v, cu, causal=causal)

    assert tuple(out.shape) == (total, num_heads, head_dim)
    ref.assert_close(out, expected, "attention")


@pytest.mark.parametrize("fa_version", FA_VERSIONS)
@pytest.mark.parametrize("causal,seq_len,num_heads,kv_heads,head_dim", SMALL_CASES)
def test_softmax_lse_matches_golden(
    fa_version, causal, seq_len, num_heads, kv_heads, head_dim
):
    """``return_softmax_lse`` yields the natural-log softmax denominator."""
    q, k, v, cu = _self_attention_inputs(
        seq_len, num_heads, kv_heads, head_dim, seed=3000 + seq_len
    )
    out, lse = _run_varlen(
        q,
        k,
        v,
        cu,
        cu,
        causal=causal,
        fa_version=fa_version,
        return_softmax_lse=True,
    )
    expected_out = _golden_varlen(q, k, v, cu, causal=causal)
    expected_lse = ref.ref_attention_lse(
        q.transpose(0, 1).unsqueeze(0),
        k.transpose(0, 1).unsqueeze(0),
        v.transpose(0, 1).unsqueeze(0),
        causal=causal,
        query_chunk=QUERY_CHUNK,
    )[0]

    assert tuple(lse.shape) == (num_heads, seq_len)
    assert lse.dtype == torch.float32
    ref.assert_close(out, expected_out, "attention")
    ref.assert_close(lse, expected_lse, "fp32", atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("causal,seq_len,num_heads,kv_heads,head_dim", SMALL_CASES)
def test_fa2_and_fa3_agree(causal, seq_len, num_heads, kv_heads, head_dim):
    """Both wheels implement the same math, so their outputs must be close."""
    q, k, v, cu = _self_attention_inputs(
        seq_len, num_heads, kv_heads, head_dim, seed=4000 + seq_len
    )
    out2 = _run_varlen(q, k, v, cu, cu, causal=causal, fa_version=2)
    out3 = _run_varlen(q, k, v, cu, cu, causal=causal, fa_version=3)
    expected = _golden_varlen(q, k, v, cu, causal=causal)

    # Anchor both to the golden first: parity alone would pass if both wheels
    # were wrong in the same way.
    ref.assert_close(out2, expected, "attention", msg="FA2 diverged from golden")
    ref.assert_close(out3, expected, "attention", msg="FA3 diverged from golden")
    ref.assert_close(out3, out2, "attention", msg="FA3 != FA2")


@pytest.mark.parametrize("fa_version", FA_VERSIONS)
def test_out_buffer_receives_the_result(fa_version):
    """The caller-supplied ``out`` tensor is filled (FA2 cannot write inplace)."""
    seq_len, num_heads, kv_heads, head_dim = 256, 16, 8, 64
    q, k, v, cu = _self_attention_inputs(
        seq_len, num_heads, kv_heads, head_dim, seed=5001
    )
    out = torch.zeros(
        (seq_len, num_heads, head_dim), dtype=torch.bfloat16, device=DEVICE
    )
    returned = _run_varlen(q, k, v, cu, cu, causal=True, fa_version=fa_version, out=out)
    expected = _golden_varlen(q, k, v, cu, causal=True)

    assert bool(out.abs().sum() > 0), "out buffer was never written"
    assert torch.equal(returned, out)
    ref.assert_close(out, expected, "attention")


@pytest.mark.parametrize("fa_version", FA_VERSIONS)
def test_forward_is_bitwise_repeatable(fa_version):
    """Repeated forwards on identical inputs must be bit-identical."""
    q, k, v, cu = _self_attention_inputs(512, 16, 8, 128, seed=6001)
    first = _run_varlen(q, k, v, cu, cu, causal=True, fa_version=fa_version)
    second = _run_varlen(q, k, v, cu, cu, causal=True, fa_version=fa_version)
    assert torch.equal(first, second)


@pytest.mark.parametrize("fa_version", FA_VERSIONS)
@pytest.mark.parametrize(
    "head_dim", [pytest.param(64, id="d64"), pytest.param(128, id="d128")]
)
def test_paged_decode_with_block_table_matches_golden(fa_version, head_dim):
    """Decode through a shuffled block table using ``seqused_k``."""
    # The FA2 paged kernel is built for a 256-token page and refuses anything
    # else outright ("Paged KV cache block size must be divisible by 256"),
    # while FA3 walks the table page by page and accepts the finer 64-token
    # page.  Each wheel is therefore measured on the page size it supports;
    # every other shape below (page count, cache size, table, golden gather)
    # is derived from ``block_size``.
    block_size = 256 if fa_version == 2 else 64
    seq_lens = (128, 200, 65, 1)
    batch = len(seq_lens)
    num_heads, kv_heads = 16, 8
    max_blocks = max((length + block_size - 1) // block_size for length in seq_lens)
    num_blocks = batch * max_blocks

    q = _randn(batch, num_heads, head_dim, seed=7001)
    k_cache = _randn(num_blocks, block_size, kv_heads, head_dim, seed=7002)
    v_cache = _randn(num_blocks, block_size, kv_heads, head_dim, seed=7003)
    permutation = torch.randperm(num_blocks, generator=_generator(7004), device="cuda")
    block_table = permutation.reshape(batch, max_blocks).to(torch.int32)
    seqused_k = torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE)
    cu_seqlens_q = _cumsum([1] * batch)

    out = fa.flash_attn_varlen_func(
        q,
        k_cache,
        v_cache,
        1,
        cu_seqlens_q,
        max(seq_lens),
        seqused_k=seqused_k,
        block_table=block_table,
        causal=True,
        fa_version=fa_version,
    )

    assert tuple(out.shape) == (batch, num_heads, head_dim)
    for index, length in enumerate(seq_lens):
        k_dense = _gather_paged(k_cache, block_table, length, block_size, index)
        v_dense = _gather_paged(v_cache, block_table, length, block_size, index)
        expected = ref.ref_attention(
            q[index].unsqueeze(0).unsqueeze(2),
            k_dense.permute(1, 0, 2).unsqueeze(0),
            v_dense.permute(1, 0, 2).unsqueeze(0),
            causal=True,
        )[0, :, 0]
        ref.assert_close(
            out[index],
            expected,
            "attention",
            msg=f"request {index} (seq_len={length}) diverged",
        )


@pytest.mark.parametrize("fa_version", FA_VERSIONS)
def test_non_contiguous_operands_are_repaired(fa_version):
    """``maybe_contiguous`` must copy operands whose last stride is not 1."""
    seq_len, num_heads, kv_heads, head_dim = 256, 16, 8, 64
    q, k, v, cu = _self_attention_inputs(
        seq_len, num_heads, kv_heads, head_dim, seed=8001
    )
    dense = _run_varlen(q, k, v, cu, cu, causal=True, fa_version=fa_version)

    # Store q head-major so the ``[S, H, D]`` view has stride(-1) == H, which
    # is exactly the case ``maybe_contiguous`` is written to repair.
    source = torch.empty(
        (seq_len, head_dim, num_heads), dtype=torch.bfloat16, device=DEVICE
    )
    source.copy_(q.transpose(1, 2))
    strided_q = source.transpose(1, 2)
    assert strided_q.stride(-1) != 1
    assert torch.equal(strided_q.contiguous(), q)

    strided = _run_varlen(strided_q, k, v, cu, cu, causal=True, fa_version=fa_version)
    expected = _golden_varlen(q, k, v, cu, causal=True)

    ref.assert_close(dense, expected, "attention")
    assert torch.equal(strided, dense)


@pytest.mark.cap89
def test_fp8_operands_with_descales_match_dequantised_golden():
    """FA3 FP8 path: descaled fp8 operands must match the dequantised golden.

    890P only -- ``flash_attn_supports_kv_cache_dtype`` is False on 810E
    because capability (8, 0) has no FP8 tensor support.

    The PPU ``flash_attn_3._C`` requires **all three** descale tensors to be
    ``(batch_size, num_heads_k)``, including ``q_descale`` -- verified from the
    wheel's own diagnostics (``strings flash_attn_3/_C.abi3.so`` reports
    "q_descale must have shape (batch_size, num_heads_k)") and from a
    single-column perturbation probe on ZW-890P: widening ``q_descale[0, 0]``
    moved query heads 0 and 1 only and left heads 2..15 byte-identical, so with
    ``num_heads / num_heads_k == 2`` the kernel indexes ``q_descale`` by GQA
    group, i.e. every query head inside one KV group shares a single descale.
    Under MQA (``kv_heads == 1``) that degenerates to ``(batch_size, 1)``.
    """
    if not current_platform.supports_fp8():
        pytest.fail("cap89 marker let a non-FP8 chip reach the FP8 case")

    seq_len, num_heads, kv_heads, head_dim = 512, 16, 8, 128
    q, k, v, cu = _self_attention_inputs(
        seq_len, num_heads, kv_heads, head_dim, seed=9001
    )
    q_fp8, q_scale = ref.quant_per_tensor_fp8(q)
    k_fp8, k_scale = ref.quant_per_tensor_fp8(k)
    v_fp8, v_scale = ref.quant_per_tensor_fp8(v)

    def _descale(scale):
        # One column per KV head, which is what the kernel asks of q/k/v alike.
        return torch.full(
            (1, kv_heads), float(scale.item()), dtype=torch.float32, device=DEVICE
        )

    out = fa.flash_attn_varlen_func(
        q_fp8.contiguous(),
        k_fp8.contiguous(),
        v_fp8.contiguous(),
        seq_len,
        cu,
        seq_len,
        cu_seqlens_k=cu,
        causal=True,
        fa_version=3,
        q_descale=_descale(q_scale),
        k_descale=_descale(k_scale),
        v_descale=_descale(v_scale),
    )

    expected = ref.ref_varlen_attention(
        ref.dequant_fp8(q_fp8, q_scale),
        ref.dequant_fp8(k_fp8, k_scale),
        ref.dequant_fp8(v_fp8, v_scale),
        cu,
        cu,
        causal=True,
        query_chunk=QUERY_CHUNK,
    )

    assert tuple(out.shape) == (seq_len, num_heads, head_dim)
    # ``attention_fp8``, not ``fp8_relaxed``: FA3 also rounds the softmax
    # probabilities to E4M3FN before ``P @ V``, which the fp32 golden keeps
    # exact.  See the ``TOL`` comment for the ZW-890P measurement.
    ref.assert_close(out, expected, "attention_fp8")
