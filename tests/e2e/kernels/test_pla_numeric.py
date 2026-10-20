# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numeric E2E oracles for the PPU PLA linear-attention kernels (CI plan PART 4).

Scope split with the existing suites (plan PART 4: "avoid re-building oracles"):

* ``tests/e2e/fork_port/test_gdn_pla_decode.py`` / ``test_gdn_pla_prefill.py``
  / ``test_kda.py`` cover **CUDA vs Triton parity** and the fallback gating.
* This file covers the two things those suites deliberately do not: an
  **absolute fp32 golden** (``tests/e2e/kernels/_reference.py``) for the
  dispatchable operators, and the **backend-selection / dispatch linkage**
  assertions that tie ``VLLM_SAIL_USE_PLA`` and the per-call constraints to
  the kernel actually invoked on real hardware.

Chip routing: PLA runs on both PPU chips, so the module marker is ``ppu``.

===============================  ==========  ========
operator                         chips       marker
===============================  ==========  ========
GDN decode (PLA k_last)          810E, 890P  ``ppu``
GDN packed decode (k_last_packed) 810E, 890P ``ppu``
KDA mode of the fused update     810E, 890P  ``ppu``
GDN prefill (Triton/FLA)         810E, 890P  ``ppu``
GDN prefill (PLA FlashQLA)       810E, 890P  ``ppu``
===============================  ==========  ========

The PLA *decode* wheel is mandatory: the ``native_kernels`` fixture resolves
``get_sail_cuda_pla_k_last()`` / ``_packed()``, which raise ``ImportError``
when the ``pla`` package is missing, so an absent operator fails the run
instead of silently degrading to Triton.  The ``pla.prefill`` namespace is a
separate opt-in build (``PLA_BUILD_NAMESPACE=pla.prefill``) whose absence is
contractually handled by a Triton fallback, so the FlashQLA-specific items
report a skip with that reason while the Triton prefill golden still runs.
"""

from __future__ import annotations

from types import SimpleNamespace

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

from tests.e2e.gdn._tolerances import (  # noqa: E402
    get_default_atol,
    get_default_rtol,
)
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (  # noqa: E402
    ChunkGatedDeltaRule,
    _pla_prefill_supported,
    _resolve_gdn_prefill_backend,
)
from vllm.third_party.flash_linear_attention.ops import (  # noqa: E402
    fused_recurrent_gated_delta_rule_packed_decode,
    fused_sigmoid_gating_delta_rule_update,
)

from tests.e2e.kernels import _reference as ref  # noqa: E402
from vllm_sail.attention import pla_decode, pla_kda, pla_prefill  # noqa: E402

pytestmark = pytest.mark.ppu

DEVICE = torch.device("cuda")

# GDN head configuration: H key/query heads shared by HV value heads (GQA-style
# expansion, ``i_h = i_hv // (HV // H)`` in the Triton kernel).
H = 4
HV = 8

# Plan PART 4 parametrisation: seq_len x head_dim x batch.  Only K == V == 128
# is eligible for the PLA CUDA kernels; 64/256 exercise the Triton fallback and
# are asserted to do so by ``test_gdn_decode_dispatch_gating``.
#
# PPU SDK defect (see ``_PLA_CUDA_LONG_SEQ_BUG``): the closed-source
# ``pla.decode`` ``cuda_k_last`` kernel is rejected by the launcher with
# ``hggcErrorLaunchOutOfResources`` ("too many resources requested for
# launch") once one sequence carries more than ~128 tokens at
# ``K == V == 128`` -- measured on ZW-890P (8, 9): GDN fails from ``T=192``
# up, KDA from ``T=128`` up, while the community Triton kernel runs every one
# of those shapes and matches the fp32 golden to ~1e-8.  Only the cases that
# actually dispatch to the PLA CUDA kernel are marked; the Triton-fallback and
# packed-decode (``T=1``) suites stay unmarked so a silent SDK fix cannot
# escape notice there.
_PLA_CUDA_LONG_SEQ_BUG = (
    "PPU SDK defect: pla.decode cuda_k_last is rejected with "
    "hggcErrorLaunchOutOfResources ('too many resources requested for launch') "
    "at K == V == 128 once one sequence carries >128 tokens (GDN from T=192, "
    "KDA from T=128); minimal repro b1-t192-d128 through "
    "fused_sigmoid_gating_delta_rule_update with the PLA fast path forced. The "
    "Triton fallback matches the fp32 golden on the same shapes, so this is a "
    "launch-configuration defect in the SDK wheel, not a numeric one. Pending "
    "SDK fix."
)


def _cuda_long_seq_defect(seq_len: int, head_dim: int) -> bool:
    """True when a case dispatches to the PLA CUDA kernel at a broken shape."""
    return head_dim == 128 and seq_len > 128


def _decode_cases(*, pla_cuda_defect_xfail: bool) -> list:
    """Build the decode parametrisation, optionally marking the SDK defect."""
    cases = []
    for batch in (1, 4):
        for seq_len in (1, 17, 512, 2048):
            for head_dim in (64, 128, 256):
                marks = ()
                if pla_cuda_defect_xfail and _cuda_long_seq_defect(seq_len, head_dim):
                    marks = (
                        pytest.mark.xfail(strict=True, reason=_PLA_CUDA_LONG_SEQ_BUG),
                    )
                cases.append(
                    pytest.param(
                        batch,
                        seq_len,
                        head_dim,
                        id=f"b{batch}-t{seq_len}-d{head_dim}",
                        marks=marks,
                    )
                )
    return cases


# Cases that go through the production dispatch (PLA CUDA at d128, Triton
# elsewhere), so the SDK launch defect is reachable.
DECODE_CASES = _decode_cases(pla_cuda_defect_xfail=True)
# Cases forced onto the Triton fallback, where the defect cannot trigger.
TRITON_DECODE_CASES = _decode_cases(pla_cuda_defect_xfail=False)
# Prefill shapes stay small: the fp32 golden is a Python loop over tokens.
PREFILL_CASES = [
    pytest.param(17, 1, id="t17-n1"),
    pytest.param(64, 1, id="t64-n1"),
    pytest.param(64, 4, id="t64-n4"),
    pytest.param(512, 4, id="t512-n4"),
]

_NO_PLA = (
    "the PPU `pla` decode wheel is mandatory for this suite: "
    "install it, or set VLLM_SAIL_USE_PLA=0 to run the Triton-only suites"
)


@pytest.fixture(scope="module", autouse=True)
def native_kernels():
    """Load the native extension strictly and resolve the PLA decode kernels."""
    from vllm_sail.native import install
    from vllm_sail.native.extensions import import_kernels

    import_kernels(strict=True)
    install()

    assert pla_decode.get_sail_cuda_pla_k_last() is not None, _NO_PLA
    assert pla_decode.get_sail_cuda_pla_k_last_packed() is not None, _NO_PLA
    # The prefill routing hooks are patched in by vllm_sail; if the patch did
    # not land, the dispatch assertions below would silently test upstream.
    assert hasattr(ChunkGatedDeltaRule, "forward_pla"), (
        "ChunkGatedDeltaRule.forward_pla was not patched in by "
        "vllm_sail.register_out_of_tree()"
    )
    assert callable(_pla_prefill_supported)
    assert callable(_resolve_gdn_prefill_backend)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(seed)


def _force_triton(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route the GDN decode wrappers onto the community Triton kernels."""
    monkeypatch.setattr(pla_decode, "_resolved", True)
    monkeypatch.setattr(pla_decode, "_k_last_fn", None)
    monkeypatch.setattr(pla_decode, "_k_last_packed_fn", None)


def _force_cuda(monkeypatch: pytest.MonkeyPatch):
    """Route the GDN decode wrappers onto the PPU SAIL CUDA PLA kernels."""
    import pla.decode as pla_decode_wheel

    monkeypatch.setattr(pla_decode, "_resolved", True)
    monkeypatch.setattr(
        pla_decode,
        "_k_last_fn",
        pla_decode_wheel.fused_sigmoid_gating_delta_rule_forward_k_last,
    )
    monkeypatch.setattr(
        pla_decode,
        "_k_last_packed_fn",
        pla_decode_wheel.fused_sigmoid_gating_delta_rule_forward_k_last_packed,
    )
    return pla_decode_wheel


def _decode_inputs(
    batch: int,
    seq_len: int,
    head_dim: int,
    *,
    seed: int,
    is_kda: bool = False,
    state_dtype: torch.dtype = torch.float32,
    indices_dtype: torch.dtype = torch.int32,
    dt_bias_dtype: torch.dtype = torch.bfloat16,
) -> SimpleNamespace:
    """GDN/KDA decode inputs mirroring the production call sites.

    ``q``/``k`` are ``[1, total, H, D]``, ``v`` is ``[1, total, HV, D]``
    (varlen-flattened), ``a``/``b`` are indexed by the *global* token position
    (``[total, HV]``, or ``[total, HV, D]`` for ``a`` in KDA mode), and the ssm
    state pool is ``[num_slots, HV, V, K]`` with slot 0 left unused because
    vLLM reserves it as ``NULL_BLOCK_ID``.

    Two index views are returned because the fused update writes the state back
    *per timestep* through ``ssm_state_indices[i_n * stride_seq + i_t]``:

    * ``slots`` is the ``[batch]`` slot list. It is what the fp32 golden and the
      ``T=1`` packed-decode entry point consume.
    * ``indices`` is what the kernel gets. ``vllm_sail``'s dispatch documents
      the supported shapes as "1D indices => the fast decode path (T=1)" and
      "2D indices => per-timestep writes to the 2D slots" (see
      ``vllm_sail/patch/enhancement/attention/gdn.py``), so ``indices`` is 1D
      for ``seq_len == 1`` and the ``[batch, seq_len]`` slot map otherwise.
      Handing a multi-token sequence a 1D index tensor makes the Triton
      write-back read past the end of the tensor and scatter the state into
      arbitrary slots (measured: ``out`` still matches the golden because the
      read-out does not go through the index, while ``state`` is wrong and a
      wild store can even NaN the output buffer).

    ``dt_bias_dtype`` defaults to q's dtype; see :func:`_kda_dt_bias_dtype` for
    the one path that reads it as float32 instead.
    """
    generator = _generator(seed)
    total = batch * seq_len
    num_slots = batch + 1

    def randn(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, generator=generator, device=DEVICE, dtype=dtype)

    q = randn(1, total, H, head_dim) * 0.1
    k = randn(1, total, H, head_dim) * 0.1
    v = randn(1, total, HV, head_dim) * 0.1
    b = randn(total, HV) * 0.1
    a_log = randn(HV, dtype=torch.float32) * 0.1
    if is_kda:
        a = randn(total, HV, head_dim) * 0.1
        dt_bias = randn(HV, head_dim, dtype=dt_bias_dtype) * 0.1
    else:
        a = randn(total, HV) * 0.1
        dt_bias = randn(HV, dtype=dt_bias_dtype) * 0.1
    state = randn(num_slots, HV, head_dim, head_dim, dtype=state_dtype) * 0.05

    # Real requests occupy slots >= 1; slot 0 is the reserved null block.
    slots = (
        torch.randperm(num_slots - 1, generator=generator, device=DEVICE)[:batch] + 1
    ).to(indices_dtype)
    # 1D for the T=1 fast-decode path, [batch, seq_len] for the per-timestep
    # write-back; see the docstring above.
    indices = (
        slots.unsqueeze(1).repeat(1, seq_len).contiguous() if seq_len > 1 else slots
    )
    cu_seqlens = torch.arange(
        0, (batch + 1) * seq_len, seq_len, dtype=torch.int32, device=DEVICE
    )
    return SimpleNamespace(
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        a_log=a_log,
        dt_bias=dt_bias,
        state=state,
        slots=slots,
        indices=indices,
        cu_seqlens=cu_seqlens,
        scale=head_dim**-0.5,
        num_slots=num_slots,
    )


def _kda_dt_bias_dtype(head_dim: int, seq_len: int) -> torch.dtype:
    """``dt_bias`` dtype expected by the kernel this case dispatches to.

    The SAIL CUDA kernel is reachable only for ``K == V == 128`` and reads
    ``dt_bias`` as float32 on the KDA MTP-verify path, i.e. when a request
    carries more than one token; it rejects any other dtype there.  Every other
    path -- single-token decode and the Triton fallback, which loads it with
    ``tl.load(p_dt_bias).to(tl.float32)`` -- reads it with q's dtype, so those
    keep bf16 and stay bit-comparable with the evidence already collected.
    """
    if head_dim == 128 and seq_len > 1:
        return torch.float32
    return torch.bfloat16


def _run_update(inp: SimpleNamespace, state: torch.Tensor, *, is_kda: bool = False):
    return fused_sigmoid_gating_delta_rule_update(
        A_log=inp.a_log,
        a=inp.a,
        b=inp.b,
        dt_bias=inp.dt_bias,
        q=inp.q,
        k=inp.k,
        v=inp.v,
        scale=inp.scale,
        initial_state=state,
        inplace_final_state=True,
        cu_seqlens=inp.cu_seqlens,
        ssm_state_indices=inp.indices,
        use_qk_l2norm_in_kernel=True,
        is_kda=is_kda,
    )


def _golden_update(inp: SimpleNamespace, state: torch.Tensor, *, is_kda: bool = False):
    return ref.ref_linear_attn(
        inp.a_log,
        inp.a,
        inp.b,
        inp.dt_bias,
        inp.q,
        inp.k,
        inp.v,
        scale=inp.scale,
        initial_state=state,
        state_indices=inp.slots,
        cu_seqlens=inp.cu_seqlens,
        use_qk_l2norm=True,
        is_kda=is_kda,
    )


def _tolerance(seq_len: int) -> str:
    """Plan PART 4: 2e-2 normally, relaxed to 5e-2 for long recurrences."""
    return "linear_attn" if seq_len <= 512 else "linear_attn_decode"


def _assert_matches_golden(
    out: torch.Tensor,
    state: torch.Tensor,
    expected_out: torch.Tensor,
    expected_state: torch.Tensor,
    inp: SimpleNamespace,
    seq_len: int,
) -> None:
    kind = _tolerance(seq_len)
    assert out.shape == expected_out.shape
    ref.assert_close(out, expected_out, kind)

    slots = inp.slots.to(torch.int64)
    ref.assert_close(state[slots], expected_state[slots], kind)
    # NULL_BLOCK_ID isolation: slot 0 is never referenced and must be pristine.
    assert torch.equal(state[0], expected_state[0])


# ---------------------------------------------------------------------------
# backend selection
# ---------------------------------------------------------------------------


def test_pla_decode_handles_follow_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """``VLLM_SAIL_USE_PLA`` gates the resolved PLA decode kernel handles."""
    monkeypatch.setattr(pla_decode, "_resolved", False)
    monkeypatch.setenv("VLLM_SAIL_USE_PLA", "1")
    k_last = pla_decode.get_sail_cuda_pla_k_last()
    k_last_packed = pla_decode.get_sail_cuda_pla_k_last_packed()
    assert callable(k_last) and callable(k_last_packed)

    import pla.decode as pla_decode_wheel

    assert k_last is pla_decode_wheel.fused_sigmoid_gating_delta_rule_forward_k_last
    assert (
        k_last_packed
        is pla_decode_wheel.fused_sigmoid_gating_delta_rule_forward_k_last_packed
    )

    monkeypatch.setattr(pla_decode, "_resolved", False)
    monkeypatch.setenv("VLLM_SAIL_USE_PLA", "0")
    assert pla_decode.get_sail_cuda_pla_k_last() is None
    assert pla_decode.get_sail_cuda_pla_k_last_packed() is None


def test_pla_kda_handles_follow_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """``get_pla_kda_kernel`` resolves eagerly and honours the env switch.

    ``pla.decode.kda`` ships with the mandatory decode namespace; ``pla.prefill``
    is a separate opt-in build, and ``get_pla_kda_kernel`` turns its absence
    into an explicit ``ImportError`` rather than a silent ``None``.
    """
    monkeypatch.setenv("VLLM_SAIL_USE_PLA", "1")
    decode_fn = pla_kda.get_pla_kda_kernel("decode")
    assert callable(decode_fn), (
        "pla.decode.kda.fused_kda_decode_mega_forward is missing from the PPU "
        "PLA SDK; KDA decode cannot be routed to the SAIL CUDA kernel"
    )
    try:
        prefill_fn = pla_kda.get_pla_kda_kernel("prefill")
    except ImportError:
        prefill_fn = None
    assert prefill_fn is None or callable(prefill_fn)

    monkeypatch.setenv("VLLM_SAIL_USE_PLA", "0")
    assert pla_kda.get_pla_kda_kernel("decode") is None
    assert pla_kda.get_pla_kda_kernel("prefill") is None


def _stub_vllm_config(
    *,
    head_dim: int = 128,
    num_k_heads: int = 16,
    num_v_heads: int = 32,
    tp_size: int = 1,
) -> SimpleNamespace:
    hf_text_config = SimpleNamespace(
        linear_key_head_dim=head_dim,
        linear_value_head_dim=head_dim,
        linear_num_key_heads=num_k_heads,
        linear_num_value_heads=num_v_heads,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=hf_text_config),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp_size),
        additional_config={"gdn_prefill_backend": "auto"},
    )


def test_pla_prefill_backend_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_resolve_gdn_prefill_backend`` picks PLA only when every gate passes."""
    monkeypatch.setattr(pla_prefill, "_resolved", False)
    monkeypatch.setenv("VLLM_SAIL_USE_PLA", "1")

    available = pla_prefill.get_sail_cuda_pla_prefill_fwd() is not None
    configs = pla_prefill.get_sail_cuda_pla_prefill_head_configs()
    # The whitelist and the kernel handle are resolved together.
    assert available == bool(configs)

    supported = next(iter(configs)) if configs else (32, 16)
    num_v_heads, num_k_heads = supported

    def resolve(**overrides):
        kwargs = dict(
            head_dim=128,
            num_k_heads=num_k_heads,
            num_v_heads=num_v_heads,
            tp_size=1,
        )
        kwargs.update(overrides)
        return _resolve_gdn_prefill_backend(_stub_vllm_config(**kwargs))

    assert resolve() == ("auto", "pla" if available else "triton")
    # FlashQLA instantiates a fixed set of head dims; anything else is Triton.
    assert resolve(head_dim=64) == ("auto", "triton")
    assert resolve(num_v_heads=num_v_heads + 1) == ("auto", "triton")
    # An explicit non-PLA request must never be upgraded.
    config = _stub_vllm_config(num_k_heads=num_k_heads, num_v_heads=num_v_heads)
    config.additional_config = {"gdn_prefill_backend": "triton"}
    assert _resolve_gdn_prefill_backend(config) == ("triton", "triton")

    # With the switch off, PLA is unreachable and the support probe agrees.
    monkeypatch.setattr(pla_prefill, "_resolved", False)
    monkeypatch.setenv("VLLM_SAIL_USE_PLA", "0")
    assert pla_prefill.get_sail_cuda_pla_prefill_fwd() is None
    assert pla_prefill.get_sail_cuda_pla_prefill_head_configs() == frozenset()
    assert resolve() == ("auto", "triton")
    assert _pla_prefill_supported(_stub_vllm_config()) is False


def test_forward_pla_requires_prenormalised_qk() -> None:
    """``forward_pla`` refuses to double-normalise q/k."""
    inp = _decode_inputs(1, 4, 128, seed=7)
    with pytest.raises(AssertionError, match="l2-normalized"):
        ChunkGatedDeltaRule.forward_pla(
            None,
            inp.q,
            inp.k,
            inp.v,
            torch.zeros(1, 4, HV, device=DEVICE),
            torch.zeros(1, 4, HV, device=DEVICE),
            None,
            True,
            use_qk_l2norm_in_kernel=True,
        )


# ---------------------------------------------------------------------------
# GDN decode: absolute fp32 golden
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("batch", "seq_len", "head_dim"), DECODE_CASES)
def test_gdn_decode_update_matches_golden(
    monkeypatch: pytest.MonkeyPatch, batch: int, seq_len: int, head_dim: int
) -> None:
    """``fused_sigmoid_gating_delta_rule_update`` against the fp32 recurrence.

    ``head_dim == 128`` is PLA-eligible and is run on the SAIL CUDA kernel;
    64/256 must fall back to Triton and still match the same golden.
    """
    inp = _decode_inputs(batch, seq_len, head_dim, seed=1000 + seq_len + head_dim)
    baseline = inp.state.clone()

    if head_dim == 128:
        _force_cuda(monkeypatch)
    else:
        _force_triton(monkeypatch)
    state = baseline.clone()
    out, final_state = _run_update(inp, state)
    torch.cuda.synchronize()
    assert final_state is state  # inplace_final_state contract

    expected_out, expected_state = _golden_update(inp, baseline)
    _assert_matches_golden(out, state, expected_out, expected_state, inp, seq_len)


@pytest.mark.parametrize(("batch", "seq_len", "head_dim"), TRITON_DECODE_CASES[:12])
def test_gdn_decode_update_triton_fallback_matches_golden(
    monkeypatch: pytest.MonkeyPatch, batch: int, seq_len: int, head_dim: int
) -> None:
    """The Triton fallback must reproduce the golden on every head dim."""
    inp = _decode_inputs(batch, seq_len, head_dim, seed=2000 + seq_len + head_dim)
    baseline = inp.state.clone()

    _force_triton(monkeypatch)
    state = baseline.clone()
    out, _ = _run_update(inp, state)
    torch.cuda.synchronize()

    expected_out, expected_state = _golden_update(inp, baseline)
    _assert_matches_golden(out, state, expected_out, expected_state, inp, seq_len)


@pytest.mark.parametrize("batch", [1, 4])
def test_gdn_decode_cuda_triton_parity(
    monkeypatch: pytest.MonkeyPatch, batch: int
) -> None:
    """PLA CUDA and Triton agree within the shared fork_port tolerances."""
    inp = _decode_inputs(batch, 17, 128, seed=3000 + batch)
    pla_decode_wheel = _force_cuda(monkeypatch)
    assert pla_decode_wheel is not None

    state_c = inp.state.clone()
    out_c, _ = _run_update(inp, state_c)
    _force_triton(monkeypatch)
    state_t = inp.state.clone()
    out_t, _ = _run_update(inp, state_t)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        out_c, out_t, atol=get_default_atol(out_t), rtol=get_default_rtol(out_t)
    )
    torch.testing.assert_close(state_c, state_t, atol=1e-3, rtol=1e-2)


@pytest.mark.parametrize(
    ("head_dim", "state_dtype", "indices_dtype", "expect_cuda"),
    [
        pytest.param(128, torch.float32, torch.int32, True, id="eligible"),
        pytest.param(64, torch.float32, torch.int32, False, id="head-dim-64"),
        pytest.param(256, torch.float32, torch.int32, False, id="head-dim-256"),
        pytest.param(128, torch.bfloat16, torch.int32, False, id="bf16-state-pool"),
        pytest.param(128, torch.float32, torch.int64, False, id="int64-indices"),
    ],
)
def test_gdn_decode_dispatch_gating(
    monkeypatch: pytest.MonkeyPatch,
    head_dim: int,
    state_dtype: torch.dtype,
    indices_dtype: torch.dtype,
    expect_cuda: bool,
) -> None:
    """A spy on the PLA handle proves which kernel the wrapper dispatched to.

    This is the real-machine counterpart of the capability matrix: the CUDA
    fast path requires an fp32 state pool, int32 slot indices and
    ``K == V == 128``; anything else must reach the Triton kernel.
    """
    real = pla_decode.get_sail_cuda_pla_k_last()
    assert callable(real), _NO_PLA
    calls: list[tuple] = []

    def spy(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(pla_decode, "_resolved", True)
    monkeypatch.setattr(pla_decode, "_k_last_fn", spy)

    inp = _decode_inputs(
        2,
        3,
        head_dim,
        seed=4000 + head_dim,
        state_dtype=state_dtype,
        indices_dtype=indices_dtype,
    )
    out, _ = _run_update(inp, inp.state.clone())
    torch.cuda.synchronize()
    assert bool(calls) is expect_cuda
    if expect_cuda:
        assert out.shape == (1, 6, HV, head_dim)


@pytest.mark.parametrize("batch", [1, 4, 32])
def test_gdn_packed_decode_matches_golden(
    monkeypatch: pytest.MonkeyPatch, batch: int
) -> None:
    """``fused_recurrent_gated_delta_rule_packed_decode`` vs the fp32 golden.

    The packed wrapper is the single-token (T=1) decode entry point: q/k/v are
    concatenated along the last dim of ``mixed_qkv`` and split again inside the
    kernel, so the golden is built from the same three slices.
    """
    head_dim = 128
    inp = _decode_inputs(batch, 1, head_dim, seed=5000 + batch)
    q, k, v = inp.q, inp.k, inp.v
    mixed_qkv = torch.cat(
        [
            q.reshape(batch, H * head_dim),
            k.reshape(batch, H * head_dim),
            v.reshape(batch, HV * head_dim),
        ],
        dim=-1,
    ).contiguous()

    def run(state: torch.Tensor) -> torch.Tensor:
        out = torch.empty(batch, 1, HV, head_dim, dtype=v.dtype, device=DEVICE)
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv,
            a=inp.a,
            b=inp.b,
            A_log=inp.a_log,
            dt_bias=inp.dt_bias,
            scale=inp.scale,
            initial_state=state,
            out=out,
            ssm_state_indices=inp.slots,
            use_qk_l2norm_in_kernel=True,
        )
        return out

    baseline = inp.state.clone()
    pla_decode_wheel = _force_cuda(monkeypatch)
    assert pla_decode_wheel is not None
    state_c = baseline.clone()
    out_c = run(state_c)

    _force_triton(monkeypatch)
    state_t = baseline.clone()
    out_t = run(state_t)
    torch.cuda.synchronize()

    expected_out, expected_state = _golden_update(inp, baseline)
    for out, state in ((out_c, state_c), (out_t, state_t)):
        # The packed wrapper writes [B, 1, HV, V]; the golden is [1, B, HV, V].
        ref.assert_close(
            out.reshape(1, batch, HV, head_dim), expected_out, "linear_attn"
        )
        slots = inp.slots.to(torch.int64)
        ref.assert_close(state[slots], expected_state[slots], "linear_attn")

    torch.testing.assert_close(
        out_c, out_t, atol=get_default_atol(out_t), rtol=get_default_rtol(out_t)
    )


# ---------------------------------------------------------------------------
# KDA mode of the fused update
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("batch", "seq_len", "head_dim"),
    [
        pytest.param(1, 1, 128, id="b1-t1-d128"),
        pytest.param(4, 1, 128, id="b4-t1-d128"),
        pytest.param(1, 17, 128, id="b1-t17-d128"),
        pytest.param(4, 17, 64, id="b4-t17-d64"),
        pytest.param(
            1,
            512,
            128,
            id="b1-t512-d128",
            marks=pytest.mark.xfail(strict=True, reason=_PLA_CUDA_LONG_SEQ_BUG),
        ),
    ],
)
def test_kda_decode_update_matches_golden(
    monkeypatch: pytest.MonkeyPatch, batch: int, seq_len: int, head_dim: int
) -> None:
    """``is_kda=True`` switches the gate to per-channel decay.

    KDA widens ``a`` to ``[total, HV, K]`` and ``dt_bias`` to ``[HV, K]`` while
    ``A_log`` stays ``[HV]``; the golden mirrors the ``IS_KDA`` branch of the
    Triton kernel (``b_h *= exp(b_g[None, :])``).
    """
    inp = _decode_inputs(
        batch,
        seq_len,
        head_dim,
        seed=6000 + seq_len,
        is_kda=True,
        dt_bias_dtype=_kda_dt_bias_dtype(head_dim, seq_len),
    )
    baseline = inp.state.clone()

    if head_dim == 128:
        _force_cuda(monkeypatch)
    else:
        _force_triton(monkeypatch)
    state = baseline.clone()
    out, _ = _run_update(inp, state, is_kda=True)
    torch.cuda.synchronize()

    expected_out, expected_state = _golden_update(inp, baseline, is_kda=True)
    _assert_matches_golden(out, state, expected_out, expected_state, inp, seq_len)


# ---------------------------------------------------------------------------
# GDN prefill: absolute fp32 golden
# ---------------------------------------------------------------------------


def _prefill_inputs(total: int, num_seqs: int, head_dim: int, seed: int):
    """Cold-start prefill inputs (log-space gate, pre-sigmoid beta)."""
    generator = _generator(seed)
    assert total % num_seqs == 0
    seq_len = total // num_seqs

    def randn(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, generator=generator, device=DEVICE, dtype=dtype)

    q = randn(1, total, H, head_dim) * 0.1
    k = randn(1, total, H, head_dim) * 0.1
    v = randn(1, total, HV, head_dim) * 0.1
    # g is the forget gate *in log space*; beta is already activated.
    g = -torch.nn.functional.softplus(randn(1, total, HV, dtype=torch.float32))
    beta = torch.sigmoid(randn(1, total, HV, dtype=torch.float32)).to(torch.bfloat16)
    cu_seqlens = torch.arange(
        0, (num_seqs + 1) * seq_len, seq_len, dtype=torch.int32, device=DEVICE
    )
    # vLLM stores the pool as [N, HV, V, K]; a zero pool keeps the golden
    # independent of the FlashQLA [N, HV, K, V] transpose.
    state = torch.zeros(
        num_seqs, HV, head_dim, head_dim, dtype=torch.float32, device=DEVICE
    )
    return SimpleNamespace(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        state=state,
        scale=head_dim**-0.5,
    )


@pytest.mark.parametrize(("total", "num_seqs"), PREFILL_CASES)
def test_gdn_prefill_native_matches_golden(total: int, num_seqs: int) -> None:
    """The Triton/FLA prefill kernel against the fp32 chunk golden."""
    inp = _prefill_inputs(total, num_seqs, 128, seed=7000 + total + num_seqs)
    out, final_state = ChunkGatedDeltaRule.forward_native(
        None,
        q=inp.q,
        k=inp.k,
        v=inp.v,
        g=inp.g,
        beta=inp.beta,
        initial_state=inp.state.clone(),
        output_final_state=True,
        cu_seqlens=inp.cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()

    expected, _ = ref.ref_chunk_linear_attn(
        inp.q,
        inp.k,
        inp.v,
        inp.g,
        inp.beta,
        scale=inp.scale,
        cu_seqlens=inp.cu_seqlens,
        use_qk_l2norm=True,
    )
    assert tuple(out.shape) == (1, total, HV, 128)
    assert final_state is not None and final_state.shape[0] == num_seqs
    ref.assert_close(out, expected, "linear_attn")


@pytest.mark.parametrize(("total", "num_seqs"), PREFILL_CASES[:2])
def test_gdn_prefill_pla_matches_golden(total: int, num_seqs: int) -> None:
    """The PLA FlashQLA prefill kernel against the same fp32 golden.

    ``forward_pla`` bridges the state layout (vLLM ``[N, HV, V, K]`` ->
    FlashQLA ``[N, HV, K, V]`` and back) and requires pre-normalised q/k.
    """
    if pla_prefill.get_sail_cuda_pla_prefill_fwd() is None:
        pytest.skip(
            "the optional pla.prefill FlashQLA namespace is not built "
            "(PLA_BUILD_NAMESPACE=pla.prefill); prefill falls back to Triton, "
            "which test_gdn_prefill_native_matches_golden covers"
        )
    inp = _prefill_inputs(total, num_seqs, 128, seed=8000 + total + num_seqs)

    def l2norm(x: torch.Tensor) -> torch.Tensor:
        xf = x.to(torch.float32)
        return (xf * torch.rsqrt((xf * xf).sum(-1, keepdim=True) + 1e-6)).to(x.dtype)

    q, k = l2norm(inp.q).contiguous(), l2norm(inp.k).contiguous()
    out, final_state = ChunkGatedDeltaRule.forward_pla(
        None,
        q,
        k,
        inp.v,
        inp.g,
        inp.beta,
        inp.state.clone(),
        True,
        cu_seqlens=inp.cu_seqlens,
        use_qk_l2norm_in_kernel=False,
    )
    torch.cuda.synchronize()

    expected, _ = ref.ref_chunk_linear_attn(
        q,
        k,
        inp.v,
        inp.g,
        inp.beta,
        scale=inp.scale,
        cu_seqlens=inp.cu_seqlens,
        use_qk_l2norm=False,
    )
    assert tuple(out.shape) == (1, total, HV, 128)
    assert final_state is not None
    # The transpose bridge must hand the pool back in the vLLM layout.
    assert tuple(final_state.shape) == (num_seqs, HV, 128, 128)
    ref.assert_close(out, expected, "linear_attn")
