# SPDX-License-Identifier: Apache-2.0
"""GDN backend choices stay usable without torch, vLLM or a device."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from vllm_sail.attention.gdn_decode import (
    has_fused_gdn_kernel,
    resolve_gdn_decode_backend,
)


@pytest.mark.parametrize(
    "requested,use_pla,pla_reason,cuda_reason,active",
    [
        ("auto", True, None, None, "pla"),
        ("auto", False, None, None, "cuda"),
        ("auto", True, "FP32 state required", None, "cuda"),
        ("auto", True, "PLA unavailable", "native op unavailable", "triton"),
        ("triton", True, None, None, "triton"),
        ("pla", False, None, None, "pla"),
        ("cuda", True, None, None, "cuda"),
    ],
)
def test_backend_selection(requested, use_pla, pla_reason, cuda_reason, active):
    choice = resolve_gdn_decode_backend(
        requested, use_pla=use_pla, pla_reason=pla_reason, cuda_reason=cuda_reason
    )
    assert choice.requested == requested
    assert choice.active == active
    if active == "triton" and requested == "auto":
        assert "PLA unavailable" in choice.reason
        assert "native op unavailable" in choice.reason


@pytest.mark.parametrize("requested", ["pla", "cuda"])
def test_explicit_unavailable_backend_fails_early(requested):
    with pytest.raises(
        ValueError, match=f"VLLM_GDN_DECODE_KERNEL={requested}.*missing"
    ):
        resolve_gdn_decode_backend(
            requested, use_pla=True, pla_reason="missing PLA", cuda_reason="missing op"
        )


def test_invalid_backend_is_rejected():
    with pytest.raises(ValueError, match="invalid"):
        resolve_gdn_decode_backend(
            "invalid", use_pla=True, pla_reason=None, cuda_reason=None
        )


@pytest.mark.parametrize("real_cuda_kernel", [False, True])
def test_schema_or_composite_stub_does_not_establish_native_support(
    monkeypatch, real_cuda_kernel
):
    calls = []

    def dispatch(op, key):
        calls.append((op, key))
        return real_cuda_kernel

    torch = SimpleNamespace(
        ops=SimpleNamespace(
            _C=SimpleNamespace(fused_gdn_decode_post_conv_mtp=object())
        ),
        _C=SimpleNamespace(_dispatch_has_kernel_for_dispatch_key=dispatch),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    assert has_fused_gdn_kernel() is real_cuda_kernel
    assert calls == [("_C::fused_gdn_decode_post_conv_mtp", "CUDA")]


def test_unregistered_native_op_is_unavailable(monkeypatch):
    def dispatch(*args):
        raise RuntimeError("operator does not exist")

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            _C=SimpleNamespace(_dispatch_has_kernel_for_dispatch_key=dispatch)
        ),
    )
    assert not has_fused_gdn_kernel()
