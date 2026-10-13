# SPDX-License-Identifier: Apache-2.0
"""KDA dispatch, caller buffers and upstream API compatibility."""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tests.support.source import assert_accepts_upstream_keywords, function


@pytest.mark.parametrize("num_heads", [12, 24])
def test_pla_prefill_preserves_caller_buffers(monkeypatch, num_heads):
    calls = []
    pla = ModuleType("pla.prefill.flashkdapro")
    pla.flashkda_fwd = lambda **kw: calls.append(kw)
    monkeypatch.setitem(sys.modules, pla.__name__, pla)
    monkeypatch.setenv("VLLM_SAIL_USE_PLA", "1")

    class Tensor:
        shape = (1, 5, num_heads, 128)

        def view(self, *shape):
            return self

    class Bias:
        shape = (num_heads * 128,)

        def view(self, *shape):
            return SimpleNamespace(shape=shape)

    namespace = {
        "current_platform": SimpleNamespace(is_ppu=lambda: True),
        "logger": SimpleNamespace(info_once=lambda *args: None),
    }
    fn = function(
        "vllm_sail/patch/enhancement/attention/kda.py", "_flashkda_prefill", namespace
    )
    tensors = [Tensor() for _ in range(12)]
    q, k, v, g, beta, a, initial, seq, out, final, workspace, checkpoint = tensors
    bias = Bias()
    result = fn(q, k, v, g, beta, a, bias, -5.0, initial, seq, out, final, workspace)
    assert result == (out, final)
    assert calls[0]["out"] is out and calls[0]["final_state"] is final
    assert calls[0]["dt_bias"].shape == (num_heads, 128)
    assert bias.shape == (num_heads * 128,)
    with pytest.raises(NotImplementedError, match="checkpoint"):
        fn(
            q,
            k,
            v,
            g,
            beta,
            a,
            bias,
            -5.0,
            initial,
            seq,
            out,
            final,
            workspace,
            checkpoint_state=checkpoint,
        )
    assert len(calls) == 1


@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_ppu_kda_cannot_select_nvidia_kernel_when_pla_disabled(monkeypatch, mode):
    sdk = ModuleType("vllm_sail.attention.pla_kda")
    sdk.get_pla_kda_kernel = lambda mode: None
    monkeypatch.setitem(sys.modules, sdk.__name__, sdk)
    namespace = {
        "current_platform": SimpleNamespace(
            is_ppu=lambda: True,
            is_cuda=lambda: True,
            get_device_capability=lambda: SimpleNamespace(major=10),
            is_device_capability=lambda cap: True,
            is_device_capability_family=lambda cap: True,
        ),
        "torch": SimpleNamespace(bfloat16="bf16", float32="fp32"),
        "is_conv_state_dim_first": lambda: False,
    }
    fn = function(
        "vllm_sail/patch/enhancement/attention/kda.py",
        "is_flashkda_supported"
        if mode == "prefill"
        else "is_fused_kda_decode_supported",
        namespace,
    )
    args = (
        (128, "bf16", "fp32", -5.0)
        if mode == "prefill"
        else (12, 128, 4, 0, "bf16", "bf16", "fp32")
    )
    assert fn(*args) is False


@pytest.mark.parametrize(
    "name", ["is_flashkda_supported", "is_fused_kda_decode_supported"]
)
def test_device_predicate_calls_bind_all_required_arguments(name):
    fn = function("vllm_sail/patch/enhancement/attention/kda.py", name, {})
    path = Path(__file__).parents[1] / "e2e/fork_port/test_kda.py"
    calls = [
        node
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    ]
    assert calls
    for call in calls:
        inspect.signature(fn).bind(
            *call.args, **{kw.arg: kw.value for kw in call.keywords}
        )


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "local,local_name,upstream,upstream_name",
    [
        (
            "attention/kda",
            "_init",
            "models/kimi_k3/nvidia/kda",
            "KimiK3DeltaAttention.__init__",
        ),
        (
            "attention/kda",
            "_flashkda_prefill",
            "models/kimi_k3/nvidia/kda",
            "_flashkda_prefill",
        ),
        (
            "attention/kda",
            "is_flashkda_supported",
            "models/kimi_k3/nvidia/kda",
            "is_flashkda_supported",
        ),
        (
            "attention/kda",
            "is_fused_kda_decode_supported",
            "models/kimi_k3/nvidia/kda",
            "is_fused_kda_decode_supported",
        ),
    ],
)
def test_replacements_accept_upstream_keywords(
    upstream_source_root, local, local_name, upstream, upstream_name
):
    assert_accepts_upstream_keywords(
        local, local_name, upstream_source_root, upstream, upstream_name
    )
