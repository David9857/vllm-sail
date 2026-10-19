# SPDX-License-Identifier: Apache-2.0
"""Device gates for loading-time FP8 conversion and existing Dense backends."""

import pytest

pytestmark = pytest.mark.ppu


@pytest.fixture
def runtime(monkeypatch):
    torch = pytest.importorskip("torch")
    import vllm_sail

    vllm_sail.register_out_of_tree()
    from vllm.platforms import current_platform

    if not current_platform.is_ppu():
        pytest.skip("requires PPU")
    # Standalone, unsharded layer test; no distributed process group is needed.
    import vllm.model_executor.parameter as parameter

    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 1)
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        yield torch


@pytest.mark.parametrize("target", ["bf16", "int8"])
@pytest.mark.parametrize("bias", [False, True])
def test_dense_conversion_forward_and_idempotency(runtime, target, bias):
    torch = runtime
    from vllm_sail.model_executor.layers.quantization.online_conversion.linear import (
        Fp8ConversionLinearMethod,
    )
    from vllm_sail.model_executor.layers.quantization.online_conversion.quant_utils import (
        dequantize_fp8,
        quantize_int8,
    )

    torch.manual_seed(23)
    with torch.device("cuda"):
        layer = torch.nn.Module()
        method = Fp8ConversionLinearMethod(target)
        method.create_weights(layer, 256, [128, 128], 256, 256, torch.bfloat16)
        layer.weight.data.copy_(torch.randn(256, 256).to(torch.float8_e4m3fn))
        layer.weight_scale_inv.data.fill_(0.03125)
        decoded = dequantize_fp8(layer.weight, layer.weight_scale_inv)
        expected_q, expected_scale = quantize_int8(decoded)
        method.process_weights_after_loading(layer)
        assert layer.weight_scale_inv is None
        pointer = layer.weight.data_ptr()
        method.process_weights_after_loading(layer)
        assert layer.weight.data_ptr() == pointer
        if target == "int8":
            expected_q = expected_q if method.kernel.weight_RowMajor else expected_q.t()
            assert torch.equal(layer.weight, expected_q)
            assert torch.equal(layer.weight_scale, expected_scale)
            reference_weight = (
                expected_q if method.kernel.weight_RowMajor else expected_q.t()
            ).float() * expected_scale
        else:
            assert torch.equal(layer.weight, decoded)
            reference_weight = decoded.float()
        x = torch.randn(2, 7, 256, dtype=torch.bfloat16)
        b = torch.randn(256, dtype=torch.bfloat16) if bias else None
        actual = method.apply(layer, x, b)
        # W8A8 adds dynamic activation quantization; compare against decoded
        # weight matmul with a bounded relative norm error, not exact outputs.
        expected = x.float() @ reference_weight.t()
        if b is not None:
            expected += b.float()
        assert actual.shape == expected.shape
        assert torch.isfinite(actual).all()
        relative = (actual.float() - expected).norm() / expected.norm()
        assert relative.item() < (0.025 if target == "int8" else 0.01)
