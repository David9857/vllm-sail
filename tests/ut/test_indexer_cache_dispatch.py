# SPDX-License-Identifier: Apache-2.0
"""Run the indexer dispatch body without torch, vLLM or a device.

Device tests cover the imported module, cache bytes and real consumers.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from tests.support.source import function


@pytest.mark.parametrize("q_dtype", ["int8", "float8_e4m3fn"])
@pytest.mark.parametrize("skip_insert", [False, True])
def test_writer_dispatch_and_speculative_padding(q_dtype, skip_insert):
    layer = "model.layers.0.self_attn.indexer.k_cache"
    slots = SimpleNamespace(shape=(3,))
    metadata = SimpleNamespace(
        slot_mapping=slots,
        num_decodes=0,
        num_prefills=0,
        num_decode_tokens=0,
    )
    int8_writer, fp8_writer = Mock(), Mock()
    namespace = {
        "torch": SimpleNamespace(int8="int8"),
        "get_forward_context": lambda: SimpleNamespace(attn_metadata={layer: metadata}),
        "_resolve_layer_name": lambda name: name,
        "DeepseekV32IndexerMetadata": SimpleNamespace,
        "indexer_k_quant_and_cache_int8": int8_writer,
        "ops": SimpleNamespace(indexer_k_quant_and_cache=fp8_writer),
    }
    indexer = function(
        "vllm_sail/attention/ops/mla_sparse.py", "ppu_sparse_attn_indexer", namespace
    )
    k, topk = MagicMock(), MagicMock()
    cache, weights = object(), object()
    result = indexer(
        SimpleNamespace(shape=(5, 1)),
        layer,
        cache,
        SimpleNamespace(dtype=q_dtype),
        None,
        k,
        weights,
        128,
        "ue8m0",
        2,
        128,
        32,
        32,
        topk,
        skip_insert,
    )

    assert result is topk
    k.__getitem__.assert_called_once_with(slice(None, 3))
    topk.__setitem__.assert_called_once_with(slice(None, 5), -1)
    if skip_insert:
        int8_writer.assert_not_called()
        fp8_writer.assert_not_called()
    elif q_dtype == "int8":
        int8_writer.assert_called_once_with(
            k.__getitem__.return_value, cache, slots, 128
        )
        fp8_writer.assert_not_called()
    else:
        fp8_writer.assert_called_once_with(
            k.__getitem__.return_value, cache, slots, 128, "ue8m0"
        )
        int8_writer.assert_not_called()
