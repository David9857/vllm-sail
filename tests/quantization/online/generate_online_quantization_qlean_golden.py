# SPDX-License-Identifier: Apache-2.0
"""Regenerate the small offline oracle on a machine with qlean and PPU.

Run explicitly; this script is not part of CI.  Inspect the JSON diff before
updating the checked-in reference.
"""

import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch
from qlean.modifiers.dequant.base import weight_dequant
from qlean.modifiers.utils import int8_channel_weight_quant

from tests.quantization.online.online_quantization_golden_inputs import (
    cases,
    tensor_digest,
)


def main():
    assert torch.cuda.is_available(), "qlean FP8 decoding requires a device"
    assert importlib.metadata.version("qlean") == "0.2.0+v0.1.0.ppu2.1.0"
    tensors = {}
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        for name, source, weight, scales in cases():
            weight = weight.cuda()
            decoded = (
                weight_dequant(weight.float().contiguous(), scales.cuda().contiguous())
                if source == "fp8"
                else weight
            )
            if source == "fp8":
                tensors[f"{name}.fp8_to_bf16.weight"] = tensor_digest(decoded)
            q, s = int8_channel_weight_quant(decoded)
            route = "fp8_to_int8" if source == "fp8" else "bf16_to_int8"
            tensors[f"{name}.{route}.weight"] = tensor_digest(q)
            tensors[f"{name}.{route}.weight_scale"] = tensor_digest(s)
    finally:
        torch.set_default_dtype(old_dtype)
    print(
        json.dumps(
            {
                "schema_version": 1,
                "qlean_version": importlib.metadata.version("qlean"),
                "torch_version": torch.__version__,
                "device": torch.cuda.get_device_name(),
                "input_recipe": "tests/quantization/online/online_quantization_golden_inputs.py",
                "input_recipe_sha256": hashlib.sha256(
                    Path(__file__)
                    .with_name("online_quantization_golden_inputs.py")
                    .read_bytes()
                ).hexdigest(),
                "tensors": tensors,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
