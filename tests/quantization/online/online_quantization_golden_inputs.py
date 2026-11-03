# SPDX-License-Identifier: Apache-2.0
"""Small, deterministic source tensors for the qlean parity golden.

The inputs are constructed from integers rather than a PRNG or a serialized
checkpoint.  Each case represents one logical projection; gate and up are
separate because their FP8 block-scale grids must be decoded independently.
"""

import hashlib

import torch


def tensor_digest(tensor: torch.Tensor) -> dict:
    """Hash the complete logical tensor, independent of storage strides/device."""
    value = tensor.detach().to("cpu").contiguous()
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "sha256": hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest(),
    }


def cases():
    """Yield (name, source dtype, weight, FP8 scale grid or None)."""
    # Includes zero rows, tiny values, negative values, and rounding ties.
    dense_bf16 = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [127, -127, 0.5, 1.5, -1.5, 2.5, -2.5],
            [0.0078125, -0.0078125, 1, -1, 3, -3, 0],
        ],
        dtype=torch.bfloat16,
    )
    yield "dense.bf16", "bf16", dense_bf16, None

    # 129x129 crosses both 128-element FP8 block boundaries.
    dense_fp8 = torch.tensor(
        [[(row * 7 + col * 3) % 15 - 7 for col in range(129)] for row in range(129)],
        dtype=torch.bfloat16,
    ).to(torch.float8_e4m3fn)
    yield (
        "dense.fp8",
        "fp8",
        dense_fp8,
        torch.tensor([[0.125, 0.25], [0.5, 1.0]], dtype=torch.float32),
    )

    # Two experts, with separate gate/up scale grids and an unaligned tail.
    for expert in range(2):
        for shard, scale in (("gate", 0.125), ("up", 0.5)):
            weight_bf16 = torch.tensor(
                [
                    [((row * 5 + col * 2 + expert * 3) % 13) - 6 for col in range(129)]
                    for row in range(65)
                ],
                dtype=torch.bfloat16,
            )
            yield f"moe.expert{expert}.{shard}.bf16", "bf16", weight_bf16, None
            yield (
                f"moe.expert{expert}.{shard}.fp8",
                "fp8",
                weight_bf16.to(torch.float8_e4m3fn),
                torch.tensor([[scale, scale * 2]], dtype=torch.float32),
            )
        down = torch.tensor(
            [
                [((row * 3 + col * 7 + expert) % 17) - 8 for col in range(65)]
                for row in range(129)
            ],
            dtype=torch.bfloat16,
        )
        yield f"moe.expert{expert}.down.bf16", "bf16", down, None
        yield (
            f"moe.expert{expert}.down.fp8",
            "fp8",
            down.to(torch.float8_e4m3fn),
            torch.tensor([[0.25], [0.5]], dtype=torch.float32),
        )
