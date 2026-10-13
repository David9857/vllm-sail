# SPDX-License-Identifier: Apache-2.0
"""Resolve ACEXT's cuBLAS dependency using the SAIL SDK compatibility library."""

from __future__ import annotations

import ctypes
from functools import cache
from pathlib import Path
from types import ModuleType

from vllm_sail import envs


@cache
def _load_sdk_cublas(sdk: str) -> ctypes.CDLL:
    # Retain the handle and load only from the selected SDK, not system CUDA paths.
    library = Path(sdk) / "CUDA_SDK/lib64/libcublas.so"
    return ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)


def load_acext() -> ModuleType:
    """Resolve the known missing symbol; propagate other import and ABI errors."""
    try:
        import acext
    except OSError as exc:
        # Torch preserves the underlying dlopen failure in __cause__.
        cause = exc.__cause__ or exc
        if not isinstance(cause, OSError) or not str(cause).endswith(
            ": undefined symbol: cublasGemmEx"
        ):
            raise
        # hggcrt3 Torch links libacblas directly; legacy ACEXT omits cuBLAS linkage.
        # Once ACEXT declares that dependency, the initial import needs no fallback.
        _load_sdk_cublas(envs.PPU_SDK)
        import acext
    return acext
