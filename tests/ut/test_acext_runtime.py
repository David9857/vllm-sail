# SPDX-License-Identifier: Apache-2.0
"""Test ACEXT dynamic-link compatibility without Torch or a device."""

from __future__ import annotations

import ast
import builtins
import ctypes
import importlib
import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture(autouse=True)
def clear_loader_cache():
    yield
    runtime = sys.modules.get("vllm_sail.utils.acext")
    if runtime is not None:
        runtime._load_sdk_cublas.cache_clear()


@pytest.mark.parametrize(
    "consumer",
    [
        "model_executor/layers/fused_moe/experts/acext.py",
        "model_executor/kernels/linear/scaled_mm/ppu.py",
        "model_executor/layers/quantization/mixed_precision_w4.py",
    ],
)
def test_acext_consumers_prepare_runtime_before_symbol_import(consumer):
    root = Path(__file__).resolve().parents[2] / "vllm_sail"
    tree = ast.parse((root / consumer).read_text())
    imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "acext"
    ]
    assert imports
    for node in imports:
        blocks = [
            value
            for parent in ast.walk(tree)
            for _, value in ast.iter_fields(parent)
            if isinstance(value, list) and node in value
        ]
        assert any(
            isinstance(previous, ast.Expr)
            and isinstance(previous.value, ast.Call)
            and isinstance(previous.value.func, ast.Name)
            and previous.value.func.id == "load_acext"
            for block in blocks
            for previous in block[: block.index(node)]
        ), f"{consumer} must load the runtime before importing SDK symbols"


def _runtime():
    name = "vllm_sail.utils.acext"
    assert importlib.util.find_spec(name) is not None, (
        "ACEXT compatibility loader is missing"
    )
    return importlib.reload(importlib.import_module(name))


def _missing_cublas(*, wrapped: bool) -> OSError:
    cause = OSError("libth_op.so: undefined symbol: cublasGemmEx")
    if not wrapped:
        return cause
    error = OSError("Could not load this library: libth_op.so")
    error.__cause__ = cause
    return error


def _intercept_import(monkeypatch, import_acext):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "acext":
            return import_acext()
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


def test_healthy_acext_does_not_load_sdk_library(monkeypatch) -> None:
    runtime = _runtime()
    backend = ModuleType("acext")
    _intercept_import(monkeypatch, lambda: backend)
    monkeypatch.setattr(
        ctypes, "CDLL", lambda *a, **k: pytest.fail("preloading must not occur")
    )
    assert runtime.load_acext() is backend


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("custom_sdk", [False, True])
def test_missing_cublas_loads_only_sdk_library_then_retries(
    monkeypatch, tmp_path, wrapped, custom_sdk
) -> None:
    runtime = _runtime()
    monkeypatch.delenv("PPU_SDK", raising=False)
    sdk = Path("/usr/local/PPU_SDK")
    if custom_sdk:
        sdk = tmp_path / "sdk"
        monkeypatch.setenv("PPU_SDK", str(sdk))
    backend = ModuleType("acext")
    attempts = []
    handles = []

    def import_acext():
        attempts.append("import")
        if not handles:
            raise _missing_cublas(wrapped=wrapped)
        return backend

    def load(path, *, mode):
        assert path == str(sdk / "CUDA_SDK/lib64/libcublas.so")
        assert mode == ctypes.RTLD_GLOBAL
        attempts.append("load")
        handle = object()
        handles.append(handle)
        return handle

    _intercept_import(monkeypatch, import_acext)
    monkeypatch.setattr(ctypes, "CDLL", load)
    assert runtime.load_acext() is backend
    assert attempts == ["import", "load", "import"]
    assert runtime.load_acext() is backend
    assert attempts == ["import", "load", "import", "import"]
    assert runtime._load_sdk_cublas(str(sdk)) is handles[0]
    assert len(handles) == 1


@pytest.mark.parametrize(
    "error",
    [
        OSError("libth_op.so: undefined symbol: other_symbol"),
        OSError("libth_op.so: undefined symbol: cublasGemmExOther"),
        ImportError("acext ABI failure"),
        ModuleNotFoundError("acext missing", name="acext"),
        ModuleNotFoundError("dependency missing", name="other_dependency"),
    ],
)
def test_unrelated_loader_errors_propagate_without_preloading(monkeypatch, error):
    runtime = _runtime()

    def import_acext():
        raise error

    _intercept_import(monkeypatch, import_acext)
    monkeypatch.setattr(
        ctypes, "CDLL", lambda *a, **k: pytest.fail("preloading must not occur")
    )
    with pytest.raises(type(error)) as caught:
        runtime.load_acext()
    assert caught.value is error


def test_sdk_library_failure_is_not_hidden(monkeypatch):
    runtime = _runtime()
    original = _missing_cublas(wrapped=True)
    failure = OSError("SDK cuBLAS unavailable")

    def import_acext():
        raise original

    def load(*args, **kwargs):
        raise failure

    _intercept_import(monkeypatch, import_acext)
    monkeypatch.setattr(ctypes, "CDLL", load)
    with pytest.raises(OSError) as caught:
        runtime.load_acext()
    assert caught.value is failure


def test_retry_failure_is_not_hidden_or_retried_again(monkeypatch):
    runtime = _runtime()
    attempts = []
    failure = _missing_cublas(wrapped=True)

    def import_acext():
        attempts.append("import")
        raise failure

    def load(*args, **kwargs):
        attempts.append("load")
        return object()

    _intercept_import(monkeypatch, import_acext)
    monkeypatch.setattr(ctypes, "CDLL", load)
    with pytest.raises(OSError) as caught:
        runtime.load_acext()
    assert caught.value is failure
    assert attempts == ["import", "load", "import"]
