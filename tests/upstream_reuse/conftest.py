# SPDX-License-Identifier: Apache-2.0
"""Self-contained harness for the upstream-test reuse suite (CI category 5).

This directory runs *upstream* vLLM test files inside the vllm-sail runtime.
The conftest is deliberately autonomous:

* It locates the upstream checkout from ``VLLM_SOURCE_ROOT`` using its own
  logic (never importing ``tests.support.source``, whose AST helpers break on
  Python 3.9). When the variable is unset or points somewhere without
  ``vllm/__init__.py``, every item under this directory is skipped at
  collection time -- so a bare host reports "all skipped, 0 errors".
* It exposes the :class:`~tests.upstream_reuse._reuse_harness.UpstreamRunner`
  through the ``upstream_runner`` fixture, which additionally skips when no
  installed vLLM + torch runtime is present (upstream modules import both at
  module scope, so without them a subprocess run could only fail on import).
* A third skip path is per-entry rather than per-session: upstream's own
  ``tests/conftest.py`` imports test-only packages (``tblib`` at the pinned
  commit) that the vllm-sail runtime does not ship.  When one is missing every
  subprocess dies before collecting a test, and
  :func:`~tests.upstream_reuse._reuse_harness.conftest_import_error` turns that
  into a skip quoting pytest's verbatim error -- see its docstring for the two
  look-alikes it deliberately refuses to absorb.
* A fourth skip path is also per-entry and also about the runtime, not the
  manifest: many upstream nodes name a checkpoint (``facebook/opt-125m``,
  ``llava-hf/llava-1.5-7b-hf``, ...) that ``vllm.config.model.ModelConfig``
  resolves through the Hugging Face hub.  On a runner with no route to the hub
  and no local copy that is a *hang* inside ``socket.create_connection``, not a
  failure, so
  :func:`~tests.upstream_reuse._reuse_harness.offline_hub_gap` probes the hub
  once per session and skips only the entries that name a checkpoint the local
  cache cannot satisfy.  Where the hub is reachable -- CI -- the probe succeeds
  and the guard is inert, so no coverage is given up.

The PPU-disguises-CUDA background: vllm-sail's ``PPUPlatform._enum`` is
``PlatformEnum.CUDA`` and ``current_platform.is_cuda()`` returns True. In CI the
runner has vllm-sail installed, so the reuse subprocess resolves
``current_platform`` to the PPU platform while upstream test bodies written
against CUDA execute unchanged -- the whole reason this whitelist can reuse
upstream tests instead of rewriting them.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.upstream_reuse._reuse_harness import (
    UpstreamRunner,
    resolve_source_root,
    runtime_available,
)

HERE = Path(__file__).resolve().parent


def _under_this_directory(item: pytest.Item) -> bool:
    """True when a collected item lives beneath ``tests/upstream_reuse``."""
    item_path = Path(str(item.fspath)).resolve()
    try:
        item_path.relative_to(HERE)
    except ValueError:
        return False
    return True


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip the whole directory when no usable upstream source is selected.

    ``tests/conftest.py`` also defines this hook (device/chip gating); pytest
    invokes both. We only touch items under this directory so the two compose.
    """
    root = resolve_source_root(os.environ.get("VLLM_SOURCE_ROOT"))
    if root is not None:
        return
    skip = pytest.mark.skip(
        reason="VLLM_SOURCE_ROOT not set to a valid upstream vLLM checkout"
    )
    for item in items:
        if _under_this_directory(item):
            item.add_marker(skip)


@pytest.fixture(scope="session")
def upstream_reuse_root(request: pytest.FixtureRequest) -> Path:
    """Resolved upstream checkout, or skip.

    Honours ``--require-upstream-source`` (declared in ``tests/conftest.py``):
    under that flag a missing/invalid root is a hard error, never a silent skip,
    matching the established upstream_source convention.
    """
    required = request.config.getoption("--require-upstream-source")
    root = resolve_source_root(os.environ.get("VLLM_SOURCE_ROOT"))
    if root is None:
        if required:
            raise pytest.UsageError(
                "--require-upstream-source requires VLLM_SOURCE_ROOT to point at "
                "an upstream vLLM checkout containing vllm/__init__.py"
            )
        pytest.skip("set VLLM_SOURCE_ROOT to run upstream reuse tests")
    return root


@pytest.fixture(scope="session")
def upstream_runner(upstream_reuse_root: Path) -> UpstreamRunner:
    """Runner bound to the upstream checkout, skipping without a vLLM runtime."""
    available, reason = runtime_available()
    if not available:
        pytest.skip(f"upstream reuse needs an installed runtime: {reason}")
    return UpstreamRunner(upstream_reuse_root)
