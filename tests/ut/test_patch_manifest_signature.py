# SPDX-License-Identifier: Apache-2.0
"""Structural and signature integrity of the runtime patch manifest.

``vllm_sail/patch/manifest.md`` is generated from live patch metadata and is the
reviewable inventory of every symbol vllm-sail hooks into upstream. This module
is the CPU-only gate on that inventory's *shape* and on each entry's *locatability*
in the pinned upstream tree. It is deliberately complementary to the neighbours:

* ``test_kernel_manifest.py`` audits the native-kernel ``manifest.toml``.
* ``test_upstream_version_precheck.py`` detects signature *drift* against the lock.
* ``test_upstream_interface_compat.py`` checks each replacement still *fits*.
* This file checks the manifest itself is complete, consistent with the lock, and
  that every declared target and every module a patch body imports actually
  resolves in upstream source -- so a renamed or relocated upstream symbol fails
  here with the affected patch named, instead of as an import-time crash.

The metadata/shape tests run anywhere; the resolution tests take the
``upstream_source_root`` fixture (skip without ``VLLM_SOURCE_ROOT``, fail early
under ``--require-upstream-source``).
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from tests.ut.test_upstream_version_precheck import (
    MANIFEST_KINDS,
    MANIFEST_PATH,
    PATCH_ROOT,
    SKIP_DIRS,
    SUPPORTED_RANGE,
    ManifestEntry,
    load_lock,
    module_file,
    parse_manifest,
    resolve_symbol,
)


def _iter_patch_bodies() -> list[Path]:
    return sorted(
        path for path in PATCH_ROOT.rglob("*.py") if not (SKIP_DIRS & set(path.parts))
    )


def _version_bounds(spec: str) -> tuple[str, str]:
    """Split ``">=0.31.0,<0.32.0"`` into its low/high version strings."""
    low = high = ""
    for clause in spec.split(","):
        clause = clause.strip()
        if clause.startswith(">="):
            low = clause[2:].strip()
        elif clause.startswith("<"):
            high = clause[1:].strip()
    return low, high


@pytest.fixture(scope="module")
def manifest() -> list[ManifestEntry]:
    return parse_manifest()


@pytest.fixture(scope="module")
def lock() -> dict[str, Any]:
    return load_lock()


# ---------------------------------------------------------------------------
# CPU-only metadata / shape tests (run everywhere)
# ---------------------------------------------------------------------------
def test_manifest_is_present_and_populated(manifest: list[ManifestEntry]) -> None:
    assert MANIFEST_PATH.is_file(), "patch manifest.md is missing"
    assert manifest, "patch manifest parsed to zero entries"


def test_every_entry_has_complete_metadata(manifest: list[ManifestEntry]) -> None:
    """No entry may ship with a blank required field.

    ``target`` / ``kind`` / ``reason`` / ``affected_versions`` / ``remove_when``
    are all mandatory: the framework validates them at install time, and the
    manifest is the reviewable proof they are meaningful, not placeholders.
    """
    incomplete: list[str] = []
    for entry in manifest:
        for field in ("target", "kind", "reason", "affected_versions", "remove_when"):
            if not getattr(entry, field).strip():
                incomplete.append(f"{entry.target or '<no target>'}: blank {field}")
        if entry.kind not in MANIFEST_KINDS:
            incomplete.append(f"{entry.target}: invalid kind {entry.kind!r}")
    assert not incomplete, "manifest entries with incomplete metadata:\n" + "\n".join(
        incomplete
    )


def test_targets_are_unique_and_sorted(manifest: list[ManifestEntry]) -> None:
    """The generator sorts by target and keys patches by target; both must hold.

    A duplicate target means two patches fight over one symbol; unsorted order
    means the committed file was hand-edited (AGENTS.md forbids that).
    """
    targets = [entry.target for entry in manifest]
    duplicates = {t for t in targets if targets.count(t) > 1}
    assert not duplicates, f"duplicate patch targets: {sorted(duplicates)}"
    assert targets == sorted(targets), "manifest.md is not sorted by target"


def test_affected_versions_are_a_supported_range(
    manifest: list[ManifestEntry],
) -> None:
    """Each patch must declare the pinned release line it applies to."""
    low_bound, high_bound = _version_bounds(SUPPORTED_RANGE)
    bad: list[str] = []
    for entry in manifest:
        low, high = _version_bounds(entry.affected_versions)
        if not low or not high:
            bad.append(f"{entry.target}: {entry.affected_versions!r} is not a range")
        elif (low, high) != (low_bound, high_bound):
            bad.append(
                f"{entry.target}: range {entry.affected_versions!r} != pinned "
                f"{SUPPORTED_RANGE!r}"
            )
    assert not bad, "patches declaring an unexpected version range:\n" + "\n".join(bad)


def test_remove_when_is_a_verifiable_condition(
    manifest: list[ManifestEntry],
) -> None:
    """``remove_when`` must be actionable, never a TODO placeholder."""
    placeholders = [
        entry.target
        for entry in manifest
        if entry.remove_when.strip().lower() in {"todo", "tbd", "n/a", "unknown", ""}
    ]
    assert not placeholders, (
        f"patches with a non-verifiable remove_when: {placeholders[:10]}"
    )


def test_manifest_and_lock_agree_on_kind(
    manifest: list[ManifestEntry], lock: dict[str, Any]
) -> None:
    """The manifest and the snapshot must describe the same targets identically.

    This is the "no orphan declaration / no orphan body" guarantee: every
    manifest entry has exactly one lock entry with a matching patch kind, so the
    pre-check cannot be reasoning about a different inventory than the docs.
    """
    locked = {entry["name"]: entry for entry in lock["patch_target"]}
    declared = {entry.target: entry for entry in manifest}
    assert set(locked) == set(declared), (
        "manifest and interface_lock.toml disagree on the target set: "
        f"only-lock={sorted(set(locked) - set(declared))[:5]} "
        f"only-manifest={sorted(set(declared) - set(locked))[:5]}"
    )
    mismatched = [
        target
        for target, entry in declared.items()
        if locked[target]["kind"] != entry.kind
    ]
    assert not mismatched, f"kind mismatch between manifest and lock: {mismatched[:10]}"


# ---------------------------------------------------------------------------
# Upstream-dependent resolution tests
# ---------------------------------------------------------------------------
@pytest.mark.upstream_source
def test_target_modules_resolve_upstream(
    upstream_source_root: Path, manifest: list[ManifestEntry]
) -> None:
    """Every patched target's module must exist as a file in upstream source.

    A module that no longer resolves means the patch is aimed at a path upstream
    moved or deleted -- the loudest possible signal that the pin is broken.
    """
    unresolved = [
        entry.target
        for entry in manifest
        if module_file(upstream_source_root, entry.target)[0] is None
    ]
    assert not unresolved, (
        f"{len(unresolved)} patch target(s) point at modules missing upstream:\n"
        + "\n".join(unresolved[:20])
    )


@pytest.mark.upstream_source
def test_present_symbols_exist_in_upstream_ast(
    upstream_source_root: Path, lock: dict[str, Any]
) -> None:
    """Every non-additive target must resolve to a symbol in the upstream AST.

    Additive patches (``present = false`` in the lock) inject a new attribute and
    are expected to be absent upstream; if one is now present that is drift the
    pre-check reports, not a manifest defect, so it is not re-asserted here.
    """
    absent: list[str] = []
    checked = 0
    for entry in lock["patch_target"]:
        if not entry["present"]:
            continue
        checked += 1
        if not resolve_symbol(upstream_source_root, entry["name"]).present:
            absent.append(entry["name"])
    assert checked, "no present targets to verify; is the lock populated?"
    assert not absent, (
        f"{len(absent)} patch target(s) recorded as present no longer resolve in "
        "the upstream AST:\n" + "\n".join(absent[:20])
    )


@pytest.mark.upstream_source
def test_patch_body_upstream_imports_resolve(upstream_source_root: Path) -> None:
    """Every ``vllm.*`` module a patch body imports must exist upstream.

    Patch bodies are copied upstream code; a body that imports a module upstream
    renamed would raise ``ModuleNotFoundError`` the moment its category loads.
    Resolving at the module granularity is robust (symbol-level imports are
    frequently conditional or re-exported) while still catching relocated code.
    """
    imported: set[str] = set()
    for path in _iter_patch_bodies():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module
                and node.module.startswith("vllm.")
            ):
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("vllm."):
                        imported.add(alias.name)
    assert imported, "patch bodies import no vllm modules; the scan regressed"
    unresolved = sorted(
        module
        for module in imported
        if module_file(upstream_source_root, module)[0] is None
    )
    assert not unresolved, (
        f"patch bodies import {len(unresolved)} vllm module(s) that no longer "
        f"resolve upstream: {unresolved[:20]}"
    )
