# SPDX-License-Identifier: Apache-2.0
"""Signature-level compatibility between vllm-sail patches and upstream vLLM.

Where :mod:`tests.ut.test_upstream_version_precheck` asks "did the pinned
upstream drift from our snapshot?", this module asks the complementary question:
"does each patch we ship still *fit* the upstream symbol it replaces?" A patch
whose replacement drops a keyword, a ``*args`` or a ``**kwargs`` that upstream
callers pass will install cleanly and then blow up at call time -- the silent
failure the whole category exists to catch.

The checks are static (AST only), so they never import torch or vLLM:

* :func:`scan_static_replacements` recovers every ``@patch("module", "attr")``
  whose target is a literal string, together with the replacement body. Targets
  applied programmatically in a loop (e.g. the MoE consumer aliases) are covered
  by the hookup-point and lock checks instead, since their target strings are not
  statically knowable.
* For each non-additive replacement, the upstream target's signature is compared
  against the replacement's, mirroring
  :func:`tests.support.source.assert_accepts_upstream_keywords` but generalized to
  any patch category and to class-method targets, and emitting a parameter-level
  diff on failure.
* Every patch hookup point recorded in ``interface_lock.toml`` as present must
  still resolve upstream, grouped by the three hookup kinds (module attribute /
  class attribute / value injection).
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, NamedTuple

import pytest

from tests.ut.test_upstream_version_precheck import (
    DEF_NODES,
    FUNC_NODES,
    MANIFEST_KINDS,
    PATCH_ROOT,
    SKIP_DIRS,
    ResolvedSymbol,
    load_lock,
    parse_manifest,
    resolve_symbol,
)


class PatchReplacement(NamedTuple):
    """A statically-declared ``@patch`` replacement and its upstream target."""

    target: str
    """Fully-qualified upstream target, e.g. ``vllm.foo.Bar.baz``."""
    hook: str
    """``"class"`` when the attribute path is dotted, else ``"module"``."""
    additive: bool
    """True when declared with ``allow_missing=True``."""
    node: ast.AST
    """The replacement function/class AST node."""
    path: Path
    """vllm-sail source file the replacement lives in."""


def _iter_patch_modules() -> list[Path]:
    return sorted(
        path for path in PATCH_ROOT.rglob("*.py") if not (SKIP_DIRS & set(path.parts))
    )


def _literal(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def scan_static_replacements() -> list[PatchReplacement]:
    """Recover every ``@patch(<str>, <str>, ...)``-decorated replacement."""
    found: list[PatchReplacement] = []
    for path in _iter_patch_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, DEF_NODES):
                continue
            for decorator in node.decorator_list:
                if not (
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Name)
                    and decorator.func.id == "patch"
                ):
                    continue
                args = decorator.args
                if len(args) < 2:
                    continue
                module = _literal(args[0])
                attribute = _literal(args[1])
                if module is None or attribute is None:
                    continue
                additive = any(
                    keyword.arg == "allow_missing"
                    and isinstance(keyword.value, ast.Constant)
                    and bool(keyword.value.value)
                    for keyword in decorator.keywords
                )
                hook = "class" if "." in attribute else "module"
                found.append(
                    PatchReplacement(
                        target=f"{module}.{attribute}",
                        hook=hook,
                        additive=additive,
                        node=node,
                        path=path,
                    )
                )
    return found


def _callable_params(
    node: ast.AST,
) -> tuple[list[str], list[str], bool, bool]:
    """Positional names, keyword-only names, has ``*args``, has ``**kwargs``."""
    assert isinstance(node, FUNC_NODES)
    arguments = node.args
    positional = [arg.arg for arg in arguments.posonlyargs + arguments.args]
    keyword_only = [arg.arg for arg in arguments.kwonlyargs]
    return (
        positional,
        keyword_only,
        arguments.vararg is not None,
        arguments.kwarg is not None,
    )


def _drop_receiver(names: list[str]) -> list[str]:
    """Remove a leading ``self``/``cls`` so methods compare with functions."""
    if names and names[0] in ("self", "cls"):
        return names[1:]
    return names


def signature_diff(replacement: ast.AST, upstream: ast.AST) -> list[str]:
    """Parameter-level differences explaining why a replacement may not fit.

    Reports parameters the replacement fails to accept, extras it invents, and
    positional-order changes for the parameters they share.
    """
    rep_pos, rep_kw, rep_var, rep_kwarg = _callable_params(replacement)
    up_pos, up_kw, up_var, up_kwarg = _callable_params(upstream)
    rep_names = _drop_receiver(rep_pos)
    up_names = _drop_receiver(up_pos)
    rep_set = set(rep_names) | set(rep_kw)
    up_set = set(up_names) | set(up_kw)

    diffs: list[str] = []
    missing = [name for name in up_names if name not in rep_set]
    missing += [name for name in up_kw if name not in rep_set]
    if missing and not rep_kwarg:
        diffs.append(f"replacement cannot accept upstream keyword(s): {missing}")
    extra = [name for name in rep_names if name not in up_set]
    extra += [name for name in rep_kw if name not in up_set]
    if extra:
        diffs.append(f"replacement adds parameter(s) upstream lacks: {extra}")

    shared_up = [name for name in up_names if name in set(rep_names)]
    shared_rep = [name for name in rep_names if name in set(up_names)]
    if shared_up != shared_rep:
        diffs.append(
            f"positional order differs: upstream {shared_up} vs "
            f"replacement {shared_rep}"
        )
    if up_var and not rep_var:
        diffs.append("replacement drops upstream *args")
    if up_kwarg and not rep_kwarg:
        diffs.append("replacement drops upstream **kwargs")
    return diffs


def compatibility_problems(replacement: ast.AST, upstream: ast.AST) -> list[str]:
    """Hard incompatibilities that would break a caller of the patched symbol.

    A replacement carrying ``**kwargs`` absorbs any upstream keyword, so only a
    genuinely unaccepted keyword (without ``**kwargs``) or a dropped variadic is
    treated as breaking; the rest of :func:`signature_diff` is advisory.
    """
    _, _, _, rep_kwarg = _callable_params(replacement)
    _, _, up_var, up_kwarg = _callable_params(upstream)
    rep_pos, rep_kw, rep_var, _ = _callable_params(replacement)
    up_pos, up_kw, _, _ = _callable_params(upstream)
    accepted = set(_drop_receiver(rep_pos)) | set(rep_kw)
    offered = set(_drop_receiver(up_pos)) | set(up_kw)

    problems: list[str] = []
    if not rep_kwarg and not offered <= accepted:
        problems.append(f"missing upstream keyword(s): {sorted(offered - accepted)}")
    if up_var and not rep_var:
        problems.append("missing *args (upstream callers may pass positionally)")
    if up_kwarg and not rep_kwarg:
        problems.append("missing **kwargs (upstream accepts arbitrary keywords)")
    return problems


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def replacements() -> list[PatchReplacement]:
    return scan_static_replacements()


@pytest.fixture(scope="module")
def lock() -> dict[str, Any]:
    return load_lock()


# ---------------------------------------------------------------------------
# CPU-only tests (run everywhere)
# ---------------------------------------------------------------------------
def test_static_replacements_are_well_formed(
    replacements: list[PatchReplacement],
) -> None:
    """The scanner must recover real, namespaced, categorized patch targets."""
    assert replacements, "no @patch replacements found; the scanner regressed"
    for replacement in replacements:
        assert replacement.target.startswith("vllm."), replacement.target
        module, _, attribute = replacement.target.rpartition(".")
        assert module and attribute, replacement.target
        assert replacement.hook in {"module", "class"}
        assert replacement.path.is_file()
        assert isinstance(replacement.node, DEF_NODES)
    # Both hookup styles and the additive flag are exercised by the corpus.
    assert {replacement.hook for replacement in replacements} >= {"module", "class"}
    assert any(replacement.additive for replacement in replacements)


def test_signature_diff_reports_parameter_changes() -> None:
    """Unit-test the diff helper so its diagnostics are trustworthy."""

    def _func(source: str) -> ast.AST:
        node = ast.parse(source).body[0]
        assert isinstance(node, ast.FunctionDef)
        return node

    upstream = _func("def f(self, a, b, *, c=1, **kw): ...")
    good = _func("def f(self, a, b, *, c=1, **kw): ...")
    assert signature_diff(good, upstream) == []
    assert compatibility_problems(good, upstream) == []

    dropped = _func("def f(self, a, *, c=1): ...")
    problems = compatibility_problems(dropped, upstream)
    assert any("missing upstream keyword" in p for p in problems)
    assert any("**kwargs" in p for p in signature_diff(dropped, upstream))

    reordered = _func("def f(self, b, a, *, c=1, **kw): ...")
    assert any("positional order" in d for d in signature_diff(reordered, upstream))
    # A **kwargs sink still satisfies the hard-compatibility rule.
    assert compatibility_problems(reordered, upstream) == []


def test_manifest_targets_are_dotted_upstream_paths() -> None:
    """Every manifest target must be a resolvable dotted upstream path."""
    for entry in parse_manifest():
        assert entry.target.startswith("vllm."), entry.target
        assert entry.kind in MANIFEST_KINDS, entry.target
        assert entry.target.count(".") >= 2, entry.target


# ---------------------------------------------------------------------------
# Upstream-dependent tests (skip without VLLM_SOURCE_ROOT; fail early under
# --require-upstream-source)
# ---------------------------------------------------------------------------
@pytest.mark.upstream_source
def test_replacement_signatures_accept_upstream_keywords(
    upstream_source_root: Path, replacements: list[PatchReplacement]
) -> None:
    """Each replacement must still accept what its upstream target accepts.

    Additive patches (``allow_missing=True``) have no upstream counterpart and
    are skipped here; targets upstream does not define as a callable are also
    skipped, since a value/class replacement has no parameter contract to fit.
    """
    failures: list[str] = []
    checked = 0
    for replacement in replacements:
        if replacement.additive:
            continue
        resolved = resolve_symbol(upstream_source_root, replacement.target)
        if not isinstance(resolved.node, FUNC_NODES):
            continue
        checked += 1
        problems = compatibility_problems(replacement.node, resolved.node)
        if problems:
            detail = "; ".join(problems)
            diff = " | ".join(signature_diff(replacement.node, resolved.node))
            failures.append(
                f"{replacement.target} ({replacement.path.name}): {detail}"
                + (f" [diff: {diff}]" if diff else "")
            )
    assert checked, "no upstream targets resolved to callables; is the source pinned?"
    assert not failures, (
        "patch replacements no longer fit upstream signatures:\n" + "\n".join(failures)
    )


@pytest.mark.upstream_source
def test_additive_replacement_modules_resolve(
    upstream_source_root: Path, replacements: list[PatchReplacement]
) -> None:
    """An additive patch injects into a real module; that module must exist."""
    additive = [replacement for replacement in replacements if replacement.additive]
    assert additive, "expected additive patches (is_ppu / forward_ppu family)"
    for replacement in additive:
        resolved = resolve_symbol(upstream_source_root, replacement.target)
        assert resolved.module_found, (
            f"additive patch {replacement.target} targets module that no longer "
            f"exists upstream ({replacement.path.name})"
        )


@pytest.mark.upstream_source
def test_hookup_points_resolve_by_kind(
    upstream_source_root: Path, lock: dict[str, Any]
) -> None:
    """Every non-additive hookup point must resolve, grouped by patch kind.

    Covers the three installation styles the framework supports -- module
    attribute replacement, class attribute replacement/wrapper, and value
    injection -- and fails with the offending targets named per kind.
    """
    by_kind: dict[str, list[str]] = {kind: [] for kind in sorted(MANIFEST_KINDS)}
    missing: dict[str, list[str]] = {kind: [] for kind in sorted(MANIFEST_KINDS)}
    for entry in lock["patch_target"]:
        if not entry["present"]:
            continue  # additive patch: intentionally absent upstream
        kind = entry["kind"]
        resolved: ResolvedSymbol = resolve_symbol(upstream_source_root, entry["name"])
        by_kind[kind].append(entry["name"])
        if not resolved.present:
            missing[kind].append(entry["name"])

    total = sum(len(names) for names in by_kind.values())
    assert total, "no present patch targets in the lock to verify"
    # Each hookup kind is actually exercised by the patch corpus.
    assert all(by_kind.values()), {kind: len(names) for kind, names in by_kind.items()}
    empty = {kind: names for kind, names in missing.items() if names}
    assert not empty, f"patch hookup points no longer resolve upstream: {empty}"
