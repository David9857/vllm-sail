# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for the PPU operator test-selection protocol, without importing device packages."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "scripts/ci/ppu_ops_tests.json"
ENVIRONMENT = ROOT / "scripts/ci/ppu_ops_environment.json"
SCRIPT = ROOT / "scripts/ci/ppu_ops_select.py"
OPTIONS = dict(diff_complete=True, force_full=False, environment_matches=True)


def test_selector_entrypoint_exists():
    assert SCRIPT.is_file(), "缺少设备选择器入口"


@pytest.fixture
def selector():
    assert SCRIPT.is_file(), "缺少设备选择器"
    spec = importlib.util.spec_from_file_location("ppu_ops_select", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def config():
    assert CONFIG.is_file(), "缺少正式测试目录"
    return json.loads(CONFIG.read_text())


def test_real_config_and_full_routes(selector, config):
    selector.validate_config(config, ROOT)
    result = selector.select_tests(config, ["csrc/unknown.cu"], **OPTIONS)
    assert result["mode"] == "full"
    assert "native_change" in result["reasons"]
    assert len(result["test_ids"]) == 15
    routes = selector.route_tests(config, result["test_ids"])
    assert [len(g["files"]) for g in routes["groups"]] == [13, 14]
    assert routes["matrix"]["include"] == [
        {"group_id": b + "-0", "board": b, "nproc_per_node": 1}
        for b in ("ppu10", "ppu15")
    ]
    assert selector.validate_environment(json.loads(ENVIRONMENT.read_text())) is False


@pytest.mark.parametrize(
    "paths,mode,ids",
    [
        (["README.md", "docs/nested/new.md"], "none", []),
        ([], "none", []),
        (["tests/ut/test_kda_dispatch.py"], "none", []),
        (
            [
                "vllm_sail/patch/enhancement/attention/kda.py",
                "tests/ut/test_kda_dispatch.py",
            ],
            "subset",
            ["kda"],
        ),
        (
            ["vllm_sail/patch/enhancement/attention/gdn.py"],
            "subset",
            ["gdn-decode", "gdn-prefill"],
        ),
        (["tests/e2e/fork_port/_tolerances.py"], "subset", ["gdn-decode"]),
        (["tests/e2e/test_native_activation.py"], "subset", ["activation"]),
        (["tests/e2e/test_fp8_linear_dispatch.py"], "subset", ["fp8-linear"]),
        (
            ["vllm_sail/model_executor/kernels/linear/scaled_mm/ppu.py"],
            "subset",
            ["fp8-linear"],
        ),
        (["vllm_sail/registry/linear_kernels/__init__.py"], "full", None),
        (["vllm_sail/utils/deep_gemm.py"], "full", None),
        (["vllm_sail/patch/enhancement/fp8_quant.py"], "full", None),
        (
            ["vllm_sail/models/deepseek_v4/ops/indexer.py"],
            "subset",
            ["deepseek-indexer-int8"],
        ),
        (["MANIFEST.in"], "full", None),
        ([".github/vllm-main-verified.commit"], "full", None),
        (["csrc/upstream/libtorch_stable/activation_kernels.cu"], "full", None),
        (["csrc/new.h"], "full", None),
        (["README.md", "vllm_sail/new.py"], "full", None),
        (["tests/ut/a.py", "vllm_sail/deleted.py"], "full", None),
        (["tests/e2e/new.py"], "full", None),
        (["AGENTS.md", "vllm_sail/patch/utils.py"], "full", None),
    ],
)
def test_selection_priority(selector, config, paths, mode, ids):
    result = selector.select_tests(config, paths, **OPTIONS)
    assert result["mode"] == mode
    assert result["test_ids"] == (
        ids if ids is not None else sorted(t["id"] for t in config["test_catalog"])
    )
    assert result["files"] == sorted(
        t["test"] for t in config["test_catalog"] if t["id"] in result["test_ids"]
    )
    if mode == "none":
        assert selector.route_tests(config, result["test_ids"]) == {
            "groups": [],
            "matrix": {"include": []},
        }


@pytest.mark.parametrize(
    "key,value,reason",
    [
        ("diff_complete", False, "incomplete_diff"),
        ("force_full", True, "forced_full"),
        ("environment_matches", False, "environment_changed"),
    ],
)
def test_full_fallback(selector, config, key, value, reason):
    result = selector.select_tests(config, ["README.md"], **{**OPTIONS, key: value})
    assert result["mode"] == "full"
    assert reason in result["reasons"]


@pytest.mark.parametrize(
    "path", ["/tmp/a", "../a", "a/../b", "a\\b", "-k", "x\x00y", "a::b", "./a"]
)
def test_rejects_unsafe_paths(selector, config, path):
    with pytest.raises(ValueError):
        selector.select_tests(config, [path], **OPTIONS)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c.update(schema=True),
        lambda c: c["test_catalog"].pop(),
        lambda c: c["test_catalog"].append(copy.deepcopy(c["test_catalog"][0])),
        lambda c: c["test_catalog"][0].update(boards=[]),
        lambda c: c["test_catalog"][0].update(boards=["unknown"]),
        lambda c: c["test_catalog"][0].update(
            board_overrides={"test_missing": ["ppu10"]}
        ),
        lambda c: c["test_catalog"][0].update(requires=["torch"]),
        lambda c: c["execution_profiles"].pop("ppu15"),
        lambda c: c["execution_profiles"]["ppu10"].update(nproc_per_node=2),
        lambda c: c["execution_profiles"]["ppu10"].update(command="arbitrary"),
        lambda c: c["selection_rules"]["source_to_tests"][0].update(
            test_ids=["missing"]
        ),
        lambda c: c["selection_rules"]["source_to_tests"][0].update(paths=["setup.py"]),
        lambda c: c["selection_rules"]["no_effect"]["paths"].append("missing.md"),
        lambda c: c["selection_rules"]["no_effect"]["prefixes"].append("vllm_sail/"),
        lambda c: c["selection_rules"]["no_effect"]["globs"].append("*.py"),
    ],
)
def test_invalid_config_fails_closed(selector, config, mutation):
    mutation(config)
    with pytest.raises(ValueError):
        selector.validate_config(config, ROOT)


@pytest.mark.parametrize(
    "text",
    [
        "import tests.ut.example",
        "from tests import ut",
        "from ..ut import example",
        "import importlib; importlib.import_module('tests.ut.example')",
        "import runpy; runpy.run_path('tests/ut/example.py')",
    ],
)
def test_ut_dependency_isolation(selector, tmp_path, text):
    path = tmp_path / "tests/e2e/helper.py"
    path.parent.mkdir(parents=True)
    path.write_text(text)
    with pytest.raises(ValueError, match="UT"):
        selector.validate_ut_isolation(tmp_path)


def test_ut_mentions_in_docstrings_are_not_dependencies(selector, tmp_path):
    path = tmp_path / "tests/conftest.py"
    path.parent.mkdir(parents=True)
    path.write_text('"""tests/ut/ runs on CPU only."""\n')
    selector.validate_ut_isolation(tmp_path)


def test_catalog_preflights_required_sdk_submodules(config):
    requirements = {t["id"]: set(t["requires"]) for t in config["test_catalog"]}
    assert {"pla.decode"} <= requirements["gdn-decode"]
    assert {"pla.prefill.flashqla", "pla.prefill.flashqla.ops"} <= requirements[
        "gdn-prefill"
    ]
    assert {"pla.decode.kda", "pla.prefill.flashkdapro"} <= requirements["kda"]


def test_fp8_linear_requires_deepgemm_on_ppu15(selector, config):
    item = next(t for t in config["test_catalog"] if t["id"] == "fp8-linear")
    assert item["boards"] == ["ppu15"]
    assert "deep_gemm" in item["requires"]
    assert item["board_overrides"] == {}
    routes = selector.route_tests(config, ["fp8-linear"])
    assert len(routes["groups"]) == 1
    assert routes["groups"][0]["board"] == "ppu15"
    assert routes["groups"][0]["files"] == ["tests/e2e/test_fp8_linear_dispatch.py"]


@pytest.mark.parametrize(
    "changed_path",
    [
        "tests/e2e/test_mxfp8_emulation.py",
        "vllm_sail/model_executor/kernels/linear/mxfp8.py",
    ],
)
def test_mxfp8_emulation_selected_on_both_boards(selector, config, changed_path):
    result = selector.select_tests(config, [changed_path], **OPTIONS)
    assert result["mode"] == "subset"
    assert result["test_ids"] == ["mxfp8-emulation"]
    item = next(t for t in config["test_catalog"] if t["id"] == "mxfp8-emulation")
    assert item == {
        "id": "mxfp8-emulation",
        "test": "tests/e2e/test_mxfp8_emulation.py",
        "boards": ["ppu10", "ppu15"],
        "requires": ["torch", "triton", "vllm", "vllm_sail"],
        "board_overrides": {},
    }
    routes = selector.route_tests(config, result["test_ids"])
    assert routes["groups"] == [
        {
            "group_id": board + "-0",
            "board": board,
            "test_ids": ["mxfp8-emulation"],
            "files": ["tests/e2e/test_mxfp8_emulation.py"],
        }
        for board in ("ppu10", "ppu15")
    ]


def test_routing_rejects_unknown_ids(selector, config):
    with pytest.raises(ValueError):
        selector.route_tests(config, ["missing"])
    routes = selector.route_tests(config, ["deepseek-indexer-int8"])
    assert len(routes["groups"]) == 1
    assert routes["groups"][0]["board"] == "ppu10"


def git(repo, *args):
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "CI",
            "GIT_AUTHOR_EMAIL": "ci@example.invalid",
            "GIT_COMMITTER_NAME": "CI",
            "GIT_COMMITTER_EMAIL": "ci@example.invalid",
        },
    )
    return result.stdout.strip()


def commit_file(repo, name):
    (repo / name).write_text(name)
    git(repo, "add", "--", name)
    git(repo, "commit", "-m", name)
    return git(repo, "rev-parse", "HEAD")


def test_cumulative_diff_merge_deletion_and_rename(selector, tmp_path):
    git(tmp_path, "init", "-b", "main")
    base = commit_file(tmp_path, "base")
    git(tmp_path, "switch", "-c", "pr")
    commit_file(tmp_path, "first file.py")
    head = commit_file(tmp_path, "second.py")
    git(tmp_path, "switch", "main")
    base_new = commit_file(tmp_path, "target.py")
    git(tmp_path, "switch", "pr")
    git(tmp_path, "merge", "--no-edit", "main")
    tested = git(tmp_path, "rev-parse", "HEAD")
    result = selector.collect_changes(tmp_path, base_new, head, tested)
    assert result["complete"] is True
    assert result["paths"] == ["first file.py", "second.py", "target.py"]
    git(tmp_path, "mv", "first file.py", "renamed.py")
    git(tmp_path, "commit", "-m", "rename")
    changed = selector.collect_changes(
        tmp_path,
        tested,
        git(tmp_path, "rev-parse", "HEAD"),
        git(tmp_path, "rev-parse", "HEAD"),
    )
    assert changed["paths"] == ["first file.py", "renamed.py"]
    missing = selector.collect_changes(tmp_path, "f" * 40, head, tested)
    assert not missing["complete"] and missing["paths"] == []
    assert (
        selector.collect_changes(tmp_path, None, None, base)["reason"]
        == "no_pr_context"
    )
    with pytest.raises(ValueError):
        selector.collect_changes(tmp_path, "main", head, tested)


@pytest.mark.parametrize(
    "image",
    [
        "",
        "repo:tag",
        "repo@sha256:abc",
        "repo@sha256:" + "A" * 64,
        "x y@sha256:" + "a" * 64,
    ],
)
def test_rejects_mutable_images(selector, image):
    with pytest.raises(ValueError):
        selector.resolve_build_image({}, image)


def test_digest_override_and_default(selector):
    first, second = (
        "registry/repo@sha256:" + "a" * 64,
        "registry/repo@sha256:" + "b" * 64,
    )
    env = {"build": {"build_image": first}}
    assert selector.resolve_build_image(env, "") == first
    assert selector.resolve_build_image(env, second) == second


@pytest.fixture
def selection_inputs(selector):
    image = "registry/image@sha256:" + "a" * 64
    build = dict(
        sdk_sha256="a" * 64,
        torch_sha256="b" * 64,
        pytorch_sail_arch=["ppu_10", "ppu_15"],
        build_image=image,
        vllm_commit="c" * 40,
    )
    profile = dict(
        capability=[8, 0],
        sdk_identity="SDK 2.2",
        driver_identity="1.2",
        packages=dict(torch="2.13", triton="3.5", pla="1.0"),
        runtime_env={"VLLM_SAIL_USE_PLA": True},
    )
    environment = dict(
        schema=1,
        qualified=True,
        build=build,
        runtime_image=image,
        profiles={b: copy.deepcopy(profile) for b in ("ppu10", "ppu15")},
        evidence=[
            dict(
                board=b,
                run_id=str(i),
                run_attempt="1",
                tested_sha="d" * 40,
                run_url=f"https://example.invalid/{i}",
            )
            for b in ("ppu10", "ppu15")
            for i in range(1, 4)
        ],
    )
    diagnostics = {
        **build,
        "pytorch_sail_arch": "ppu_15;ppu_10",
        "status": "succeeded",
        "vllm_sail_commit": "e" * 40,
    }
    wheel = dict(schema=1, commits={"vllm": "c" * 40, "vllm-sail": "e" * 40})
    changes = dict(
        schema=1,
        base_sha="a" * 40,
        head_sha="e" * 40,
        tested_sha="e" * 40,
        complete=True,
        paths=["vllm_sail/attention/pla_kda.py"],
        reason=None,
    )
    return dict(
        config_bytes=CONFIG.read_bytes(),
        environment_bytes=json.dumps(environment).encode(),
        changes=changes,
        diagnostics=diagnostics,
        wheel_manifest=wheel,
        build_run_id="123",
        build_image=image,
        runtime_image=image,
        expected_vllm_commit="c" * 40,
        force_full=False,
    )


def test_selection_protocol_and_sail_identity_not_environment(
    selector, selection_inputs
):
    result = selector.make_selection(**selection_inputs)
    assert result["mode"] == "subset" and result["test_ids"] == ["kda"]
    assert result["build_fingerprint"]["pytorch_sail_arch"] == ["ppu_10", "ppu_15"]
    selector.validate_selection(
        result,
        selection_inputs["config_bytes"],
        selection_inputs["environment_bytes"],
        tested_sha="e" * 40,
        build_run_id="123",
        build_image=selection_inputs["build_image"],
        runtime_image=selection_inputs["runtime_image"],
        expected_vllm_commit="c" * 40,
    )
    for field in ("head_sha", "tested_sha"):
        selection_inputs["changes"][field] = "f" * 40
    selection_inputs["diagnostics"]["vllm_sail_commit"] = "f" * 40
    selection_inputs["wheel_manifest"]["commits"]["vllm-sail"] = "f" * 40
    assert selector.make_selection(**selection_inputs)["mode"] == "subset"


@pytest.mark.parametrize(
    "key,value",
    [("sdk_sha256", "f" * 64), ("torch_sha256", ""), ("pytorch_sail_arch", "ppu_10")],
)
def test_build_observation_changes_force_full(selector, selection_inputs, key, value):
    selection_inputs["diagnostics"][key] = value
    result = selector.make_selection(**selection_inputs)
    assert result["mode"] == "full"
    assert "environment_changed" in result["reasons"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("vllm_commit", "f" * 40),
        ("vllm_sail_commit", "f" * 40),
        ("build_image", ""),
        ("status", "failed"),
    ],
)
def test_manifest_identity_mismatch_is_not_full_fallback(
    selector, selection_inputs, key, value
):
    selection_inputs["diagnostics"][key] = value
    with pytest.raises(ValueError):
        selector.make_selection(**selection_inputs)


def test_unqualified_environment_forces_full(selector, selection_inputs):
    selection_inputs["environment_bytes"] = ENVIRONMENT.read_bytes()
    result = selector.make_selection(**selection_inputs)
    assert result["mode"] == "full" and "environment_unqualified" in result["reasons"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s: s["groups"].append(copy.deepcopy(s["groups"][0])),
        lambda s: s["groups"][0]["files"].pop(),
        lambda s: s["matrix"]["include"][0].update(nproc_per_node=2),
        lambda s: s.update(config_sha256="0" * 64),
        lambda s: s.update(tested_sha="0" * 40),
        lambda s: s.update(build_run_id="999"),
        lambda s: s.update(runtime_image="registry/other@sha256:" + "b" * 64),
        lambda s: s.update(mode="none"),
    ],
)
def test_tampered_selection_rejected(selector, selection_inputs, mutation):
    result = selector.make_selection(**selection_inputs)
    mutation(result)
    with pytest.raises(ValueError):
        selector.validate_selection(
            result,
            selection_inputs["config_bytes"],
            selection_inputs["environment_bytes"],
            tested_sha="e" * 40,
            build_run_id="123",
            build_image=selection_inputs["build_image"],
            runtime_image=selection_inputs["runtime_image"],
            expected_vllm_commit="c" * 40,
        )


@pytest.mark.parametrize("damage", ["empty", "native", "no-pr"])
def test_nonfull_selection_cannot_drop_required_tests(
    selector, selection_inputs, damage
):
    result = selector.make_selection(**selection_inputs)
    if damage == "empty":
        result.update(
            mode="none", test_ids=[], files=[], groups=[], matrix={"include": []}
        )
    elif damage == "native":
        result["changed_files"] = ["csrc/changed.cu"]
    else:
        result["base_sha"] = None
    with pytest.raises(ValueError):
        selector.validate_selection(
            result,
            selection_inputs["config_bytes"],
            selection_inputs["environment_bytes"],
            tested_sha="e" * 40,
            build_run_id="123",
            build_image=selection_inputs["build_image"],
            runtime_image=selection_inputs["runtime_image"],
            expected_vllm_commit="c" * 40,
        )


def test_real_shallow_repository_and_git_failure_fall_back(
    selector, config, tmp_path, monkeypatch
):
    origin, shallow = tmp_path / "origin", tmp_path / "shallow"
    origin.mkdir()
    git(origin, "init", "-b", "main")
    base = commit_file(origin, "first.py")
    head = commit_file(origin, "second.py")
    git(tmp_path, "clone", "--depth=1", origin.as_uri(), str(shallow))
    result = selector.collect_changes(shallow, base, head, head)
    assert result["complete"] is False

    def unavailable(*args, **kwargs):
        raise subprocess.CalledProcessError(128, args[0])

    monkeypatch.setattr(selector.subprocess, "run", unavailable)
    failed = selector.collect_changes(origin, base, head, head)
    assert failed["reason"] == "incomplete_diff"
    assert (
        selector.select_tests(
            config, failed["paths"], **{**OPTIONS, "diff_complete": failed["complete"]}
        )["mode"]
        == "full"
    )


def test_config_hash_uses_original_bytes(selector, selection_inputs):
    first = selector.make_selection(**selection_inputs)
    selection_inputs["config_bytes"] += b"\n"
    second = selector.make_selection(**selection_inputs)
    assert first["config_sha256"] != second["config_sha256"]


def test_validate_cli_and_missing_pr_context(tmp_path):
    import sys

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "validate",
            "--repo",
            str(ROOT),
            "--config",
            str(CONFIG),
            "--environment",
            str(ENVIRONMENT),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["qualified"] is False
    output = tmp_path / "changes.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "changes",
            "--repo",
            str(ROOT),
            "--tested-sha",
            "a" * 40,
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["reason"] == "no_pr_context"


def test_qualified_environment_requires_evidence(selector):
    env = json.loads(ENVIRONMENT.read_text())
    assert selector.validate_environment(env) is False
    env["qualified"] = True
    with pytest.raises(ValueError):
        selector.validate_environment(env)
