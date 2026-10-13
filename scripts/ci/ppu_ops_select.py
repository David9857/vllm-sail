# SPDX-License-Identifier: Apache-2.0
"""Explicit PPU operator test selection, config validation, and cumulative Git diffs using only the standard library."""

from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path

BOARDS = {"ppu10": "OAM-810E", "ppu15": "ZW-M890P"}
ID_PATTERN = r"[a-z0-9]+(?:-[a-z0-9]+)*"
IMAGE_PATTERN = r"[a-zA-Z0-9][a-zA-Z0-9._:/-]*@sha256:[0-9a-f]{64}"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def keys(value, expected):
    require(
        isinstance(value, dict) and set(value) == set(expected),
        f"字段不合法: {expected}",
    )


def strings(value, *, nonempty=False):
    require(isinstance(value, list), "必须为数组")
    require(all(isinstance(x, str) and x for x in value), "必须为非空字符串")
    require(len(value) == len(set(value)), "数组包含重复项")
    require(not nonempty or bool(value), "数组不得为空")
    return value


def safe_path(value, *, pattern=False):
    require(isinstance(value, str) and bool(value), "路径为空")
    require(not value.startswith(("/", "-")), "路径必须相对仓库")
    require(not any(c in value for c in ("\\", "\0", ":", "\n", "\r")), "非法路径字符")
    require(
        all(p not in ("", ".", "..") for p in value.rstrip("/").split("/")),
        "非法路径分量",
    )
    require(pattern or not any(c in value for c in "*?[]"), "精确路径不能包含 glob")
    return value


def sha(value):
    require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value),
        "需要不可变 40 位 SHA",
    )
    return value


def digest_image(value):
    require(
        isinstance(value, str) and re.fullmatch(IMAGE_PATTERN, value),
        "镜像必须为完整 digest 引用",
    )
    return value


def decimal(value):
    require(
        isinstance(value, str) and re.fullmatch(r"[1-9][0-9]*", value),
        "需要正整数身份字符串",
    )
    return value


def matches(path, rule):
    return (
        path in rule["paths"]
        or any(path.startswith(p) for p in rule["prefixes"])
        or any(fnmatch.fnmatchcase(path, p) for p in rule["globs"])
    )


def validate_profiles(profiles):
    keys(profiles, BOARDS)
    for board, name in BOARDS.items():
        item = profiles[board]
        keys(item, ("runner_group", "node_selector", "nproc_per_node", "group_count"))
        require(item["runner_group"] == "k8s-runner-group-cpu-thead", "未知 CPU runner")
        require(item["node_selector"] == {"board-type": name}, "板型路由不合法")
        for key in ("nproc_per_node", "group_count"):
            require(type(item[key]) is int and item[key] == 1, "首版仅允许单卡单组")


def validate_ut_isolation(repo):
    repo = Path(repo)
    paths = set()
    for directory in ("vllm_sail", "tests/e2e", "tests/support"):
        paths.update((repo / directory).rglob("*.py"))
    paths.update(
        repo / p
        for p in ("tests/conftest.py", "tests/__init__.py")
        if (repo / p).is_file()
    )
    for path in sorted(paths):
        package = ".".join(path.relative_to(repo).parent.parts)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = {
            n.body[0].value
            for n in ast.walk(tree)
            if isinstance(
                n, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
            )
            and ast.get_docstring(n, clean=False) is not None
        }
        for node in ast.walk(tree):
            if node in docstrings:
                continue
            references = []
            if isinstance(node, ast.Import):
                references.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                try:
                    module = importlib.util.resolve_name(
                        "." * node.level + (node.module or ""), package
                    )
                except (ImportError, ValueError) as error:
                    raise ValueError(f"无法解析 import: {path}") from error
                references += [module] + [module + "." + a.name for a in node.names]
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                references.append(node.value.replace("/", "."))
            require(
                not any("tests.ut" in ref for ref in references),
                f"禁止生产/设备代码依赖 UT: {path}",
            )


def validate_config(config, repo):
    repo = Path(repo).resolve()
    keys(config, ("schema", "selection_rules", "test_catalog", "execution_profiles"))
    require(type(config["schema"]) is int and config["schema"] == 1, "未知 schema")
    validate_profiles(config["execution_profiles"])
    catalog = config["test_catalog"]
    require(isinstance(catalog, list) and bool(catalog), "缺少测试目录")
    ids, tests = set(), set()
    for item in catalog:
        keys(item, ("id", "test", "boards", "requires", "board_overrides"))
        require(
            isinstance(item["id"], str) and re.fullmatch(ID_PATTERN, item["id"]),
            "非法测试 ID",
        )
        require(item["id"] not in ids, "重复测试 ID")
        ids.add(item["id"])
        path = safe_path(item["test"])
        require(
            path.startswith("tests/e2e/") and path not in tests, "重复或非法测试路径"
        )
        tests.add(path)
        actual = (repo / path).resolve()
        require(
            actual.is_relative_to(repo) and actual.is_file(), "测试路径不存在或逃逸仓库"
        )
        boards = set(strings(item["boards"], nonempty=True))
        require(boards <= BOARDS.keys(), "未知适用板型")
        requires = strings(item["requires"], nonempty=True)
        require(requires == sorted(requires), "依赖必须排序")
        require(
            all(re.fullmatch(r"[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*", x) for x in requires),
            "非法 import 名",
        )
        required = {"torch", "triton", "vllm", "vllm_sail"}
        if item["id"] in {"gdn-decode", "gdn-prefill", "kda"}:
            required.add("pla")
        require(required <= set(requires), "缺少必需依赖")
        functions = {
            n.name
            for n in ast.walk(ast.parse(actual.read_text(encoding="utf-8")))
            if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
            and n.name.startswith("test_")
        }
        require(bool(functions), "测试文件无测试函数")
        overrides = item["board_overrides"]
        require(isinstance(overrides, dict), "非法板型例外")
        for function, supported in overrides.items():
            require(function in functions, "失效函数例外")
            require(
                set(strings(supported, nonempty=True)) <= boards, "例外超出文件板型"
            )
        for board in boards:
            require(
                any(board in overrides.get(f, boards) for f in functions),
                "板型没有适用函数",
            )
    discovered = {
        p.relative_to(repo).as_posix() for p in (repo / "tests/e2e").rglob("test_*.py")
    }
    require(discovered == tests, f"测试目录不完整: {sorted(discovered ^ tests)}")
    rules = config["selection_rules"]
    keys(rules, ("full", "no_effect", "source_to_tests"))
    for category in ("full", "no_effect"):
        rule = rules[category]
        keys(rule, ("paths", "prefixes", "globs"))
        for kind, values in rule.items():
            for path in strings(values):
                safe_path(path, pattern=(kind == "globs"))
                if kind == "paths":
                    require(
                        (repo / path).is_file()
                        and (repo / path).resolve().is_relative_to(repo),
                        "精确规则路径不存在或逃逸",
                    )
                if kind == "prefixes":
                    require(path.endswith("/"), "目录前缀必须以 / 结尾")
                if category == "no_effect":
                    # This bounds the no-effect policy; it is not another test-selection mapping.
                    allowed = (
                        (
                            kind == "paths"
                            and (
                                path
                                in {
                                    "README.md",
                                    "CONTRIBUTING.md",
                                    "AGENTS.md",
                                    "LICENSE",
                                }
                                or (path.startswith("docs/") and path.endswith(".md"))
                            )
                        )
                        or (kind == "prefixes" and path == "tests/ut/")
                        or (kind == "globs" and path == "docs/*.md")
                    )
                    require(allowed, "未批准的无影响规则")
    rule_ids = set()
    require(isinstance(rules["source_to_tests"], list), "局部规则必须为数组")
    for rule in rules["source_to_tests"]:
        keys(rule, ("id", "paths", "test_ids"))
        require(
            isinstance(rule["id"], str) and re.fullmatch(ID_PATTERN, rule["id"]),
            "非法规则 ID",
        )
        require(rule["id"] not in rule_ids, "重复规则 ID")
        rule_ids.add(rule["id"])
        require(
            set(strings(rule["test_ids"], nonempty=True)) <= ids, "规则引用未知测试"
        )
        for path in strings(rule["paths"], nonempty=True):
            safe_path(path)
            require(
                (repo / path).is_file()
                and (repo / path).resolve().is_relative_to(repo),
                "悬空源码路径",
            )
            require(not matches(path, rules["full"]), "局部规则被全量覆盖")
    validate_ut_isolation(repo)
    routed = route_tests(config, sorted(ids))
    pairs = [(test, g["board"]) for g in routed["groups"] for test in g["test_ids"]]
    expected = {(t["id"], b) for t in catalog for b in t["boards"]}
    require(
        len(pairs) == len(set(pairs)) and set(pairs) == expected, "路由不完整或重复"
    )


def select_tests(config, paths, *, diff_complete, force_full, environment_matches):
    paths = sorted(set(safe_path(path) for path in paths))
    catalog = config["test_catalog"]
    rules = config["selection_rules"]
    selected, reasons, matched, unmapped = set(), set(), set(), set()
    full = force_full or not diff_complete or not environment_matches
    if force_full:
        reasons.add("forced_full")
    if not diff_complete:
        reasons.add("incomplete_diff")
    if not environment_matches:
        reasons.add("environment_changed")
    by_file = {t["test"]: t["id"] for t in catalog}
    for path in paths:
        if matches(path, rules["full"]):
            full = True
            reasons.add(
                "native_change" if path.startswith("csrc/") else "global_dependency"
            )
            continue
        found = False
        if path in by_file:
            selected.add(by_file[path])
            reasons.add("test_changed")
            found = True
        for rule in rules["source_to_tests"]:
            if path in rule["paths"]:
                selected.update(rule["test_ids"])
                matched.add(rule["id"])
                reasons.add("mapped_source")
                found = True
        if not found and not matches(path, rules["no_effect"]):
            full = True
            unmapped.add(path)
            reasons.add("unmapped_path")
    if full:
        selected = {t["id"] for t in catalog}
    return {
        "mode": "full" if full else "subset" if selected else "none",
        "test_ids": sorted(selected),
        "files": sorted(t["test"] for t in catalog if t["id"] in selected),
        "reasons": sorted(reasons or {"no_device_changes"}),
        "matched_rule_ids": sorted(matched),
        "unmapped_files": sorted(unmapped),
    }


def route_tests(config, test_ids):
    validate_profiles(config["execution_profiles"])
    catalog = {t["id"]: t for t in config["test_catalog"]}
    require(set(test_ids) <= catalog.keys(), "未知测试 ID")
    groups, matrix = [], []
    for test_id in test_ids:
        require(
            bool(catalog[test_id]["boards"])
            and set(catalog[test_id]["boards"]) <= BOARDS.keys(),
            "无有效路由",
        )
    for board in sorted(BOARDS):
        ids = sorted(t for t in set(test_ids) if board in catalog[t]["boards"])
        if ids:
            groups.append(
                {
                    "group_id": board + "-0",
                    "board": board,
                    "test_ids": ids,
                    "files": sorted(catalog[t]["test"] for t in ids),
                }
            )
            matrix.append(
                {"group_id": board + "-0", "board": board, "nproc_per_node": 1}
            )
    return {"groups": groups, "matrix": {"include": matrix}}


def collect_changes(repo, base_sha, head_sha, tested_sha):
    sha(tested_sha)
    for value in (base_sha, head_sha):
        if value is not None:
            sha(value)
    result = {
        "schema": 1,
        "base_sha": base_sha,
        "head_sha": head_sha,
        "tested_sha": tested_sha,
        "complete": False,
        "paths": [],
        "reason": "no_pr_context",
    }
    if base_sha is None or head_sha is None:
        return result

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(repo), *args], check=True, capture_output=True
        ).stdout

    try:
        # A shallow clone may find a common ancestor without having the full ancestry graph.
        require(
            git("rev-parse", "--is-shallow-repository").strip() == b"false",
            "shallow_repository",
        )
        base = git("merge-base", base_sha, head_sha).decode().strip()
        sha(base)
        paths = set()
        for old, new in ((base, head_sha), (head_sha, tested_sha)):
            raw = git("diff", "--name-only", "-z", "--no-renames", old, new, "--")
            paths.update(safe_path(p.decode("utf-8")) for p in raw.split(b"\0") if p)
        result.update(complete=True, paths=sorted(paths), reason=None)
    except (subprocess.SubprocessError, OSError, UnicodeError, ValueError):
        result["reason"] = "incomplete_diff"
    return result


def resolve_build_image(environment, override):
    return digest_image(override or (environment.get("build") or {}).get("build_image"))


def validate_environment(environment):
    keys(
        environment,
        ("schema", "qualified", "build", "runtime_image", "profiles", "evidence"),
    )
    require(
        type(environment["schema"]) is int and environment["schema"] == 1,
        "环境 schema 错误",
    )
    require(type(environment["qualified"]) is bool, "qualified 必须为 boolean")
    build = environment["build"]
    if build is not None:
        validate_fingerprint(build)
    if environment["runtime_image"] is not None:
        digest_image(environment["runtime_image"])
    require(isinstance(environment["profiles"], dict), "环境 profiles 必须为对象")
    require(environment["profiles"].keys() <= BOARDS.keys(), "未知环境板型")
    for profile in environment["profiles"].values():
        validate_runtime_profile(profile)
    evidence = environment["evidence"]
    require(isinstance(evidence, list), "证据必须为数组")
    identities = {board: set() for board in BOARDS}
    for record in evidence:
        keys(record, ("board", "run_id", "run_attempt", "tested_sha", "run_url"))
        require(record["board"] in BOARDS, "未知证据板型")
        decimal(record["run_id"])
        decimal(record["run_attempt"])
        sha(record["tested_sha"])
        require(
            isinstance(record["run_url"], str)
            and record["run_url"].startswith("https://"),
            "缺少证据 URL",
        )
        identity = (record["run_id"], record["run_attempt"])
        require(identity not in identities[record["board"]], "重复资格证据")
        identities[record["board"]].add(identity)
    if environment["qualified"]:
        require(
            build is not None and environment["runtime_image"] is not None,
            "批准环境缺少镜像/构建",
        )
        require(environment["profiles"].keys() == BOARDS.keys(), "批准环境缺少双板型")
        require(
            all(len(runs) >= 3 for runs in identities.values()),
            "每板型至少三次独立证据",
        )
    return environment["qualified"]


def validate_fingerprint(build):
    keys(
        build,
        (
            "sdk_sha256",
            "torch_sha256",
            "pytorch_sail_arch",
            "build_image",
            "vllm_commit",
        ),
    )
    for name in ("sdk_sha256", "torch_sha256"):
        require(
            isinstance(build[name], str) and re.fullmatch(r"[0-9a-f]{64}", build[name]),
            "构建 hash 不完整",
        )
    require(
        build["pytorch_sail_arch"] == ["ppu_10", "ppu_15"], "架构必须包含两个目标并排序"
    )
    digest_image(build["build_image"])
    sha(build["vllm_commit"])


def validate_runtime_profile(profile):
    keys(
        profile,
        ("capability", "sdk_identity", "driver_identity", "packages", "runtime_env"),
    )
    cap = profile["capability"]
    require(
        isinstance(cap, list)
        and len(cap) == 2
        and all(type(n) is int and n >= 0 for n in cap),
        "非法 capability",
    )
    for name in ("sdk_identity", "driver_identity"):
        require(
            isinstance(profile[name], str)
            and profile[name].strip()
            and profile[name].lower() != "unknown",
            "运行时身份缺失",
        )
    keys(profile["packages"], ("torch", "triton", "pla"))
    require(
        all(
            isinstance(v, str) and v.strip() and v.lower() != "unknown"
            for v in profile["packages"].values()
        ),
        "版本缺失",
    )
    require(profile["runtime_env"] == {"VLLM_SAIL_USE_PLA": True}, "PLA 必须启用")


def checksum(data):
    return hashlib.sha256(data).hexdigest()


def load_sibling(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_diagnostics(path):
    result = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        require(separator and key, "非法构建诊断")
        require(key == "status" or key not in result, "重复构建字段")
        result[key] = value
    return result


def build_fingerprint(
    diagnostics, wheel_manifest, *, tested_sha, expected_vllm_commit, build_image
):
    sha(tested_sha)
    sha(expected_vllm_commit)
    digest_image(build_image)
    require(diagnostics.get("status") == "succeeded", "构建未成功")
    for field, package, expected in (
        ("vllm_commit", "vllm", expected_vllm_commit),
        ("vllm_sail_commit", "vllm-sail", tested_sha),
    ):
        require(diagnostics.get(field) == expected, f"构建诊断身份不符: {field}")
        require(
            wheel_manifest.get("commits", {}).get(package) == expected,
            f"wheel 身份不符: {package}",
        )
    require(diagnostics.get("build_image") == build_image, "实际构建镜像不符")
    result = {"build_image": build_image, "vllm_commit": expected_vllm_commit}
    for key in ("sdk_sha256", "torch_sha256"):
        value = diagnostics.get(key, "")
        result[key] = value if re.fullmatch(r"[0-9a-f]{64}", value) else None
    arches = sorted(
        set(filter(None, diagnostics.get("pytorch_sail_arch", "").split(";")))
    )
    result["pytorch_sail_arch"] = arches or None
    return result


def validate_changes(changes):
    keys(
        changes,
        ("schema", "base_sha", "head_sha", "tested_sha", "complete", "paths", "reason"),
    )
    require(
        type(changes["schema"]) is int and changes["schema"] == 1, "diff schema 错误"
    )
    sha(changes["tested_sha"])
    for name in ("base_sha", "head_sha"):
        if changes[name] is not None:
            sha(changes[name])
    require(type(changes["complete"]) is bool, "diff 完整性非法")
    for path in strings(changes["paths"]):
        safe_path(path)
    if changes["complete"]:
        require(
            changes["base_sha"] is not None
            and changes["head_sha"] is not None
            and changes["reason"] is None,
            "完整 diff 缺少身份",
        )
    else:
        require(
            changes["reason"] in ("no_pr_context", "incomplete_diff"),
            "缺少 diff 失败原因",
        )


def make_selection(
    *,
    config_bytes,
    environment_bytes,
    changes,
    diagnostics,
    wheel_manifest,
    build_run_id,
    build_image,
    runtime_image,
    expected_vllm_commit,
    force_full,
):
    config, environment = json.loads(config_bytes), json.loads(environment_bytes)
    qualified = validate_environment(environment)
    validate_changes(changes)
    decimal(build_run_id)
    digest_image(runtime_image)
    fingerprint = build_fingerprint(
        diagnostics,
        wheel_manifest,
        tested_sha=changes["tested_sha"],
        expected_vllm_commit=expected_vllm_commit,
        build_image=build_image,
    )
    matched = (
        qualified
        and environment["build"] == fingerprint
        and environment["runtime_image"] == runtime_image
    )
    selected = select_tests(
        config,
        changes["paths"],
        diff_complete=changes["complete"],
        force_full=force_full,
        environment_matches=matched,
    )
    if not qualified:
        selected["reasons"] = sorted(
            (set(selected["reasons"]) - {"environment_changed"})
            | {"environment_unqualified"}
        )
    return {
        "schema": 1,
        **selected,
        **route_tests(config, selected["test_ids"]),
        **{k: changes[k] for k in ("base_sha", "head_sha", "tested_sha")},
        "changed_files": sorted(changes["paths"]),
        "build_run_id": build_run_id,
        "config_sha256": checksum(config_bytes),
        "environment_sha256": checksum(environment_bytes),
        "build_fingerprint": fingerprint,
        "runtime_image": runtime_image,
    }


def validate_selection(
    selection,
    config_bytes,
    environment_bytes,
    *,
    tested_sha,
    build_run_id,
    build_image,
    runtime_image,
    expected_vllm_commit,
):
    keys(
        selection,
        (
            "schema",
            "mode",
            "reasons",
            "matched_rule_ids",
            "changed_files",
            "unmapped_files",
            "test_ids",
            "files",
            "groups",
            "matrix",
            "base_sha",
            "head_sha",
            "tested_sha",
            "build_run_id",
            "config_sha256",
            "environment_sha256",
            "build_fingerprint",
            "runtime_image",
        ),
    )
    require(
        type(selection["schema"]) is int and selection["schema"] == 1,
        "选择 schema 错误",
    )
    require(selection["tested_sha"] == sha(tested_sha), "tested SHA 不符")
    require(selection["build_run_id"] == decimal(build_run_id), "build run 不符")
    require(selection["runtime_image"] == digest_image(runtime_image), "运行镜像不符")
    require(selection["config_sha256"] == checksum(config_bytes), "配置 checksum 不符")
    require(
        selection["environment_sha256"] == checksum(environment_bytes),
        "环境 checksum 不符",
    )
    config = json.loads(config_bytes)
    environment = json.loads(environment_bytes)
    qualified = validate_environment(environment)
    for name in ("base_sha", "head_sha"):
        if selection[name] is not None:
            sha(selection[name])
    for name in (
        "test_ids",
        "files",
        "reasons",
        "matched_rule_ids",
        "changed_files",
        "unmapped_files",
    ):
        require(strings(selection[name]) == sorted(selection[name]), f"未排序: {name}")
    for path in (
        selection["files"] + selection["changed_files"] + selection["unmapped_files"]
    ):
        safe_path(path)
    routes = route_tests(config, selection["test_ids"])
    require(
        selection["groups"] == routes["groups"]
        and selection["matrix"] == routes["matrix"],
        "组/矩阵不符",
    )
    expected_files = sorted(
        t["test"] for t in config["test_catalog"] if t["id"] in selection["test_ids"]
    )
    require(selection["files"] == expected_files, "ID/文件不符")
    mode = selection["mode"]
    require(mode in ("none", "subset", "full"), "未知选择模式")
    require(bool(selection["test_ids"]) == (mode != "none"), "空/非空模式不符")
    if mode == "full":
        require(
            selection["test_ids"] == sorted(t["id"] for t in config["test_catalog"]),
            "全量清单不完整",
        )
    fingerprint = selection["build_fingerprint"]
    keys(
        fingerprint,
        (
            "sdk_sha256",
            "torch_sha256",
            "pytorch_sail_arch",
            "build_image",
            "vllm_commit",
        ),
    )
    require(fingerprint["build_image"] == digest_image(build_image), "构建镜像不符")
    require(
        fingerprint["vllm_commit"] == sha(expected_vllm_commit), "upstream SHA 不符"
    )
    if mode != "full":
        require(
            qualified
            and fingerprint == environment["build"]
            and runtime_image == environment["runtime_image"],
            "非全量清单的环境未经批准",
        )
        require(
            selection["base_sha"] is not None and selection["head_sha"] is not None,
            "非全量清单缺少 PR 身份",
        )
        expected = select_tests(
            config,
            selection["changed_files"],
            diff_complete=True,
            force_full=False,
            environment_matches=True,
        )
        require(
            all(selection[key] == value for key, value in expected.items()),
            "非全量清单与源码规则不一致",
        )
    return routes


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def github_outputs(values):
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as stream:
            for key, value in values.items():
                text = (
                    value
                    if isinstance(value, str)
                    else json.dumps(value, separators=(",", ":"))
                )
                require("\n" not in text and "\r" not in text, "多行 output 不合法")
                stream.write(f"{key}={text}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("changes", "select", "validate"):
        command = commands.add_parser(name)
        command.add_argument("--repo", type=Path, default=Path.cwd())
        if name == "changes":
            command.add_argument("--base-sha")
            command.add_argument("--head-sha")
            command.add_argument("--tested-sha", required=True)
        else:
            command.add_argument("--config", type=Path, required=True)
            command.add_argument("--environment", type=Path, required=True)
        if name != "validate":
            command.add_argument("--output", type=Path, required=True)
        if name == "select":
            for field in ("changes", "build-manifest", "wheel-manifest"):
                command.add_argument("--" + field, type=Path, required=True)
            for field in (
                "build-run-id",
                "build-image",
                "runtime-image",
                "expected-vllm-commit",
            ):
                command.add_argument("--" + field, required=True)
            command.add_argument("--scope", choices=("auto", "full"), required=True)
    args = parser.parse_args()
    if args.command == "changes":
        result = collect_changes(
            args.repo, args.base_sha, args.head_sha, args.tested_sha
        )
    else:
        config_bytes, environment_bytes = (
            args.config.read_bytes(),
            args.environment.read_bytes(),
        )
        validate_config(json.loads(config_bytes), args.repo)
        qualified = validate_environment(json.loads(environment_bytes))
        if args.command == "validate":
            print(json.dumps({"schema_valid": True, "qualified": qualified}))
            return
        changes = json.loads(args.changes.read_bytes())
        require(
            args.wheel_manifest.name == "wheel-manifest.json",
            "wheel manifest 文件名不符",
        )
        manifest = load_sibling("ppu_wheel_manifest").verify(
            args.wheel_manifest.parent,
            vllm_commit=sha(args.expected_vllm_commit),
            sail_commit=sha(changes["tested_sha"]),
        )
        result = make_selection(
            config_bytes=config_bytes,
            environment_bytes=environment_bytes,
            changes=changes,
            diagnostics=read_diagnostics(args.build_manifest),
            wheel_manifest=manifest,
            build_run_id=args.build_run_id,
            build_image=args.build_image,
            runtime_image=args.runtime_image,
            expected_vllm_commit=args.expected_vllm_commit,
            force_full=args.scope == "full",
        )
    write_json(args.output, result)
    if args.command == "select":
        github_outputs(
            {
                "mode": result["mode"],
                "matrix": result["matrix"],
                "has_tests": bool(result["groups"]),
            }
        )


if __name__ == "__main__":
    main()
