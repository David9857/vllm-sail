# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "scripts" / "ci" / "ppu_e2e_smoke.sh"
SELECT = ROOT / "scripts" / "ci" / "ppu_e2e_select.py"
MODELS = ROOT / "scripts" / "ci" / "ppu_e2e_models.json"
WORKFLOW = ROOT / ".github" / "workflows" / "e2e-ppu.yaml"


def _load_select():
    spec = importlib.util.spec_from_file_location("ppu_e2e_select", SELECT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# --- model catalog -----------------------------------------------------------


def test_models_catalog_is_valid_json_with_required_fields() -> None:
    catalog = json.loads(MODELS.read_text())
    assert isinstance(catalog, list) and catalog
    required = {"key", "checkpoint", "served_name", "board", "tp"}
    for entry in catalog:
        assert required <= set(entry), entry
    keys = [entry["key"] for entry in catalog]
    assert len(keys) == len(set(keys)), "model keys must be unique"


def test_default_matrix_targets_qwen_on_810e() -> None:
    catalog = json.loads(MODELS.read_text())
    qwen = next(e for e in catalog if e["key"] == "qwen3.8-27b")
    # Checkpoints are read from /nas_aisw/datasets, the internal NAS root
    # recorded in the model catalog sheet.
    assert qwen["checkpoint"] == (
        "/nas_aisw/datasets/checkpoints/LLM/Qwen/v3.8/Qwen3.8-27B"
    )
    # The built wheels (arch ppu_15;ppu_10) are only verified on 810E.
    assert qwen["board"] == "OAM-810E"
    assert int(qwen["tp"]) == 2


# --- matrix selector ----------------------------------------------------------


def test_selector_default_returns_every_model() -> None:
    select = _load_select()
    catalog = json.loads(MODELS.read_text())
    matrix = select.build_matrix(catalog, "")
    assert [m["key"] for m in matrix["include"]] == [e["key"] for e in catalog]


def test_selector_filters_to_requested_keys() -> None:
    select = _load_select()
    catalog = [
        {"key": "a", "checkpoint": "/x", "served_name": "a", "board": "b", "tp": 1},
        {"key": "b", "checkpoint": "/y", "served_name": "b", "board": "b", "tp": 1},
    ]
    matrix = select.build_matrix(catalog, " b , a ")
    assert [m["key"] for m in matrix["include"]] == ["b", "a"]


def test_selector_rejects_unknown_key() -> None:
    select = _load_select()
    catalog = [
        {"key": "a", "checkpoint": "/x", "served_name": "a", "board": "b", "tp": 1},
    ]
    try:
        select.build_matrix(catalog, "does-not-exist")
    except (KeyError, ValueError, SystemExit):
        return
    raise AssertionError("unknown model key must be rejected")


def test_selector_cli_emits_github_output_matrix(tmp_path: Path) -> None:
    result = subprocess.run(
        ["python3", str(SELECT), ""],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    line = next(ln for ln in result.stdout.splitlines() if ln.startswith("matrix="))
    payload = json.loads(line[len("matrix=") :])
    assert "include" in payload and payload["include"]


# --- pod smoke script ---------------------------------------------------------


def test_smoke_script_has_valid_bash_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", str(SMOKE)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_smoke_script_preserves_required_e2e_contract() -> None:
    text = SMOKE.read_text()
    required = (
        "set -Eeuo pipefail",
        # wheels are installed without pulling deps to keep the SAIL native ABI.
        "--no-deps",
        "--force-reinstall",
        "from vllm_sail import collect_env",
        "vllm serve",
        "/v1/chat/completions",
        # readiness is confirmed by the vLLM startup banner, not a fixed sleep.
        "Application startup complete",
        # occupancy evidence for the run log.
        "ppu-smi",
        "unset VLLM_VERSION",
        "check_vllm_compatibility(force=True)",
        "ppu_wheel_manifest.py",
    )
    for marker in required:
        assert marker in text, marker


def test_smoke_script_fails_when_serve_dies_early() -> None:
    text = SMOKE.read_text()
    # The readiness poll must detect an early-exiting serve process instead of
    # blocking until timeout.
    assert "kill -0" in text


# --- workflow -----------------------------------------------------------------


def test_e2e_workflow_is_reusable_and_manual_and_uses_cpu_runner() -> None:
    text = WORKFLOW.read_text()
    assert "workflow_dispatch:" in text
    # PR orchestration calls the same workflow as manual smoke runs.
    assert "workflow_call:" in text
    assert "pull_request:" not in text
    assert "push:" not in text
    # ppu-scheduler-action itself runs on the CPU runner scale set and dispatches
    # a PPU pod; the workflow must not try to run on PPU directly.
    assert "runs-on: k8s-runner-group-cpu-thead" in text
    assert "t-head/ppu-scheduler-action@main" in text
    # board is data-driven from the matrix (the literal OAM-810E lives in the
    # catalog and is asserted by test_default_matrix_targets_qwen_on_810e).
    assert "board-type=${{ matrix.model.board }}" in text
    assert "contents: read" in text
    # cross-run artifact download needs the actions:read scope.
    assert "actions: read" in text


def test_e2e_workflow_only_exposes_model_smoke() -> None:
    text = WORKFLOW.read_text()
    workflow = yaml.safe_load(text)
    trigger = workflow.get("on", workflow.get(True))
    assert set(trigger) == {"workflow_call", "workflow_dispatch"}
    for event in trigger.values():
        assert set(event["inputs"]) == {"build_run_id", "models", "image"}
    assert set(workflow["jobs"]) == {"prepare", "e2e"}
    assert "if" not in workflow["jobs"]["prepare"]
    assert workflow["jobs"]["e2e"]["needs"] == "prepare"
    for marker in (
        "device_probe_only",
        "probe_suite",
        "probe_sail_commit",
        "ppu_ops_probe",
    ):
        assert marker not in text
    caller = yaml.safe_load(
        (ROOT / ".github/workflows/build-ppu-wheels.yaml").read_text()
    )
    assert caller["jobs"]["smoke"]["uses"] == "./.github/workflows/e2e-ppu.yaml"
    assert caller["jobs"]["gate"]["needs"] == ["build", "pin-unit-tests", "smoke"]


def test_e2e_workflow_pulls_wheels_by_run_id() -> None:
    text = WORKFLOW.read_text()
    assert "build_run_id:" in text
    assert "actions/download-artifact@v4" in text
    assert "run-id: ${{ inputs.build_run_id }}" in text
    assert "ppu-wheels-${{ inputs.build_run_id }}" in text


def test_e2e_workflow_keeps_python_frontend_and_privileged() -> None:
    text = WORKFLOW.read_text()
    assert "VLLM_VERSION=" not in text
    assert "VLLM_USE_RUST_FRONTEND=0" in text
    assert "VLLM_USE_RUST_BENCH=0" in text
    assert "privileged: true" in text
    # smoke work is delegated to the checked-in script, not inlined in YAML.
    assert "scripts/ci/ppu_e2e_smoke.sh" in text


def test_e2e_workflow_gives_every_cpu_runner_job_a_container() -> None:
    text = WORKFLOW.read_text()
    # k8s-runner-group-cpu-thead rejects container-less jobs ("Jobs without a
    # job container are forbidden on this runner"), so every job on that runner
    # must declare a container image.
    runner_jobs = text.count("runs-on: k8s-runner-group-cpu-thead")
    container_decls = sum(
        1 for line in text.splitlines() if line.strip() == "container:"
    )
    assert runner_jobs >= 1
    assert container_decls >= runner_jobs, (runner_jobs, container_decls)


def test_smoke_checks_manifest_before_install_and_uses_installed_package() -> None:
    text = SMOKE.read_text()
    assert text.index("ppu_wheel_manifest.py") < text.index("python -m pip install")
    assert text.index("cd /tmp") < text.index("check_vllm_compatibility(force=True)")
    assert "export VLLM_USE_RUST_FRONTEND=0 VLLM_USE_RUST_BENCH=0" in text
    assert "unset PYTHONPATH" in text
