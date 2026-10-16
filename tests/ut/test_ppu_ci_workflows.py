# SPDX-License-Identifier: Apache-2.0
import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github/workflows"
DEFAULT_BRANCH = "${{ github.event.repository.default_branch }}"
EXPECTED_SHA = "${{ inputs.pr_head_sha || github.sha }}"
DEFAULT_REF_GUARD = (
    "github.ref == format('refs/heads/{0}', github.event.repository.default_branch)"
)


def workflow(name):
    path = WORKFLOWS / name
    assert path.is_file(), f"Missing workflow: {name}"
    return yaml.safe_load(path.read_text())


def checkouts(job):
    return [
        step
        for step in job["steps"]
        if step.get("uses", "").startswith("actions/checkout@")
    ]


@pytest.mark.parametrize(
    "name,job_id,ref",
    [
        ("run-ci-command.yml", "command", DEFAULT_BRANCH),
        ("notify-ci-authorized.yml", "notify", DEFAULT_BRANCH),
        ("ppu-pr-ci.yaml", "validate", "${{ github.sha }}"),
        ("ppu-pr-ci.yaml", "report", "${{ github.sha }}"),
    ],
)
def test_control_jobs_only_execute_trusted_code(name, job_id, ref):
    job = workflow(name)["jobs"][job_id]
    assert job["runs-on"] == "ubuntu-latest"
    assert len(checkouts(job)) == 1
    checkout = checkouts(job)[0]
    assert checkout["with"]["ref"] == ref
    assert checkout["with"]["persist-credentials"] is False
    if name == "ppu-pr-ci.yaml":
        assert DEFAULT_REF_GUARD in job.get("if", checkout.get("if", "")) or (
            job_id == "report"
            and all(DEFAULT_REF_GUARD in step.get("if", "") for step in job["steps"])
        )


def test_controller_filters_comments_serializes_all_commands_and_reports_failure():
    config = workflow("run-ci-command.yml")
    assert config[True] == {"issue_comment": {"types": ["created"]}}
    job = config["jobs"]["command"]
    assert "github.event.issue.pull_request" in job["if"]
    assert "github.event.comment.user.type == 'User'" in job["if"]
    assert "contains(github.event.comment.body, '/ci')" in job["if"]
    assert job["concurrency"] == {
        "group": "ppu-ci-command-${{ github.event.issue.number }}",
        "cancel-in-progress": False,
        "queue": "max",
    }
    fallback = next(step for step in job["steps"] if step.get("if") == "failure()")
    assert fallback["uses"].startswith("actions/github-script@")
    assert "ppu-ci-command:" in fallback["with"]["script"]
    assert "context.runId" in fallback["with"]["script"]


def test_notify_only_receives_label_events_with_minimum_permissions():
    config = workflow("notify-ci-authorized.yml")
    assert config[True] == {"pull_request_target": {"types": ["labeled"]}}
    job = config["jobs"]["notify"]
    assert "ready-run-all-tests" in job["if"]
    assert job["permissions"] == {
        "contents": "read",
        "issues": "write",
        "pull-requests": "read",
    }


def test_executor_uses_only_dispatch_with_a_stable_request_identity():
    config = workflow("ppu-pr-ci.yaml")
    assert set(config[True]) == {"workflow_dispatch"}
    inputs = config[True]["workflow_dispatch"]["inputs"]
    assert set(inputs) == {"pr_number", "head_sha", "requester", "comment_id"}
    assert all(
        item["type"] == "string" and item["required"] for item in inputs.values()
    )
    assert config["run-name"] == (
        "PPU PR #${{ inputs.pr_number }} @ ${{ inputs.head_sha }} "
        "by ${{ inputs.requester }} (comment ${{ inputs.comment_id }})"
    )
    assert config["permissions"] == {}
    assert config["cache-mode"] == "none"
    assert config["concurrency"] == {
        "group": "ppu-pr-ci-${{ inputs.pr_number }}",
        "cancel-in-progress": False,
        "queue": "max",
    }
    assert set(config["jobs"]) == {"validate", "ppu", "report"}


def test_executor_separates_read_only_pr_execution_and_always_reporting():
    jobs = workflow("ppu-pr-ci.yaml")["jobs"]
    ppu = jobs["ppu"]
    assert ppu["needs"] == "validate"
    assert ppu["uses"] == "./.github/workflows/build-ppu-wheels.yaml"
    assert ppu["permissions"] == {"contents": "read", "actions": "read"}
    assert "secrets" not in ppu
    assert ppu["with"] == {"pr_mode": True, "pr_head_sha": "${{ inputs.head_sha }}"}
    report = jobs["report"]
    assert report["if"] == "always()"
    assert report["needs"] == ["validate", "ppu"]
    assert all("always()" in step["if"] for step in report["steps"])
    step = next(step for step in report["steps"] if "run" in step)
    assert step["env"]["NEEDS"] == "${{ toJSON(needs) }}"
    assert "continue-on-error" not in step
    assert step["run"] == "python -m scripts.ci.ppu_ci_command report"


@pytest.mark.parametrize(
    "name", ["build-ppu-wheels.yaml", "ppu-pin-ut.yaml", "e2e-ppu.yaml"]
)
def test_untrusted_workflows_have_no_write_permissions_or_inherited_secrets(name):
    config = workflow(name)
    assert config["cache-mode"] == "none"
    assert config["permissions"] == {"contents": "read", "actions": "read"}
    for job in config["jobs"].values():
        assert "secrets" not in job
        assert all(value != "write" for value in job.get("permissions", {}).values())
        for step in checkouts(job) if "steps" in job else []:
            assert step["with"]["persist-credentials"] is False


@pytest.mark.parametrize(
    "name,job_ids",
    [
        ("build-ppu-wheels.yaml", ["build"]),
        ("ppu-pin-ut.yaml", ["pin-unit-tests"]),
        ("e2e-ppu.yaml", ["prepare", "e2e"]),
    ],
)
def test_every_pr_checkout_verifies_the_requested_sha_before_execution(name, job_ids):
    config = workflow(name)
    inputs = config[True]["workflow_call"]["inputs"]
    assert inputs["pr_mode"]["type"] == "boolean"
    assert inputs["pr_mode"]["default"] is False
    assert inputs["pr_head_sha"]["type"] == "string"
    assert inputs["pr_head_sha"]["default"] == ""
    for job_id in job_ids:
        job = config["jobs"][job_id]
        checkout = checkouts(job)[0]
        assert checkout["with"]["ref"] == EXPECTED_SHA
        index = job["steps"].index(checkout)
        verify = job["steps"][index + 1]
        assert verify["env"]["EXPECTED_SHA"] == EXPECTED_SHA
        assert (
            'test "$(git -c safe.directory="$PWD" rev-parse HEAD)" = "$EXPECTED_SHA"'
            == verify["run"]
        )
        guard = next(
            step
            for step in job["steps"]
            if step.get("name") == "Require disabled cache access"
        )
        assert job["steps"].index(guard) < index
        assert "process.env.ACTIONS_CACHE_MODE !== 'none'" in guard["with"]["script"]


@pytest.mark.parametrize(
    "name,job_id",
    [
        ("build-ppu-wheels.yaml", "build"),
        ("ppu-pin-ut.yaml", "pin-unit-tests"),
        ("e2e-ppu.yaml", "prepare"),
        ("e2e-ppu.yaml", "e2e"),
    ],
)
def test_sha_verification_allows_different_owner_but_rejects_wrong_sha(
    name, job_id, tmp_path
):
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    sha = "a" * 40
    (tmp_path / ".git" / "HEAD").write_text(sha + "\n")
    env = {
        **os.environ,
        "LC_ALL": "C",
        "GIT_TEST_ASSUME_DIFFERENT_OWNER": "1",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "safe.directory",
        "GIT_CONFIG_VALUE_0": "",
        "EXPECTED_SHA": sha,
    }
    untrusted = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert untrusted.returncode != 0
    assert "dubious ownership" in untrusted.stderr
    job = workflow(name)["jobs"][job_id]
    command = next(
        step["run"]
        for step in job["steps"]
        if step.get("name") == "Verify checked-out revision"
    )
    for expected, returncode in ((sha, 0), ("b" * 40, 1)):
        result = subprocess.run(
            ["sh", "-e", "-c", command],
            cwd=tmp_path,
            env={**env, "EXPECTED_SHA": expected},
            capture_output=True,
            text=True,
        )
        assert result.returncode == returncode, result.stderr
        assert not result.stderr


def test_build_conditions_preserve_automatic_and_manual_routes():
    config = workflow("build-ppu-wheels.yaml")
    assert set(config[True]) == {"pull_request", "workflow_dispatch", "workflow_call"}
    assert set(config[True]["workflow_call"]["inputs"]) == {"pr_mode", "pr_head_sha"}
    assert set(config[True]["workflow_dispatch"]["inputs"]) == {
        "existing_build_run_id",
        "build_image",
        "ops_image",
        "ops_test_scope",
        "vllm_ref",
        "pytorch_sail_arch",
    }
    jobs = config["jobs"]
    assert jobs["build"]["if"] == (
        "inputs.existing_build_run_id == '' && "
        "(inputs.pr_mode || github.event_name != 'workflow_dispatch')"
    )
    assert jobs["detect-ops-changes"]["if"] == (
        "${{ !inputs.pr_mode && github.event_name == 'workflow_dispatch' "
        "&& inputs.existing_build_run_id == '' }}"
    )
    assert jobs["pin-unit-tests"]["if"] == (
        "${{ !cancelled() && (needs.build.result == 'success' || "
        "(needs.build.result == 'skipped' && inputs.existing_build_run_id != '')) }}"
    )
    assert (
        jobs["smoke"]["if"] == "inputs.pr_mode || github.event_name == 'pull_request'"
    )
    assert (
        jobs["gate"]["if"]
        == "${{ always() && (inputs.pr_mode || github.event_name == 'pull_request') }}"
    )
    assert config["concurrency"]["cancel-in-progress"] == "${{ !inputs.pr_mode }}"
    assert "inputs.pr_mode" in config["concurrency"]["group"]
    assert "github.run_id" in config["concurrency"]["group"]
    for job_id in ("pin-unit-tests", "smoke"):
        assert jobs[job_id]["with"]["pr_mode"] == "${{ inputs.pr_mode || false }}"
        assert jobs[job_id]["with"]["pr_head_sha"] == "${{ inputs.pr_head_sha }}"


def test_pr_wheel_verification_uses_explicit_sha_and_is_not_skippable():
    pin = workflow("ppu-pin-ut.yaml")["jobs"]["pin-unit-tests"]
    verify = next(step for step in pin["steps"] if step.get("id") == "manifest")
    assert (
        verify["env"]["REQUIRE_PR_ARTIFACT"]
        == "${{ inputs.pr_mode || github.event_name == 'pull_request' }}"
    )
    assert verify["env"]["EXPECTED_SHA"] == EXPECTED_SHA
    assert '"--sail-commit", os.environ["EXPECTED_SHA"]' in verify["run"]
    e2e = workflow("e2e-ppu.yaml")["jobs"]["e2e"]
    verify = next(
        step
        for step in e2e["steps"]
        if step.get("name", "").startswith("Require the current PR artifact")
    )
    assert verify["if"] == "inputs.pr_mode || github.event_name == 'pull_request'"
    assert verify["env"]["EXPECTED_SHA"] == EXPECTED_SHA
    assert '--sail-commit "${EXPECTED_SHA}"' in verify["run"]


def test_retry_reuses_build_artifacts_but_keeps_pin_diagnostics_per_attempt():
    build = workflow("build-ppu-wheels.yaml")["jobs"]
    for job_id in ("build", "ops-build"):
        uploads = {
            step["with"]["name"]: step["with"]
            for step in build[job_id]["steps"]
            if step.get("uses") == "actions/upload-artifact@v4"
        }
        assert uploads["ppu-wheels-${{ github.run_id }}"]["overwrite"] is True
        assert (
            uploads["ppu-build-diagnostics-${{ github.run_id }}"]["overwrite"] is True
        )
    pin = workflow("ppu-pin-ut.yaml")["jobs"]["pin-unit-tests"]
    upload = next(
        step
        for step in pin["steps"]
        if step.get("uses") == "actions/upload-artifact@v4"
    )
    assert (
        upload["with"]["name"]
        == "ppu-pin-main-verified-${{ github.run_id }}-${{ github.run_attempt }}"
    )


def test_cpu_checks_still_run_automatically_with_read_only_checkout():
    config = workflow("ut.yaml")
    assert set(config[True]) == {"pull_request", "push", "workflow_dispatch"}
    assert config["permissions"] == {"contents": "read"}
    for job in config["jobs"].values():
        for checkout in checkouts(job):
            assert checkout["with"]["persist-credentials"] is False


def test_ops_partial_retry_downloads_diagnostics_from_successful_previous_attempt():
    build = workflow("build-ppu-wheels.yaml")["jobs"]
    producer = next(
        step["with"]["name"]
        for step in build["ops-build"]["steps"]
        if step.get("uses") == "actions/upload-artifact@v4"
        and step["with"]["name"].startswith("ppu-build-diagnostics-")
    )
    uploaded = producer.replace("${{ github.run_id }}", "100").replace(
        "${{ github.run_attempt }}", "1"
    )
    consumers = [
        build["select-ops-tests"],
        *workflow("_ppu-ops-tests.yaml")["jobs"].values(),
    ]
    for job in consumers:
        name = next(
            step["with"]["name"]
            for step in job["steps"]
            if step.get("uses") == "actions/download-artifact@v4"
            and step["with"].get("name", "").startswith("ppu-build-diagnostics-")
        )
        downloaded = (
            name.replace("${{ github.run_id }}", "100")
            .replace("${{ inputs.build_run_id }}", "100")
            .replace("${{ github.run_attempt }}", "2")
        )
        assert downloaded == uploaded
