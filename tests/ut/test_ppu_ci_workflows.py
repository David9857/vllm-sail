# SPDX-License-Identifier: Apache-2.0
import itertools
import json
import os
import re
import shlex
import subprocess
import sys
import zipfile
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


# GITHUB_TOKEN needs pull-requests: write to comment or react on a PR;
# issues: write alone returns 403 "Resource not accessible by integration".
@pytest.mark.parametrize(
    "name,job_id,expected",
    [
        (
            "run-ci-command.yml",
            "command",
            {
                "contents": "read",
                "actions": "write",
                "issues": "write",
                "pull-requests": "write",
                "statuses": "write",
            },
        ),
        (
            "notify-ci-authorized.yml",
            "notify",
            {"contents": "read", "issues": "write", "pull-requests": "write"},
        ),
        (
            "ppu-pr-ci.yaml",
            "report",
            {
                "contents": "read",
                "actions": "read",
                "statuses": "write",
                "issues": "write",
                "pull-requests": "write",
            },
        ),
    ],
)
def test_pr_commenting_jobs_can_write_pull_request_comments(name, job_id, expected):
    assert workflow(name)["jobs"][job_id]["permissions"] == expected


def test_executor_uses_only_dispatch_with_a_stable_request_identity():
    config = workflow("ppu-pr-ci.yaml")
    assert set(config[True]) == {"workflow_dispatch"}
    inputs = config[True]["workflow_dispatch"]["inputs"]
    required_inputs = {"pr_number", "head_sha", "requester", "comment_id"}
    assert set(inputs) == required_inputs
    for name in required_inputs:
        assert inputs[name]["type"] == "string" and inputs[name]["required"]
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
    assert ppu["with"] == {
        "pr_mode": True,
        "pr_head_sha": "${{ inputs.head_sha }}",
        "existing_build_run_id": "${{ needs.validate.outputs.existing_build_run_id }}",
        "wheel_source_sha": "${{ needs.validate.outputs.wheel_source_sha }}",
    }
    validate = jobs["validate"]
    assert validate["permissions"] == {
        "contents": "read",
        "actions": "read",
        "pull-requests": "read",
        "statuses": "write",
    }
    step = next(step for step in validate["steps"] if "run" in step)
    assert step["run"] == "python -m scripts.ci.ppu_ci_command validate"
    assert "continue-on-error" not in step
    assert validate["outputs"] == {
        key: "${{ steps." + step["id"] + ".outputs." + key + " }}"
        for key in ("existing_build_run_id", "wheel_source_sha")
    }
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


def test_build_conditions_preserve_on_demand_and_manual_routes():
    config = workflow("build-ppu-wheels.yaml")
    assert set(config[True]) == {"workflow_dispatch", "workflow_call"}
    assert set(config[True]["workflow_call"]["inputs"]) == {
        "pr_mode",
        "pr_head_sha",
        "existing_build_run_id",
        "wheel_source_sha",
    }
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
    assert jobs["gate"]["if"] == (
        "${{ always() && (inputs.pr_mode || github.event_name == 'pull_request') }}"
    )
    assert config["concurrency"]["cancel-in-progress"] == "${{ !inputs.pr_mode }}"
    assert "inputs.pr_mode" in config["concurrency"]["group"]
    assert "github.run_id" in config["concurrency"]["group"]
    for job_id in ("pin-unit-tests", "smoke"):
        assert jobs[job_id]["with"]["pr_mode"] == "${{ inputs.pr_mode || false }}"
        assert jobs[job_id]["with"]["pr_head_sha"] == "${{ inputs.pr_head_sha }}"
        assert (
            jobs[job_id]["with"]["wheel_source_sha"] == "${{ inputs.wheel_source_sha }}"
        )
        assert jobs[job_id]["with"]["build_run_id"] == (
            "${{ inputs.existing_build_run_id || github.run_id }}"
        )


def test_pr_wheel_verification_uses_explicit_sha_and_is_not_skippable():
    pin = workflow("ppu-pin-ut.yaml")["jobs"]["pin-unit-tests"]
    verify = next(step for step in pin["steps"] if step.get("id") == "manifest")
    assert (
        verify["env"]["REQUIRE_PR_ARTIFACT"]
        == "${{ inputs.pr_mode || github.event_name == 'pull_request' }}"
    )
    source = "${{ inputs.wheel_source_sha || inputs.pr_head_sha || github.sha }}"
    assert verify["env"]["WHEEL_SOURCE_SHA"] == source
    assert "EXISTING_BUILD_RUN_ID" not in verify["run"]
    e2e = workflow("e2e-ppu.yaml")["jobs"]["e2e"]
    verifications = [
        step for step in e2e["steps"] if "ppu_wheel_manifest.py" in step.get("run", "")
    ]
    assert len(verifications) == 1
    verify = verifications[0]
    assert verify["if"] == "inputs.pr_mode || github.event_name == 'pull_request'"
    assert verify["env"]["WHEEL_SOURCE_SHA"] == source
    for job in (pin, e2e):
        for step in job["steps"]:
            assert "${{ inputs.wheel_source_sha" not in step.get("run", "")


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


@pytest.mark.parametrize(
    "name", ["build-ppu-wheels.yaml", "ppu-pin-ut.yaml", "e2e-ppu.yaml"]
)
def test_wheel_source_is_optional_internal_input_only(name):
    triggers = workflow(name)[True]
    source = triggers["workflow_call"]["inputs"].get("wheel_source_sha")
    assert source is not None
    assert source["type"] == "string"
    assert source["default"] == ""
    assert not source.get("required", False)
    assert "wheel_source_sha" not in triggers.get("workflow_dispatch", {}).get(
        "inputs", {}
    )


def test_all_local_reusable_calls_use_declared_inputs():
    for path in WORKFLOWS.iterdir():
        if path.suffix not in {".yaml", ".yml"}:
            continue
        for job_id, job in workflow(path.name)["jobs"].items():
            target = job.get("uses", "")
            if not target.startswith("./.github/workflows/"):
                continue
            declared = workflow(Path(target).name)[True]["workflow_call"].get(
                "inputs", {}
            )
            supplied = job.get("with", {})
            assert supplied.keys() <= declared.keys(), (path.name, job_id)
            required = {key for key, value in declared.items() if value.get("required")}
            assert required <= supplied.keys(), (path.name, job_id)


def expression(value, context):
    """Evaluate only the Boolean/context subset used by these workflow guards."""
    value = value.removeprefix("${{").removesuffix("}}").strip()
    value = re.sub(
        r"\b(?:inputs|github|needs)(?:\.[\w-]+)+",
        lambda match: repr(context[match[0]]),
        value,
    )
    value = value.replace("always()", "True").replace("cancelled()", "False")
    value = value.replace("&&", " and ").replace("||", " or ")
    value = re.sub(r"!(?!=)", "not ", value)
    value = re.sub(r"\btrue\b", "True", value)
    value = re.sub(r"\bfalse\b", "False", value)
    return eval(value.strip(), {"__builtins__": {}}, {})


@pytest.mark.parametrize(
    "pr_mode,event",
    [
        (True, "workflow_dispatch"),
        (False, "workflow_dispatch"),
        (False, "pull_request"),
        ("", "workflow_dispatch"),  # Direct manual dispatch has no pr_mode input.
    ],
)
def test_e2e_concurrency_isolates_reuse_consumers_only_in_pr_mode(pr_mode, event):
    concurrency = workflow("e2e-ppu.yaml")["concurrency"]
    assert concurrency["cancel-in-progress"] is False
    assert "queue" not in concurrency
    refs = (
        ("refs/pull/47/merge", "refs/pull/48/merge")
        if event == "pull_request"
        else ("refs/heads/main", "refs/heads/release")
    )
    groups_by_ref = []
    for ref in refs:
        groups_by_attempt = []
        for attempt in (1, 2):
            groups = {}
            for run_id in ("901", "902", "903"):
                context = {
                    "inputs.pr_mode": pr_mode,
                    "inputs.build_run_id": "555",
                    "github.ref": ref,
                    "github.event_name": event,
                    "github.run_id": run_id,
                    "github.run_attempt": attempt,
                }
                groups[run_id] = re.sub(
                    r"\$\{\{.*?\}\}",
                    lambda match, context=context: str(expression(match[0], context)),
                    concurrency["group"],
                )
            # One running and two pending consumers must not share a PR-mode key.
            assert len(set(groups.values())) == (3 if pr_mode else 1)
            assert groups == {
                run_id: f"e2e-ppu-{ref}-{run_id if pr_mode else '555'}"
                for run_id in groups
            }
            groups_by_attempt.append(groups)
        assert groups_by_attempt[0] == groups_by_attempt[1]
        groups_by_ref.append(set(groups_by_attempt[0].values()))
    assert groups_by_ref[0].isdisjoint(groups_by_ref[1])


@pytest.fixture(scope="module")
def build_jobs():
    return workflow("build-ppu-wheels.yaml")["jobs"]


@pytest.mark.parametrize(
    "pr_mode,event",
    [
        (True, "workflow_dispatch"),
        (False, "pull_request"),
        (False, "workflow_dispatch"),
    ],
)
@pytest.mark.parametrize("build", ["success", "failure", "cancelled", "skipped"])
@pytest.mark.parametrize(
    "run_id,source", [("", ""), ("123", "a" * 40), ("999", ""), ("", "a" * 40)]
)
def test_consumers_require_build_or_valid_reuse(
    build_jobs, pr_mode, event, build, run_id, source
):
    context = {
        "inputs.pr_mode": pr_mode,
        "github.event_name": event,
        "inputs.existing_build_run_id": run_id,
        "inputs.wheel_source_sha": source,
        "needs.build.result": build,
    }
    manual = not pr_mode and event == "workflow_dispatch"
    reuse = bool(run_id) and (manual or (pr_mode and bool(source)))
    eligible = build == "success" or (build == "skipped" and reuse)
    for job_id in ("pin-unit-tests", "smoke"):
        assert bool(expression(build_jobs[job_id]["if"], context)) == (
            eligible and (job_id == "pin-unit-tests" or not manual)
        )
    assert bool(expression(build_jobs["build"]["if"], context)) == (
        not run_id and not manual
    )
    assert bool(expression(build_jobs["detect-ops-changes"]["if"], context)) == (
        manual and not run_id
    )
    # Gate must execute even after failed/cancelled/skipped dependencies.
    assert bool(expression(build_jobs["gate"]["if"], context)) == (not manual)


@pytest.mark.parametrize(
    "states",
    itertools.product(["success", "failure", "cancelled", "skipped"], repeat=3),
)
@pytest.mark.parametrize(
    "pr_mode,run_id,source",
    [
        (True, "", ""),
        (True, "123", "a" * 40),
        (True, "999", ""),
        (True, "", "a" * 40),
        (False, "123", "a" * 40),
    ],
)
def test_gate_executes_fail_closed_truth_table(
    build_jobs, monkeypatch, states, pr_mode, run_id, source
):
    step = build_jobs["gate"]["steps"][0]
    context = {
        "inputs.pr_mode": pr_mode,
        "inputs.existing_build_run_id": run_id,
        "inputs.wheel_source_sha": source,
    }
    for key, value in step["env"].items():
        if key != "JOB_RESULTS":
            resolved = expression(value, context)
            monkeypatch.setenv(
                key, str(resolved).lower() if isinstance(resolved, bool) else resolved
            )
    results = dict(
        zip(
            ("build", "pin-unit-tests", "smoke"),
            ({"result": state} for state in states),
            strict=False,
        )
    )
    monkeypatch.setenv("JOB_RESULTS", json.dumps(results))
    script = step["run"].removeprefix("python - <<'PYTHON'\n").removesuffix("PYTHON\n")
    try:
        exec(compile(script, "ppu-ci-gate", "exec"), {})
        passed = True
    except SystemExit as error:
        assert error.code
        passed = False
    expected = (
        states[0] == "success"
        or (states[0] == "skipped" and pr_mode and bool(run_id) and bool(source))
    ) and states[1:] == ("success", "success")
    assert passed is expected


@pytest.fixture
def wheel_artifacts(tmp_path):
    from scripts.ci.ppu_wheel_manifest import wheel_records

    wheels = tmp_path / "wheels"
    wheels.mkdir()
    for package in ("vllm", "vllm_sail"):
        with zipfile.ZipFile(
            wheels / f"{package}-1.0-py3-none-any.whl", "w"
        ) as archive:
            archive.writestr(
                f"{package}-1.0.dist-info/METADATA", f"Name: {package}\nVersion: 1.0\n"
            )
            if package == "vllm":
                archive.writestr("vllm/vllm-rs", "test executable")
                archive.writestr("vllm/_rust_tool_parser.abi3.so", "test extension")
            else:
                for module in ("_C", "_moe_C", "_upstream_C", "_upstream_moe_C"):
                    archive.writestr(f"vllm_sail/{module}.abi3.so", "test extension")
    (tmp_path / "scripts").symlink_to(ROOT / "scripts", target_is_directory=True)
    (tmp_path / ".github").mkdir()
    pin = (ROOT / ".github/vllm-main-verified.commit").read_text().strip()
    (tmp_path / ".github/vllm-main-verified.commit").write_text(pin + "\n")
    return wheels, {
        "schema": 1,
        "wheels": wheel_records(wheels),
        "commits": {"vllm": pin},
    }


@pytest.mark.parametrize(
    "name,job_id", [("ppu-pin-ut.yaml", "pin-unit-tests"), ("e2e-ppu.yaml", "e2e")]
)
@pytest.mark.parametrize("legacy_override", ["", "999"])
@pytest.mark.parametrize(
    "pr_mode,event,build_id,source,head,actual,wrong_pin,passes",
    [
        (True, "workflow_dispatch", "200", "", "b", "b", False, True),
        (True, "workflow_dispatch", "200", "", "b", "c", False, False),
        (True, "workflow_dispatch", "123", "a", "b", "a", False, True),
        (True, "workflow_dispatch", "123", "a", "b", "b", False, False),
        (True, "workflow_dispatch", "123", "a", "b", "c", False, False),
        (True, "workflow_dispatch", "123", "a", "b", "a", True, False),
        (True, "workflow_dispatch", "999", "", "b", "a", False, False),
        (True, "workflow_dispatch", "999", "", "b", "b", False, True),
        (True, "workflow_dispatch", "999", "", "", "c", False, False),
        (False, "pull_request", "200", "", "", "d", False, True),
        (False, "pull_request", "200", "", "", "c", False, False),
        (False, "pull_request", "200", "", "", "d", True, False),
        (False, "workflow_dispatch", "123", "", "", "a", False, True),
        (False, "workflow_dispatch", "123", "", "", "a", True, True),
    ],
)
def test_manifest_shell_checks_actual_source_and_current_pin(
    name,
    job_id,
    legacy_override,
    pr_mode,
    event,
    build_id,
    source,
    head,
    actual,
    wrong_pin,
    passes,
    wheel_artifacts,
    tmp_path,
):
    wheels, data = wheel_artifacts
    data["commits"]["vllm-sail"] = actual * 40
    if wrong_pin:
        data["commits"]["vllm"] = "e" * 40
    (wheels / "wheel-manifest.json").write_text(json.dumps(data))
    context = {
        "inputs.pr_mode": pr_mode,
        "github.event_name": event,
        "inputs.build_run_id": build_id,
        "github.run_id": "200",
        "inputs.wheel_source_sha": source * 40,
        "inputs.pr_head_sha": head * 40,
        "github.sha": "d" * 40,
    }
    env = {
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.defpath,
        "WHEELS_DIR": str(wheels),
        "GITHUB_OUTPUT": str(tmp_path / "output"),
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
        # Even a stale inherited variable must never bypass SAIL verification.
        "EXISTING_BUILD_RUN_ID": legacy_override,
    }
    steps = [
        step
        for step in workflow(name)["jobs"][job_id]["steps"]
        if "ppu_wheel_manifest.py" in step.get("run", "")
    ]
    executed = 0
    for step in steps:
        if "if" in step and not expression(step["if"], context):
            continue
        executed += 1
        for key, value in step.get("env", {}).items():
            resolved = expression(value, context)
            env[key] = str(resolved).lower() if isinstance(resolved, bool) else resolved
        result = subprocess.run(
            [
                "/bin/bash",
                "--noprofile",
                "--norc",
                "-eo",
                "pipefail",
                "-c",
                step["run"],
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert (result.returncode == 0) is passes, result.stdout + result.stderr
        if not passes:
            assert "source commit differs" in result.stderr
    assert executed == (
        1 if name == "ppu-pin-ut.yaml" or pr_mode or event == "pull_request" else 0
    )
    if name == "ppu-pin-ut.yaml" and passes:
        assert (
            tmp_path / "output"
        ).read_text() == f"matches={str(not wrong_pin).lower()}\n"


def scheduler_command(name):
    for job in workflow(name)["jobs"].values():
        for step in job.get("steps", []):
            if "ppu-scheduler-action" in step.get("uses", ""):
                return step["with"]["command"]
    raise AssertionError(f"No scheduler step in {name}")


# Privileged worker pods see every card on the host; without this binding the
# runtime defaults to cards 0..N-1, which may belong to other jobs.
@pytest.mark.parametrize(
    "name,script",
    [("e2e-ppu.yaml", "ppu_e2e_smoke.sh"), ("ppu-pin-ut.yaml", "ppu_pin_ut.sh")],
)
@pytest.mark.parametrize("allocated", ["3", "6,7", None, ""])
def test_device_workers_bind_allocated_cards_before_running(
    tmp_path, name, script, allocated
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "bash"
    fake.write_text(
        '#!/bin/sh\nprintf "%s %s\\n" "${CUDA_VISIBLE_DEVICES-<unset>}" "$*"\n'
    )
    fake.chmod(0o755)
    env = {"PATH": f"{bin_dir}{os.pathsep}{os.defpath}", "CUDA_VISIBLE_DEVICES": "0,1"}
    if allocated is not None:
        env["NVIDIA_VISIBLE_DEVICES"] = allocated
    command = scheduler_command(name).replace(
        "/workspace/source", shlex.quote(str(ROOT))
    )
    result = subprocess.run(
        ["/bin/bash", "--noprofile", "--norc", "-c", command],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if not allocated:
        assert result.returncode != 0
        assert "NVIDIA_VISIBLE_DEVICES" in result.stderr
        assert result.stdout == ""
    else:
        assert result.returncode == 0, result.stderr
        assert result.stdout == f"{allocated} scripts/ci/{script}\n"
