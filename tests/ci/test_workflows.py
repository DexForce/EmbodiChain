# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Contract tests for GitHub Actions workflows."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess
import sys
from zipfile import ZipFile

import pytest
import yaml

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _load_workflow(name: str) -> dict:
    workflow_path = _REPOSITORY_ROOT / ".github" / "workflows" / name
    return yaml.load(workflow_path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def test_automatic_pages_deployment_requires_existing_docs_artifact() -> None:
    jobs = _load_workflow("docs-pages.yml")["jobs"]

    assert "inspect-docs-artifact" in jobs

    inspection = jobs["inspect-docs-artifact"]
    deployment = jobs["deploy-from-artifact"]
    assert "environment" not in inspection
    assert inspection["outputs"]["exists"] == "${{ steps.artifact.outputs.exists }}"
    assert deployment["needs"] == "inspect-docs-artifact"
    deploy_condition = deployment["if"]
    assert deploy_condition.startswith("always() &&")
    assert "needs.inspect-docs-artifact.outputs.exists == 'true'" in deploy_condition
    assert "inputs.artifact_run_id != ''" in deploy_condition


def test_manual_docs_build_trusts_checkout_before_architecture_generation() -> None:
    jobs = _load_workflow("docs-pages.yml")["jobs"]
    steps = jobs["build-and-deploy"]["steps"]
    step_names = [step.get("name", "") for step in steps]

    trust_index = step_names.index("Trust checked-out repository")
    build_index = step_names.index("Build docs site")
    trust_command = steps[trust_index]["run"]

    assert trust_index < build_index
    assert (
        'git config --global --add safe.directory "$GITHUB_WORKSPACE"' in trust_command
    )


@pytest.mark.parametrize("job_name", ["deploy-from-artifact", "deploy-manual-build"])
def test_pages_paths_merge_under_deployment_lock_and_verify_versions(
    job_name: str,
) -> None:
    workflow = _load_workflow("docs-pages.yml")
    assert "'github-pages'" in workflow["concurrency"]["group"]
    assert workflow["concurrency"]["cancel-in-progress"] == "false"
    steps = workflow["jobs"][job_name]["steps"]
    names = [step.get("name", "") for step in steps]
    prepare = names.index("Preserve currently published versions")
    upload = names.index("Upload Pages artifact")
    publish = names.index("Deploy GitHub Pages")
    verify = names.index("Verify deployed artifact")
    assert prepare < upload < publish < verify
    assert "prepare_docs_deployment.py prepare" in steps[prepare]["run"]
    assert "prepare_docs_deployment.py verify" in steps[verify]["run"]
    assert steps[upload]["with"]["retention-days"] == "90"
    checkout = next(
        step for step in steps if step.get("uses", "").startswith("actions/checkout@")
    )
    assert checkout["with"]["ref"] == "${{ github.workflow_sha }}"


def test_release_dispatch_reports_child_without_timing_out_while_queued() -> None:
    steps = _load_workflow("docs-pages.yml")["jobs"]["redispatch-release"]["steps"]
    command = steps[0]["run"]
    assert "gh workflow run docs-pages.yml" in command
    assert "gh run watch" not in command
    assert "GITHUB_STEP_SUMMARY" in command


@pytest.mark.parametrize(
    ("workflow_name", "job_name", "step_name"),
    [
        ("main.yml", "test", "Install test package"),
        ("main.yml", "build", "Build docs"),
        ("docs-pages.yml", "build-and-deploy", "Build docs site"),
    ],
)
def test_package_install_uses_explicit_public_index_and_private_sources(
    workflow_name: str, job_name: str, step_name: str
) -> None:
    steps = _load_workflow(workflow_name)["jobs"][job_name]["steps"]
    install = next(step for step in steps if step.get("name") == step_name)
    commands = install["run"].split("pip install")[1:]

    # Each invocation must override any mirror inherited from the runner image.
    assert commands
    assert all("--index-url https://pypi.org/simple" in command for command in commands)
    package_install = next(
        command for command in commands if "--extra-index-url" in command
    )
    assert "--extra-index-url http://pyp.open3dv.site:2345/simple/" in package_install
    assert "--trusted-host pyp.open3dv.site" in package_install
    assert "--extra-index-url https://download.blender.org/pypi/" in package_install


@pytest.mark.parametrize(
    ("workflow_name", "job_name", "step_name"),
    [
        ("main.yml", "build", "Build docs"),
        ("main.yml", "test_docs", "Install docs test dependencies"),
        ("docs-pages.yml", "build-and-deploy", "Build docs site"),
    ],
)
def test_docs_dependencies_override_inherited_pip_index(
    workflow_name: str, job_name: str, step_name: str
) -> None:
    steps = _load_workflow(workflow_name)["jobs"][job_name]["steps"]
    install = next(step for step in steps if step.get("name") == step_name)
    commands = install["run"].replace("\\\n", " ").splitlines()
    arguments = shlex.split(
        next(command for command in commands if "docs/requirements.txt" in command)
    )
    assert "--isolated" in arguments
    assert arguments[arguments.index("--index-url") + 1] == "https://pypi.org/simple"


def test_simulator_test_job_installs_node_before_workflow_contract_tests() -> None:
    steps = _load_workflow("main.yml")["jobs"]["test"]["steps"]
    setup_index = next(
        i
        for i, step in enumerate(steps)
        if step.get("uses", "").startswith("actions/setup-node@")
    )
    tests_index = next(
        i for i, step in enumerate(steps) if step.get("name") == "Run selected tests"
    )
    assert steps[setup_index]["with"]["node-version"] == "22"
    assert setup_index < tests_index
    capture_index = next(
        i
        for i, step in enumerate(steps)
        if step.get("name") == "Preserve Node path for login shells"
    )
    assert setup_index < capture_index < tests_index
    assert "GITHUB_ENV" in steps[capture_index]["run"]
    assert 'export PATH="${CI_NODE_BIN_DIR}:${PATH}"' in steps[tests_index]["run"]


def test_test_job_restores_git_access_before_running_selected_tests(
    tmp_path: Path,
) -> None:
    """Run the workflow preparation against a checkout with untrusted ownership."""
    steps = _load_workflow("main.yml")["jobs"]["test"]["steps"]
    tests = next(step for step in steps if step.get("name") == "Run selected tests")
    assert tests["shell"] == "bash -l {0}"
    preparation = tests["run"].split("python scripts/ci/run_test_plan.py", 1)[0]
    preparation = preparation.replace("source activate py311", ":").replace(
        "node --version", ":"
    )
    repository = tmp_path / "checkout with spaces"
    repository.mkdir()
    config = tmp_path / "gitconfig"
    config.write_text("")
    environment = os.environ.copy()
    environment.update(
        GIT_CONFIG_GLOBAL=str(config),
        GIT_CONFIG_NOSYSTEM="1",
        GITHUB_WORKSPACE=str(repository),
        CI_NODE_BIN_DIR="",
    )
    for args in (
        ["init", "-q"],
        [
            "-c",
            "user.name=CI Test",
            "-c",
            "user.email=ci@example.org",
            "commit",
            "--allow-empty",
            "-qm",
            "fixture",
        ],
    ):
        subprocess.run(["git", *args], cwd=repository, env=environment, check=True)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, env=environment, text=True
    ).strip()
    environment["GIT_TEST_ASSUME_DIFFERENT_OWNER"] = "1"
    untrusted = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert untrusted.returncode != 0
    assert "dubious ownership" in untrusted.stderr
    prepared = subprocess.run(
        ["bash", "-e", "-c", preparation],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert prepared.returncode == 0, prepared.stdout + prepared.stderr
    assert (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, env=environment, text=True
        ).strip()
        == revision
    )


@pytest.mark.parametrize(
    "invalid_contents",
    [
        None,
        "missing_robots",
        "missing_entry_point",
        "wrong_entry_point_group",
        "foreign_package",
        "bytecode",
    ],
)
def test_release_guard_validates_bundled_robot_catalog(
    tmp_path: Path, invalid_contents: str | None
) -> None:
    """Execute the release guard against wheels, including invalid artifacts."""
    steps = _load_workflow("main.yml")["jobs"]["release-build"]["steps"]
    guard = next(
        step for step in steps if step.get("name") == "Validate wheel contents"
    )
    script = guard["run"].split("python - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    version = "0.1.0"
    (tmp_path / "VERSION").write_text(version)
    task_config = (
        "embodichain_tasks/configs/tasks/manipulation/tableware/pour_water/env.yaml"
    )
    source = tmp_path / "extensions" / task_config
    source.parent.mkdir(parents=True)
    source.write_text("physics: default\n")
    entry_points = (
        "[embodichain.tasks]\nembodichain_tasks = embodichain_tasks\n\n"
        "[embodichain.robots]\nofficial = embodichain_robots._catalog:ROBOT_CONFIGS\n"
    )
    if invalid_contents == "missing_entry_point":
        entry_points = entry_points.split("[embodichain.robots]", 1)[0]
    elif invalid_contents == "wrong_entry_point_group":
        entry_points = entry_points.replace("[embodichain.robots]", "[other.robots]")
    members = {
        "embodichain/VERSION": version,
        "embodichain/gen_sim/simready_pipeline/configs/gen_config.json": "{}",
        "embodichain_tasks/__init__.py": "",
        task_config: source.read_text(),
        "embodichain_robots/__init__.py": "",
        "embodichain_robots/_catalog.py": "ROBOT_CONFIGS = {}\n",
        f"embodichain-{version}.dist-info/METADATA": f"Version: {version}\n",
        f"embodichain-{version}.dist-info/entry_points.txt": entry_points,
    }
    if invalid_contents == "missing_robots":
        members = {
            path: data
            for path, data in members.items()
            if not path.startswith("embodichain_robots/")
        }
    elif invalid_contents == "foreign_package":
        members["foreign/__init__.py"] = ""
    elif invalid_contents == "bytecode":
        members["embodichain_robots/__pycache__/_catalog.cpython-311.pyc"] = ""
    dist = tmp_path / "dist"
    dist.mkdir()
    with ZipFile(dist / f"embodichain-{version}-py3-none-any.whl", "w") as wheel:
        for path, data in members.items():
            wheel.writestr(path, data)

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )

    if invalid_contents is None:
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode != 0
        assert "Invalid wheel contents" in result.stderr
