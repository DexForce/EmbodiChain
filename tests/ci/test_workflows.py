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

from pathlib import Path
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


def test_test_package_install_uses_explicit_public_index_and_private_sources() -> None:
    steps = _load_workflow("main.yml")["jobs"]["test"]["steps"]
    install = next(step for step in steps if step.get("name") == "Install test package")
    commands = install["run"].split("pip install")[1:]

    # Each invocation must override any mirror inherited from the runner image.
    assert commands
    assert all("--index-url https://pypi.org/simple" in command for command in commands)
    package_install = next(command for command in commands if "${package}" in command)
    assert "--extra-index-url http://pyp.open3dv.site:2345/simple/" in package_install
    assert "--trusted-host pyp.open3dv.site" in package_install
    assert "--extra-index-url https://download.blender.org/pypi/" in package_install


def test_curobo_source_is_bundled_before_self_hosted_installation() -> None:
    jobs = _load_workflow("main.yml")["jobs"]
    planner = jobs["test_plan"]
    assert planner["runs-on"] == "ubuntu-latest"
    source = next(
        step
        for step in planner["steps"]
        if step.get("with", {}).get("repository") == "NVlabs/curobo"
    )
    assert source["with"]["ref"] == "refs/tags/v0.8.0"
    assert source["with"]["persist-credentials"] == "false"
    build = next(
        step
        for step in planner["steps"]
        if step.get("name") == "Build cuRobo source distribution"
    )
    upload = next(
        step
        for step in planner["steps"]
        if step.get("with", {}).get("name") == "curobo-sdist"
    )
    for step in (source, build, upload):
        assert step["if"] == "steps.plan.outputs.needs_curobo == 'true'"
    assert "python -m build --sdist" in build["run"]
    assert source["with"]["path"] in build["run"]
    assert upload["with"]["path"] in build["run"]
    assert upload["with"]["if-no-files-found"] == "error"

    steps = jobs["test"]["steps"]
    download = next(
        step for step in steps if step.get("with", {}).get("name") == "curobo-sdist"
    )
    assert download["if"] == "env.NEEDS_CUROBO == 'true'"
    assert download["uses"].startswith("actions/download-artifact@")
    install = next(step for step in steps if step.get("name") == "Install test package")
    assert steps.index(download) < steps.index(install)
    command = install["run"]
    assert "git+https://github.com/NVlabs/curobo" not in command
    assert (
        "nvidia-curobo[cu12] @ file://${GITHUB_WORKSPACE}/"
        f"{download['with']['path']}/nvidia_curobo-0.8.0.tar.gz"
    ) in command
    # Installation errors must still fail the test job and its required gate.
    assert "set -euo pipefail" in command
    assert install.get("continue-on-error", "false") == "false"
    assert "test" in jobs["test_gate"]["needs"]


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
