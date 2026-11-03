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

"""Showcase acceptance must distinguish evidence from declarations and survive relocation."""

from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import torch
from examples.expansion.engine.showcase._cases import (
    dump,
    recover_portability,
    worker_environment,
    stop_process_tree,
)
from examples.expansion.engine.showcase._report import audit


def test_attempted_preflight_failure_is_not_reported_as_unattempted(tmp_path):
    case = {
        "id": "motion",
        "job": "configs/job.motion.yaml",
        "tags": ["motion"],
        "required": True,
    }
    dump(tmp_path / "cases.json", {"cases": [case]})
    dump(
        tmp_path / "results/job.motion.json",
        {"ok": False, "error": "source artifact hash differs"},
    )
    audit(tmp_path)
    report = json.loads((tmp_path / "acceptance.json").read_text())
    assert report["checks"][0]["status"] == "failed"
    assert report["checks"][0]["reason"] == "source artifact hash differs"
    assert not report["acceptance_passed"]


def test_nominal_commits_cannot_qualify_a_requested_motion_operator(
    tmp_path, monkeypatch
):
    import embodichain.lab.expansion as expansion
    import embodichain.lab.expansion.visual as visual

    case = {
        "id": "motion",
        "job": "configs/job.motion.yaml",
        "tags": ["motion", "via_points"],
        "required": True,
        "source": "ur5",
        "backend": "default",
    }
    dump(tmp_path / "cases.json", {"cases": [case]})
    dump(tmp_path / "jobs/motion/job.json", {})
    receipt = SimpleNamespace(storage_id="fixture", commit_id="commit")
    monkeypatch.setattr(
        expansion,
        "ArtifactEpisodeSink",
        lambda _: SimpleNamespace(inspect=lambda: (receipt,)),
    )
    monkeypatch.setattr(
        expansion, "inspect_expansion_job", lambda _: {"complete": True}
    )
    monkeypatch.setattr(visual, "inspect_observation_variants", lambda _: ())
    episode = SimpleNamespace(
        timestamps=torch.tensor([0.0, 0.1, 0.2]),
        actions=torch.zeros(2, 1),
        observations={"joint_positions": torch.zeros(3, 1)},
        validation=SimpleNamespace(accepted=True),
        metadata={
            "recipe_spec": {"context": {"ordinal": 0}},
            "recipe_id": "recipe",
            "expansion": [
                {
                    "trajectory_selected": 0,
                    "trajectory_variants": [
                        {"variant_ordinal": 0, "spatial_operator": "none"},
                        {"variant_ordinal": 1, "spatial_operator": "via_points"},
                    ],
                }
            ],
        },
    )
    monkeypatch.setattr(expansion, "load_expansion_episode", lambda _: episode)
    audit(tmp_path)
    report = json.loads((tmp_path / "acceptance.json").read_text())
    assert report["checks"][0]["status"] == "failed"
    assert "No admitted via_points" in report["checks"][0]["reason"]


def test_relocation_intent_restores_hidden_job_and_dataset(tmp_path):
    (tmp_path / "diagnostics/hidden-job/exports").mkdir(parents=True)
    (tmp_path / "diagnostics/moved").mkdir()
    (tmp_path / "diagnostics/moved/evidence").write_text("frozen")
    dump(
        tmp_path / "diagnostics/portability-move.json",
        {
            "job": "jobs/job",
            "dataset": "jobs/job/exports/dataset",
            "destination": "diagnostics/moved",
            "hidden": "diagnostics/hidden-job",
        },
    )
    (tmp_path / "jobs").mkdir()
    recover_portability(tmp_path)
    assert (tmp_path / "jobs/job/exports/dataset/evidence").read_text() == "frozen"
    assert not (tmp_path / "diagnostics/portability-move.json").exists()
    recover_portability(tmp_path)


def test_worker_can_launch_from_outside_checkout_without_inherited_pythonpath(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    result = subprocess.run(
        [sys.executable, "-m", "examples.expansion.engine.showcase._worker", "--help"],
        cwd=tmp_path,
        env=worker_environment(),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "collect" in result.stdout and "replay" in result.stdout


def test_cancellation_reaps_owned_descendant_processes(tmp_path):
    import psutil
    import time

    marker = tmp_path / "child.pid"
    script = "import subprocess,sys,time; from pathlib import Path; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);Path(sys.argv[1]).write_text(str(p.pid));time.sleep(60)"
    process = subprocess.Popen([sys.executable, "-c", script, str(marker)])
    try:
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists()
        child = int(marker.read_text())
        stop_process_tree(process)
        assert process.poll() is not None
        assert (
            not psutil.pid_exists(child)
            or psutil.Process(child).status() == psutil.STATUS_ZOMBIE
        )
    finally:
        if process.poll() is None:
            stop_process_tree(process, force=True)


def test_recipe_catalogue_survives_json_roundtrip_and_capacity_change(
    tmp_path, monkeypatch
):
    from embodichain.lab.expansion.recipes import (
        ExpansionRecipe,
        ExpansionContext,
        BatchCompatibility,
    )
    import embodichain.lab.expansion.host as native
    from examples.expansion.engine.showcase._worker import install_host_observer

    class Host:
        def __init__(self, capacity):
            self.root = tmp_path
            self.capacity = capacity

        def recipe(self, ordinal):
            return ExpansionRecipe(
                ExpansionContext("task_program", "source", 7, ordinal),
                BatchCompatibility(
                    "robot",
                    ("joint",),
                    "topology",
                    "program",
                    "default",
                    0.04,
                    "schema",
                    "world",
                ),
                (),
            )

        def close(self):
            pass

    monkeypatch.setattr(native, "GymExpansionHost", Host)
    install_host_observer()
    native.GymExpansionHost(1)
    original = (tmp_path / "recipe-catalog.json").read_bytes()
    native.GymExpansionHost(8)
    assert (tmp_path / "recipe-catalog.json").read_bytes() == original
    assert len(json.loads(original)["recipes"]) == 33


def test_unmarked_truncated_preparation_recovers_but_marked_corruption_is_rejected(
    tmp_path,
):
    from examples.expansion.engine.showcase._cases import ensure_prepared
    import pytest

    root = tmp_path / "carton"
    root.mkdir()
    profile = root / "profile.measured.single.yaml"
    import yaml

    profile.write_text("source_revision: [")
    attempts = []

    def prepare():
        attempts.append(1)
        root.mkdir()
        profile.write_text('{"source_revision":"pinned"}')

    def validate():
        return yaml.safe_load(profile.read_text())

    ensure_prepared(root, prepare, validate)
    assert attempts == [1]
    assert len(list(tmp_path.glob("carton.interrupted-*"))) == 1
    ensure_prepared(root, prepare, validate)
    assert attempts == [1]
    profile.write_text("source_revision: [")
    with pytest.raises(yaml.YAMLError):
        ensure_prepared(root, prepare, validate)
    assert attempts == [1]


def test_outer_timeout_kills_a_descendant_that_ignores_sigterm(tmp_path, monkeypatch):
    import psutil
    from examples.expansion.engine.showcase import __main__ as runner

    marker = tmp_path / "ignored-term.pid"
    child = (
        "import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)"
    )
    script = "import subprocess,sys,time;from pathlib import Path;p=subprocess.Popen([sys.executable,'-c',sys.argv[2]]);Path(sys.argv[1]).write_text(str(p.pid));time.sleep(60)"
    original = subprocess.Popen

    def launch(command, **kwargs):
        return original([sys.executable, "-c", script, str(marker), child], **kwargs)

    monkeypatch.setattr(runner.subprocess, "Popen", launch)
    try:
        assert runner.run_worker(tmp_path, "probe", timeout=1) == 124
        pid = int(marker.read_text())
        assert (
            not psutil.pid_exists(pid)
            or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
        )
    finally:
        if marker.exists():
            try:
                psutil.Process(int(marker.read_text())).kill()
            except psutil.NoSuchProcess:
                pass


def test_deployment_resolves_all_official_references_outside_checkout(
    tmp_path, monkeypatch
):
    import yaml
    from examples.expansion.engine.showcase._cases import ENGINE, absolute_deployment

    value = yaml.safe_load((ENGINE / "task.default.yaml").read_text())
    value["task_program"]["program"] = str(
        (ENGINE.parent / "scene_robot_feasibility/program.yaml").resolve()
    )
    source = tmp_path / "task.yaml"
    dump(source, value)
    monkeypatch.chdir(tmp_path)
    resolved = absolute_deployment(source)
    assert Path(resolved["environment"]["component"]).is_file()
    assert Path(resolved["embodiment"]["component"]).is_file()
    assert all(Path(v).is_file() for v in resolved["task_program"].values())


def test_generated_newton_clock_and_resume_preserve_frozen_inputs(tmp_path):
    import yaml
    from examples.expansion.engine.showcase._cases import (
        ENGINE,
        absolute_deployment,
        build_cases,
    )

    prepared = tmp_path / "inputs/carton"
    prepared.mkdir(parents=True)
    task = absolute_deployment(ENGINE / "task.default.yaml")
    task["environment"]["component"] = str(prepared / "environment.yaml")
    dump(prepared / "task.measured.yaml", task)
    dump(
        prepared / "environment.yaml",
        {
            "environment_id": "clock_fixture",
            "physics": "default",
            "physics_config": {"physics_dt": 1 / 240},
            "env": {"target_control_frequency": 30, "sim_steps_per_control": 8},
        },
    )
    dump(prepared / "profile.measured.single.yaml", {"recipes": ["nominal"]})
    build_cases(tmp_path, "full")
    environment = yaml.safe_load(
        (tmp_path / "configs/carton.environment.newton.yaml").read_text()
    )
    assert environment["physics"] == "newton"
    assert environment["physics_config"]["physics_dt"] == 1 / 240
    assert environment["env"]["target_control_frequency"] == 30
    job_path = tmp_path / "configs/job.ur5-baseline.yaml"
    job = yaml.safe_load(job_path.read_text())
    job["max_attempts"] = 99
    dump(job_path, job)
    visual = tmp_path / "configs/visual.ur5.yaml"
    visual.write_text(visual.read_text() + "# frozen input\n")
    frozen = visual.read_bytes()
    inventory = json.loads((tmp_path / "cases.json").read_text())
    extra = {**inventory["cases"][0], "id": "wide-scene", "job": "configs/custom.yaml"}
    inventory["cases"].append(extra)
    dump(tmp_path / "cases.json", inventory)
    build_cases(tmp_path, "full", preserve=True)
    assert json.loads((tmp_path / "cases.json").read_text())["cases"][-1] == extra
    assert visual.read_bytes() == frozen
    assert yaml.safe_load(job_path.read_text())["max_attempts"] == 99


def test_browser_comparison_selects_actual_augmentation_and_preserves_units():
    import shutil
    import pytest

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed to verify the browser comparison logic")
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            node,
            str(Path(__file__).with_name("showcase_comparison.cjs")),
            str(root / "examples/expansion/engine/showcase/comparison.js"),
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_preview_joint_types_are_explicit_and_survive_asset_removal(tmp_path):
    from examples.expansion.engine.showcase._report import preview_joint_types

    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        '<robot name="test"><joint name="arm" type="continuous"/><joint name="finger" type="prismatic"/></robot>'
    )
    dump(
        tmp_path / "jobs/example/job.json",
        {"resolved_deployment": {"robot": {"urdf_cfg": {"fpath": str(urdf)}}}},
    )
    episode = {"model": "model", "metadata": {"joint_names": ["arm", "finger"]}}
    assert preview_joint_types(tmp_path, {"id": "example"}, episode) == [
        "continuous",
        "prismatic",
    ]
    urdf.unlink()
    assert preview_joint_types(tmp_path, {"id": "example"}, episode) == [
        "continuous",
        "prismatic",
    ]
    unknown = {"model": "other", "metadata": {"joint_names": ["unknown"]}}
    assert preview_joint_types(tmp_path, {"id": "missing"}, unknown) == [None]
