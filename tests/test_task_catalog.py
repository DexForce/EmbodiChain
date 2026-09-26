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

from __future__ import annotations

import json
from pathlib import Path

import pytest

from embodichain.cli import _task_catalog as catalog
from embodichain.cli.main import main


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def task_root(tmp_path: Path) -> Path:
    task = tmp_path / "manipulation" / "pick"
    write(
        task / "task.a.yaml",
        {
            "id": "Pick-A-v1",
            "physics": "default",
            "task_program": {},
            "robot": {"uid": "Arm"},
        },
    )
    write(task / "task.b.json", {"id": "Pick-B-v1", "physics": "newton"})
    write(
        task / "agents" / "ppo.json",
        {"trainer": {"gym_config": "demo/configs/tasks/manipulation/pick/task.a.yaml"}},
    )
    write(
        task / "catalog.yaml",
        {
            "task_key": "pick",
            "title": "Pick <cube>",
            "summary": "Move & hold",
            "tags": ["cube"],
            "default_deployment": "a",
            "deployments": {
                "a": {"config": "task.a.yaml"},
                "b": {"config": "task.b.json"},
            },
        },
    )
    return tmp_path


def test_only_linked_deployment_acquires_rl(task_root: Path) -> None:
    (task,) = catalog._load_catalog({"demo": task_root})
    assert task.qualified_key == "demo:pick"
    a, b = task.deployments
    assert a.capabilities == {catalog._TASK_PROGRAM, catalog._RL}
    assert b.capabilities == set()
    assert a.agent_refs == ("demo/configs/tasks/manipulation/pick/agents/ppo.json",)
    assert (a.physics, b.physics) == ("default", "newton")
    assert a.embodiments == ("Arm",)


def test_legacy_config_and_lightweight_rl_are_preserved(task_root: Path) -> None:
    (task_root / "manipulation/pick/catalog.yaml").unlink()
    write(
        task_root / "classic_control/point/agents/ppo.json",
        {"trainer": {"learning_env": {"name": "PointRL"}}},
    )
    tasks = catalog._load_catalog({"demo": task_root})
    assert {task.key for task in tasks} == {"pick", "point"}
    pick = catalog._select_task(tasks, "pick")
    assert {d.env_id for d in pick.deployments} == {"Pick-A-v1", "Pick-B-v1"}
    point = catalog._select_task(tasks, "demo:point").deployments[0]
    assert point.capabilities == {catalog._RL}
    assert point.config_ref is None
    assert point.agent_refs


@pytest.mark.parametrize(
    "change,match",
    [
        ({"id": "NotAnEnv"}, "Unknown"),
        ({"default_deployment": "missing"}, "default_deployment"),
        ({"tags": "bad"}, "tags"),
        ({"deployments": {"a": {"config": "missing.yaml"}}}, "missing"),
        ({"deployments": {"a": {"config": "../outside.yaml"}}}, "relative"),
        ({"deployments": {"a": {"config": "task.a.yaml", "rl": True}}}, "Unknown"),
    ],
)
def test_strict_metadata(task_root: Path, change: dict, match: str) -> None:
    path = task_root / "manipulation/pick/catalog.yaml"
    value = json.loads(path.read_text())
    value.update(change)
    write(path, value)
    with pytest.raises((ValueError, TypeError), match=match):
        catalog._load_catalog({"demo": task_root})


def test_duplicate_keys_fail_and_cross_package_keys_require_qualification(
    task_root: Path,
) -> None:
    tasks = catalog._load_catalog({"demo": task_root, "other": task_root})
    with pytest.raises(ValueError, match="ambiguous"):
        catalog._select_task(tasks, "pick")
    assert catalog._select_task(tasks, "other:pick").package == "other"
    write(
        task_root / "other/pick/catalog.yaml",
        json.loads((task_root / "manipulation/pick/catalog.yaml").read_text()),
    )
    write(task_root / "other/pick/task.a.yaml", {"id": "Other-A"})
    write(task_root / "other/pick/task.b.json", {"id": "Other-B"})
    with pytest.raises(ValueError, match="Duplicate"):
        catalog._load_catalog({"demo": task_root})


def test_show_and_filtered_gallery_use_static_records(
    task_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "gallery.html"
    main(["show-task", "demo:pick", "--config-root", f"demo={task_root}"])
    detail = capsys.readouterr().out
    assert "Pick <cube>" in detail
    assert "embodichain run-env --gym_config" in detail
    assert "Qualification: unavailable" in detail
    main(
        [
            "list-task",
            "--category",
            "manipulation",
            "--config-root",
            f"demo={task_root}",
            "--export-html",
            str(output),
        ]
    )
    html = output.read_text()
    assert "Pick &lt;cube&gt;" in html and "Move &amp; hold" in html
    assert "<cube>" not in html
    assert "task.a.yaml" in html and "task.b.json" in html
    assert (task_root / "manipulation/pick/task.a.yaml").as_uri() in html
    main(
        [
            "list-task",
            "--category",
            "classic_control",
            "--config-root",
            f"demo={task_root}",
            "--export-html",
            str(output),
        ]
    )
    assert "Pick &lt;cube&gt;" not in output.read_text()


def test_environment_component_supplies_backend(task_root: Path) -> None:
    write(
        task_root / "manipulation/pick/task.a.yaml",
        {"id": "Pick-A-v1", "environment": {"component": "env.yaml"}},
    )
    write(task_root / "manipulation/pick/env.yaml", {"physics": "newton"})
    (task,) = catalog._load_catalog({"demo": task_root})
    assert task.deployments[0].physics == "newton"


def test_duplicate_authored_fields_fail(task_root: Path) -> None:
    path = task_root / "manipulation/pick/catalog.yaml"
    path.write_text("task_key: pick\ntask_key: other\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        catalog._load_catalog({"demo": task_root})


def test_metadata_rejects_non_runnable_config(task_root: Path) -> None:
    write(task_root / "manipulation/pick/task.a.yaml", {"physics": "default"})
    with pytest.raises(ValueError, match="runnable"):
        catalog._load_catalog({"demo": task_root})


def test_runtime_only_and_registered_capabilities_are_preserved(
    task_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace
    from embodichain.lab.gym.utils import registration
    from embodichain.learning.rl import env

    runtime_cls = type("RuntimeEnv", (), {"__module__": "demo.special.runtime"})
    spec = SimpleNamespace(
        cls=runtime_cls,
        task_program_registration=None,
        task_program_adapter_factory=None,
        supports_rl=True,
    )
    monkeypatch.setattr(registration, "REGISTERED_ENVS", {"Runtime-v1": spec})
    monkeypatch.setattr(
        env, "get_registered_learning_env_names", lambda: ["UnconfiguredRL"]
    )
    monkeypatch.setattr(catalog, "_implements_handwritten_demo", lambda cls: True)
    tasks = catalog._load_catalog({"demo": task_root})
    catalog._augment_runtime(tasks, ["demo"])
    runtime = catalog._select_task(tasks, "demo:runtime").deployments[0]
    assert runtime.config_ref is None
    assert runtime.capabilities == {catalog._HANDWRITTEN_DEMO, catalog._RL}
    assert catalog._select_task(tasks, "runtime:UnconfiguredRL").deployments[
        0
    ].capabilities == {catalog._RL}


def test_official_catalogs_resolve_real_deployments() -> None:
    root = Path(__file__).resolve().parents[1] / "embodichain_tasks/configs/tasks"
    tasks = catalog._load_catalog({"embodichain_tasks": root})
    pick = catalog._select_task(tasks, "repeated_pick_place")
    assert {d.name for d in pick.deployments} >= {
        "franka",
        "ur5",
        "franka_newton",
        "ur5_newton",
        "franka_rlinf",
        "franka_rlinf_joint",
        "franka_rlinf_expert",
    }
    deployments = {deployment.name: deployment for deployment in pick.deployments}
    assert deployments["franka_rlinf"].config_ref.endswith("task.franka.rlinf.yaml")
    assert deployments["franka_rlinf_joint"].config_ref.endswith(
        "task.franka.rlinf_joint.yaml"
    )
    assert deployments["franka_rlinf_expert"].config_ref.endswith(
        "task.franka.rlinf_expert.yaml"
    )
    assert catalog._select_task(tasks, "push_cube").deployments[0].capabilities == {
        catalog._RL
    }
    assert (
        catalog._select_task(tasks, "stack_cups").deployments[0].capabilities == set()
    )
    for task in tasks:
        for deployment in task.deployments:
            if deployment.config_ref:
                assert deployment.resource.is_file()


def test_offline_gallery_explains_runtime_capability_limit(
    task_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "offline.html"
    main(
        [
            "list-task",
            "--config-root",
            f"demo={task_root}",
            "--export-html",
            str(output),
        ]
    )
    assert "Runtime registrations were not inspected" in output.read_text()
    main(["show-task", "pick", "--config-root", f"demo={task_root}"])
    assert "Runtime registrations were not inspected" in capsys.readouterr().out


def test_same_env_id_does_not_spread_inferred_rl(task_root: Path) -> None:
    write(
        task_root / "manipulation/pick/task.b.json",
        {"id": "Pick-A-v1", "physics": "newton"},
    )
    (task,) = catalog._load_catalog({"demo": task_root})
    assert catalog._RL in task.deployments[0].capabilities
    assert catalog._RL not in task.deployments[1].capabilities


def test_authored_nested_deployment_stays_in_one_logical_task(task_root: Path) -> None:
    directory = task_root / "manipulation/pick"
    value = json.loads((directory / "catalog.yaml").read_text())
    value["deployments"]["a"]["config"] = "deployments/arm.yaml"
    write(directory / "catalog.yaml", value)
    write(directory / "deployments/arm.yaml", {"id": "Nested-v1", "physics": "default"})
    tasks = catalog._load_catalog({"demo": task_root})
    assert len(tasks) == 1
    assert tasks[0].deployments[0].env_id == "Nested-v1"


def test_official_discovery_prefers_core_colocated_worktree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stale_package = tmp_path / "stale"
    (stale_package / "tasks").mkdir(parents=True)
    monkeypatch.setattr(
        catalog.importlib.resources, "files", lambda name: stale_package
    )
    expected = Path(__file__).resolve().parents[1] / "embodichain_tasks/configs/tasks"
    assert catalog._task_config_roots(("embodichain_tasks",)) == (expected,)


def test_validation_links_independent_physical_failure(
    task_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = task_root / "manipulation/pick"
    value = json.loads((directory / "catalog.yaml").read_text())
    value["deployments"]["a"]["validation"] = "validation/report.json"
    write(directory / "catalog.yaml", value)
    write(
        directory / "validation/report.json",
        {
            "status": "completed",
            "code": {"revision": "abc123", "dirty": True},
            "results": [
                {
                    "action_source": "task_program",
                    "execution_outcome": {"status": "completed"},
                    "physical_outcome": {"success": False},
                }
            ],
        },
    )
    (task,) = catalog._load_catalog({"demo": task_root})
    html = catalog._render_html([task])
    assert "execution=completed, physical=false" in html
    assert "abc123" in html and "dirty=true" in html
    assert (directory / "validation/report.json").as_uri() in html
    main(["show-task", "pick", "--config-root", f"demo={task_root}"])
    assert "execution=completed, physical=false" in capsys.readouterr().out


@pytest.mark.parametrize("contents", [None, "not json", "[]"])
def test_validation_reference_rejects_missing_or_invalid_file(
    task_root: Path, contents: str | None
) -> None:
    directory = task_root / "manipulation/pick"
    value = json.loads((directory / "catalog.yaml").read_text())
    value["deployments"]["a"]["validation"] = "validation.json"
    write(directory / "catalog.yaml", value)
    if contents is not None:
        (directory / "validation.json").write_text(contents)
    with pytest.raises((ValueError, TypeError)):
        catalog._load_catalog({"demo": task_root})


def test_failed_validation_startup_does_not_claim_physical_failure(
    task_root: Path,
) -> None:
    directory = task_root / "manipulation/pick"
    value = json.loads((directory / "catalog.yaml").read_text())
    value["deployments"]["a"]["validation"] = "validation.json"
    write(directory / "catalog.yaml", value)
    write(
        directory / "validation.json",
        {"status": "error", "error": {"message": "startup <error>"}, "results": []},
    )
    (task,) = catalog._load_catalog({"demo": task_root})
    html = catalog._render_html([task])
    assert "Report status: error" in html
    assert "physical=unavailable" in html
    assert "physical=false" not in html


def test_declared_deployment_rejects_runtime_unsupported_extension(
    task_root: Path,
) -> None:
    directory = task_root / "manipulation/pick"
    value = json.loads((directory / "catalog.yaml").read_text())
    value["deployments"]["a"]["config"] = "env.txt"
    write(directory / "catalog.yaml", value)
    write(directory / "env.txt", {"id": "LooksRunnable-v1", "physics": "default"})
    with pytest.raises(ValueError, match="extension"):
        catalog._load_catalog({"demo": task_root})


def test_agent_relative_config_precedes_cwd_reference(
    task_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = task_root / "manipulation/pick"
    value = json.loads((directory / "catalog.yaml").read_text())
    value["deployments"]["a"]["config"] = "env.yaml"
    value["deployments"]["b"]["config"] = "other/env.yaml"
    write(directory / "catalog.yaml", value)
    write(directory / "env.yaml", {"id": "AgentRelative-v1", "physics": "default"})
    write(directory / "other/env.yaml", {"id": "CwdRelative-v1", "physics": "default"})
    write(directory / "agents/ppo.json", {"trainer": {"gym_config": "../env.yaml"}})
    cwd = directory / "other/child"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    (task,) = catalog._load_catalog({"demo": task_root})
    assert catalog._RL in task.deployments[0].capabilities
    assert catalog._RL not in task.deployments[1].capabilities
