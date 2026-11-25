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
import os
from pathlib import Path
import subprocess
import sys

import pytest

from embodichain.cli import _task_catalog as catalog
from embodichain.cli.main import main


def test_runtime_resources_prefer_the_selected_source_checkout(tmp_path: Path) -> None:
    foreign = tmp_path / "foreign_configs"
    foreign.mkdir()
    (foreign / "__init__.py").write_text("ORIGIN = 'another checkout'\n")
    root = Path(__file__).resolve().parents[1]
    script = """
import importlib.abc
import importlib.machinery
import importlib.util
from importlib.resources import files
from pathlib import Path
import sys

class StaleEditableFinder(importlib.abc.MetaPathFinder):
    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        if fullname == 'embodichain.extensions.tasks.configs':
            return importlib.util.spec_from_file_location(fullname, sys.argv[1])

sys.meta_path.insert(sys.meta_path.index(importlib.machinery.PathFinder) + 1, StaleEditableFinder())
import embodichain.extensions.tasks
assert Path(str(files('embodichain.extensions.tasks.configs'))).resolve() == Path(sys.argv[2]).resolve()
from embodichain.extensions.tasks.locomotion.velocity.contracts._config import load_task_data
assert len(load_task_data('anymal_c_flat')['robot']['joint_names']) == 12
assert 'embodichain.extensions.tasks.embodichain.extensions.tasks' not in sys.modules
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(foreign / "__init__.py"),
            str(root / "embodichain/extensions/tasks/configs"),
        ],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=45,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                [
                    str(root),
                    str(root / "extensions"),
                    str(root / "embodichain/extensions/tasks"),
                    os.environ.get("PYTHONPATH", ""),
                ]
            ),
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_pick_place_catalog_groups_tasks_and_preserves_deployment_lookups() -> None:
    root = (
        Path(__file__).resolve().parents[1]
        / "embodichain/extensions/tasks/configs/tasks"
    )
    tasks = catalog._load_catalog({"embodichain.extensions.tasks": root})
    task = catalog._select_task(
        tasks, "embodichain.extensions.tasks:gaussian_pick_place"
    )
    assert {
        t.key for t in tasks if t.task_path[:-1] == ("manipulation", "pick_place")
    } == {
        "gaussian_pick_place",
        "repeated_pick_place",
        "workspace_pose_pick_place",
    }
    acceptance = catalog._select_task(tasks, "scene_first")
    assert acceptance.key == "workspace_pose_pick_place"
    assert acceptance.default_deployment == "scene_first_acceptance"
    repeated = catalog._select_task(tasks, "repeated_pick_place")
    for key in (
        "rubiks_cube_pick_place",
        "embodichain.extensions.tasks:rubiks_cube_pick_place",
    ):
        alias = catalog._select_task(tasks, key)
        assert alias.qualified_key == repeated.qualified_key
        assert alias.default_deployment == "ur5_rubiks_cube"
        (rubiks,) = alias.deployments
        assert rubiks in repeated.deployments
        assert rubiks.env_id == "TaskProgramRubiksCubePickPlace-v1"
        assert (
            rubiks.resource
            == root / "manipulation/rubiks_cube_pick_place/task.ur5.yaml"
        )
    assert task.default_deployment == "office"
    assert {deployment.name: deployment.env_id for deployment in task.deployments} == {
        "office": "FrankaGSPickPlace-Office-v1",
        "discoverse": "FrankaGSPickPlace-Discoverse-v1",
        "robosimgs": "FrankaGSPickPlace-Robosimgs-v1",
        "dexuni": "TaskProgramOfficeUR5-DexUni-v1",
    }
    assert all(
        deployment.capabilities == {catalog._TASK_PROGRAM}
        and deployment.physics
        == ("newton" if deployment.name == "dexuni" else "default")
        for deployment in task.deployments
    )


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


def test_official_agent_references_match_outside_checkout(
    task_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use the packaged YAML path to match agent and deployment metadata."""
    monkeypatch.chdir(task_root)
    write(
        task_root / "manipulation/pick/agents/ppo.json",
        {
            "trainer": {
                "gym_config": "embodichain/extensions/tasks/configs/tasks/manipulation/pick/task.a.yaml"
            }
        },
    )
    (task,) = catalog._load_catalog({"embodichain.extensions.tasks": task_root})
    deployment = task.deployments[0]
    assert (
        deployment.config_ref
        == "embodichain/extensions/tasks/configs/tasks/manipulation/pick/task.a.yaml"
    )
    assert catalog._RL in deployment.capabilities
    assert deployment.agent_refs == (
        "embodichain/extensions/tasks/configs/tasks/manipulation/pick/agents/ppo.json",
    )


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
    assert "embodichain run-task --gym_config" in detail
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


@pytest.mark.parametrize(
    ("component", "backend"),
    [
        ("env.yaml", "newton"),
        (
            "embodichain/extensions/tasks/configs/tasks/manipulation/tableware/stack_blocks_two/env.yaml",
            "default",
        ),
    ],
    ids=["relative", "packaged"],
)
def test_environment_component_supplies_backend(
    task_root: Path, component: str, backend: str
) -> None:
    write(
        task_root / "manipulation/pick/task.a.yaml",
        {"id": "Pick-A-v1", "environment": {"component": component}},
    )
    write(task_root / "manipulation/pick/env.yaml", {"physics": "newton"})
    (task,) = catalog._load_catalog({"demo": task_root})
    assert task.deployments[0].physics == backend


def test_duplicate_authored_fields_fail(task_root: Path) -> None:
    path = task_root / "manipulation/pick/catalog.yaml"
    path.write_text("task_key: pick\ntask_key: other\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        catalog._load_catalog({"demo": task_root})


def test_metadata_rejects_non_runnable_config(task_root: Path) -> None:
    write(task_root / "manipulation/pick/task.a.yaml", {"physics": "default"})
    with pytest.raises(ValueError, match="runnable"):
        catalog._load_catalog({"demo": task_root})


@pytest.mark.parametrize(
    "inferred,declared,expected",
    [(True, None, True), (True, False, False), (False, True, True)],
)
def test_runtime_only_and_registered_capabilities_are_preserved(
    task_root: Path, monkeypatch: pytest.MonkeyPatch, inferred, declared, expected
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
        supports_handwritten_demo=declared,
    )
    monkeypatch.setattr(registration, "REGISTERED_ENVS", {"Runtime-v1": spec})
    monkeypatch.setattr(
        env, "get_registered_learning_env_names", lambda: ["UnconfiguredRL"]
    )
    monkeypatch.setattr(catalog, "_implements_handwritten_demo", lambda cls: inferred)
    tasks = catalog._load_catalog({"demo": task_root})
    catalog._augment_runtime(tasks, ["demo"])
    runtime = catalog._select_task(tasks, "demo:runtime").deployments[0]
    assert runtime.config_ref is None
    assert runtime.capabilities == (
        {catalog._HANDWRITTEN_DEMO, catalog._RL} if expected else {catalog._RL}
    )
    assert catalog._select_task(tasks, "runtime:UnconfiguredRL").deployments[
        0
    ].capabilities == {catalog._RL}


def test_official_catalogs_resolve_real_deployments() -> None:
    root = (
        Path(__file__).resolve().parents[1]
        / "embodichain/extensions/tasks/configs/tasks"
    )
    tasks = catalog._load_catalog({"embodichain.extensions.tasks": root})
    pick = catalog._select_task(tasks, "repeated_pick_place")
    assert {d.name for d in pick.deployments} >= {
        "franka",
        "ur5",
        "ur5_objective",
    }
    deployments = {deployment.name: deployment for deployment in pick.deployments}
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


def test_official_dexuni_deployments_resolve_to_the_selected_solver() -> None:
    root = (
        Path(__file__).resolve().parents[1]
        / "embodichain/extensions/tasks/configs/tasks"
    )
    tasks = catalog._load_catalog({"embodichain.extensions.tasks": root})
    ppo_tasks = {
        "cart_pole",
        "humanoid",
        "push_cube",
        "peg_insertion",
        "gear_meshing",
        "nut_threading",
    }
    expected = ppo_tasks | {
        "carton_stocking",
        "drawer_storage",
        "gaussian_pick_place",
        "hand_over",
        "workspace_pose_pick_place",
        "blocks_ranking_rgb",
        "blocks_ranking_size",
        "match_object_container",
        "place_object_drawer",
        "stack_blocks_two",
        "stack_cups",
    }
    seen = set()
    for task in tasks:
        if task.key not in expected:
            continue
        (dexuni,) = [d for d in task.deployments if d.name == "dexuni"]
        assert dexuni.physics == "newton"
        assert dexuni.env_id.endswith("-DexUni-v1")
        assert all(
            d.env_id != dexuni.env_id for d in task.deployments if d.name != "dexuni"
        )
        deployment = catalog._load_mapping(dexuni.resource)
        if "environment" in deployment:
            deployment = catalog._load_mapping(
                dexuni.resource.parent / deployment["environment"]["component"]
            )
        assert deployment["physics_config"]["solver_cfg"]["solver_type"] == "dexuni"
        if task.key in ppo_tasks:
            training = next(
                (d for d in task.deployments if d.name == "dexuni_nominal"), dexuni
            )
            assert training.physics == "newton"
            (agent,) = training.agent_refs
            assert Path(agent).name.endswith((".dexuni.yaml", ".dexuni.json"))
        seen.add(task.key)
    assert seen == expected


def test_official_portfolio_keeps_deployment_purposes_and_evidence_separate() -> None:
    root = (
        Path(__file__).resolve().parents[1]
        / "embodichain/extensions/tasks/configs/tasks"
    )
    tasks = catalog._load_catalog({"embodichain.extensions.tasks": root})
    assert all(
        d.purpose in catalog._PURPOSES for task in tasks for d in task.deployments
    )
    pour = catalog._select_task(tasks, "pour_water")
    assert pour.default_deployment == "franka_fluid"
    assert {d.env_id for d in pour.deployments} == {
        "PourCup-Franka-Fluid-v1",
        "PourWater-v1",
        "PourWater-CobotMagic-Fluid-v1",
        "PourWater-TianjiMarvin-v1",
        "PourWater-DualUR5-v1",
        "PourWater-DexforceW1V030Pika-v1",
    }
    rigid = next(d for d in pour.deployments if d.name == "cobotmagic")
    fluid = next(d for d in pour.deployments if d.name == "cobotmagic_fluid")
    assert (rigid.env_id, rigid.physics, rigid.validation) == (
        "PourWater-v1",
        "default",
        None,
    )
    assert (fluid.env_id, fluid.physics) == ("PourWater-CobotMagic-Fluid-v1", "newton")
    assert fluid.validation["deployment"] == fluid.name
    assert catalog._select_task(tasks, "pour_cup").qualified_key == pour.qualified_key
    research = catalog._portfolio_view(tasks, "research", False)
    stack = catalog._select_task(research, "stack_blocks_two")
    assert [d.env_id for d in stack.deployments] == ["StackBlocksTwoPolicy-v1"]
    featured = catalog._portfolio_view(tasks, None, True)
    drawer = catalog._select_task(featured, "drawer_storage")
    assert drawer.default_deployment == "ur5"
    assert drawer.deployments[0].validation["attempts"] == 10
    assert all(task.key != "cart_pole" for task in featured)
    assert (
        catalog._select_task(tasks, "cart_pole").deployments[0].purpose == "reference"
    )


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


def test_carton_source_and_ordinary_rgbd_deployments_without_loading_assets(
    monkeypatch,
) -> None:
    root = (
        Path(__file__).resolve().parents[1]
        / "embodichain/extensions/tasks/configs/tasks"
    )

    def forbidden(*args):
        raise AssertionError("catalog browsing must not initialize source assets")

    monkeypatch.setattr("embodichain.data.get_data_class", forbidden)
    task = next(
        t
        for t in catalog._load_catalog({"embodichain.extensions.tasks": root})
        if t.qualified_key == "embodichain.extensions.tasks:carton_stocking"
    )
    assert task.default_deployment == "panda"
    deployments = {deployment.name: deployment for deployment in task.deployments}
    assert set(deployments) == {"panda", "panda_rgbd", "dexuni"}
    assert deployments["panda"].env_id == "CartonStockingPanda-v1"
    assert deployments["panda"].capabilities == {catalog._TASK_PROGRAM}
    assert deployments["panda"].requires_dataset_dir
    assert "--dataset-dir ./datasets" in deployments["panda"].command
    assert deployments["panda_rgbd"].env_id == "CartonStockingPandaRGBD-v1"
    assert deployments["panda_rgbd"].capabilities == {catalog._TASK_PROGRAM}
    assert not deployments["panda_rgbd"].requires_dataset_dir
    assert "--dataset-dir" not in deployments["panda_rgbd"].command


@pytest.mark.parametrize("variant", [False, True], ids=["primary", "variant"])
def test_grouped_runtime_deployments_keep_identity_and_purpose(
    task_root, monkeypatch, variant: bool
) -> None:
    from types import SimpleNamespace
    from embodichain.lab.gym.utils import registration
    from embodichain.learning.rl import env

    path = task_root / "manipulation/pick/catalog.yaml"
    write(path, {**json.loads(path.read_text()), "group": "pick_place"})
    if variant:
        directory = task_root / "manipulation/alternate"
        write(directory / "task.yaml", {"id": "Alternate-v1", "physics": "default"})
        write(
            directory / "catalog.yaml",
            {
                "task_key": "alternate",
                "variant_of": "pick",
                "title": "Alternate Pick",
                "summary": "An additional deployment for picking.",
                "default_deployment": "alternate",
                "deployments": {"alternate": {"config": "task.yaml"}},
            },
        )
    module = "demo.manipulation.alternate" if variant else "demo.manipulation.pick"
    cls = type("PickEnv", (), {"__module__": module})
    spec = SimpleNamespace(
        cls=cls,
        supports_handwritten_demo=False,
        supports_rl=True,
        task_program_registration=None,
        task_program_adapter_factory=None,
    )
    monkeypatch.setattr(
        registration, "REGISTERED_ENVS", {"PickAlt": spec, "PickResearch": spec}
    )
    monkeypatch.setattr(env, "get_registered_learning_env_names", lambda: [])
    write(
        task_root / "portfolio.yaml",
        {
            "pick": {
                "purpose": "training",
                "featured": ["PickAlt"],
                "overrides": {"PickResearch": "research"},
            }
        },
    )
    roots = {"demo": task_root}
    static = catalog._load_catalog(roots)
    configured = {"a", "b", "alternate"} if variant else {"a", "b"}
    assert {
        d.name for d in catalog._select_task(static, "pick").deployments
    } == configured
    tasks = catalog._load_catalog(roots, runtime_packages=["demo"])
    (pick,) = tasks
    assert catalog._select_task(tasks, "demo:pick") is pick
    assert pick.task_path == ("manipulation", "pick_place", "pick")
    if variant:
        alias = catalog._select_task(tasks, "demo:alternate")
        assert alias.key == pick.key
        assert alias.default_deployment == "alternate"
        assert [d.env_id for d in alias.deployments] == ["Alternate-v1"]
    training = catalog._portfolio_view(tasks, "training", False)
    assert {d.name for d in training[0].deployments} == configured | {"PickAlt"}
    assert [
        d.name for d in catalog._portfolio_view(tasks, None, True)[0].deployments
    ] == ["PickAlt"]
    assert [
        d.name for d in catalog._portfolio_view(tasks, "research", False)[0].deployments
    ] == ["PickResearch"]
