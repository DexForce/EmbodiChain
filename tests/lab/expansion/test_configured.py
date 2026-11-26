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

"""Native file jobs fail before execution for unsupported schemas and scopes."""

from __future__ import annotations
from pathlib import Path
import pytest
from embodichain.lab.expansion import ExpansionEngine, load_expansion_job


def test_task_program_declarations_are_pure_and_protect_contact_phases(tmp_path):
    import json
    import os
    import subprocess
    import sys
    import yaml
    from embodichain.lab.expansion import load_task_program_expansion

    path = Path(__file__).parent / "fixtures/task_program_job.yaml"
    code = "from embodichain.lab.expansion import load_task_program_expansion; import json,sys; value=load_task_program_expansion(sys.argv[1]); assert not any(k=='dexsim' or k.startswith('dexsim.') for k in sys.modules); print(json.dumps(value))"
    process = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ},
    )
    value = json.loads(process.stdout)
    assert (value["engine"]["workload_count"], value["engine"]["batch_size"]) == (2, 1)
    assert value["trajectory"]["operators"] == ["object_pose", "joint_residual"]
    value["engine"]["nominal_seed"] = 7
    nominal = tmp_path / "nominal.yaml"
    nominal.write_text(yaml.safe_dump(value))
    assert load_task_program_expansion(nominal)["engine"]["nominal_seed"] == 7
    value["engine"]["nominal_seed"] = True
    nominal.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="nominal_seed"):
        load_task_program_expansion(nominal)
    value["engine"]["nominal_seed"] = 7
    value["trajectory"]["free_phases"]["pick_up"] = ["approach", "lift"]
    invalid = tmp_path / "protected.yaml"
    invalid.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="protected free phase"):
        load_task_program_expansion(invalid)


@pytest.mark.parametrize("steps", [0, -1, False])
def test_task_program_settling_steps_can_skip_preliminary_physics(steps, tmp_path):
    import yaml
    from embodichain.lab.expansion import load_task_program_expansion

    fixture = Path(__file__).parent / "fixtures/task_program_job.yaml"
    job = yaml.safe_load(fixture.read_text())
    job["engine"]["settling_steps"] = steps
    path = tmp_path / "settling.yaml"
    path.write_text(yaml.safe_dump(job))
    if type(steps) is int and steps == 0:
        assert load_task_program_expansion(path)["engine"]["settling_steps"] == 0
    else:
        with pytest.raises(ValueError, match="engine.settling_steps"):
            load_task_program_expansion(path)


def test_official_layout_jobs_load_without_simulation_imports():
    import subprocess
    import sys

    task_root = (
        Path(__file__).resolve().parents[3]
        / "embodichain/extensions/tasks/configs/tasks/manipulation"
    )
    paths = sorted(task_root.glob("**/acceptance/scene_first/expansion.yaml"))
    assert paths
    code = """
import importlib.abc
import sys

class NoSimulator(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("dexsim", "embodichain.lab.sim", "embodichain.lab.gym")):
            raise AssertionError(f"configuration loader imported {fullname}")

sys.meta_path.insert(0, NoSimulator())
from embodichain.lab.expansion import load_task_program_expansion
for path in sys.argv[1:]:
    sampling = load_task_program_expansion(path)["scene_sampling"]
    assert sampling["layout"]["policy"] and sampling["task_mode"]["phases"]
    assert sampling["robot_anchors"]
"""
    result = subprocess.run(
        [sys.executable, "-c", code, *map(str, paths)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("name", ["default", "newton", "rgbd", "motion"])
def test_example_preflight_resolves_owned_backend_and_budget(name, tmp_path):
    path = (
        Path(__file__).resolve().parents[3]
        / "examples/expansion/engine"
        / f"job.{name}.yaml"
    )
    cfg = load_expansion_job(path)
    cfg.output_dir = str(tmp_path / "untouched")
    engine = ExpansionEngine.from_job(cfg)
    report = engine.preflight()
    assert report["physics_backend"] == ("newton" if name == "newton" else "default")
    assert report["pending_numeric_bytes"] == cfg.batch_size * cfg.episode_byte_budget
    assert not Path(cfg.output_dir).exists()
    assert not report["capabilities"]["single_body_damping"]


def test_native_media_requires_complete_decision_contract():
    path = (
        Path(__file__).resolve().parents[3]
        / "examples/expansion/engine/job.default.yaml"
    )
    cfg = load_expansion_job(path)
    cfg.render_profiles = ("blue",)
    with pytest.raises(ValueError, match="observation_independent"):
        ExpansionEngine.from_job(cfg)


def test_resource_changes_preserve_semantics_but_record_new_execution(tmp_path):
    from embodichain.lab.expansion.configured import preflight_expansion_job

    path = (
        Path(__file__).resolve().parents[3]
        / "examples/expansion/engine/job.default.yaml"
    )
    cfg = load_expansion_job(path)
    a = preflight_expansion_job(cfg)
    cfg.batch_size = 32
    cfg.max_attempts = 64
    cfg.target_episodes = 32
    cfg.episode_byte_budget *= 2 // 2
    cfg.output_dir = str(tmp_path)
    b = preflight_expansion_job(cfg)
    assert a["semantic_id"] == b["semantic_id"]
    assert a["execution_id"] != b["execution_id"]
    cfg.mass_scale = {"cube": (2.0, 3.0)}
    assert preflight_expansion_job(cfg)["semantic_id"] != a["semantic_id"]


@pytest.mark.parametrize("owner", ["environment", "scene"])
def test_deployment_snapshot_preserves_component_relative_assets(tmp_path, owner):
    import json
    import yaml
    from embodichain.lab.expansion.task_program_jobs import _prepare_deployment
    from embodichain.lab.gym.utils.composition import resolve_gym_components

    source = tmp_path / "source"
    source.mkdir()
    mesh = source / "mesh.obj"
    mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    simulation = {
        "rigid_object": [
            {"uid": "mesh", "shape": {"shape_type": "Mesh", "fpath": "./mesh.obj"}}
        ]
    }
    component = (
        {
            "environment_id": "mesh-test",
            "physics": "default",
            "simulation": simulation,
            "env": {},
        }
        if owner == "environment"
        else {"scene_id": "mesh-test", "simulation": simulation}
    )
    (source / "component.yaml").write_text(yaml.safe_dump(component))
    declaration = {"id": "SnapshotTest-v1", owner: {"component": "component.yaml"}}
    deployment = source / "deployment.yaml"
    deployment.write_text(yaml.safe_dump(declaration))
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    output = tmp_path / "output"
    output.mkdir()
    derived = _prepare_deployment(deployment, output)
    snapshot = json.loads(derived.read_text())
    resolved = resolve_gym_components(snapshot, base_dir=derived.parent).config
    assert resolved["rigid_object"][0]["shape"]["fpath"] == str(mesh)
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before


@pytest.mark.parametrize("selector", ["path", "component", "inline"])
def test_expansion_boundary_policy_preserves_selector_and_original_file(
    tmp_path, selector
):
    import json
    from pathlib import Path
    import yaml
    from embodichain.lab.expansion.task_program_jobs import _prepare_deployment

    root = Path(__file__).resolve().parents[3]
    source = (
        root
        / "embodichain/extensions/tasks/configs/components/execution_policies/trajectory_open_loop.yaml"
    )
    policy = yaml.safe_load(source.read_text())
    owned = tmp_path / "policy.yaml"
    owned.write_text(yaml.safe_dump(policy))
    before = owned.read_bytes()
    selection = str(owned)
    if selector == "component":
        selection = {
            "component": str(owned),
            "overrides": {"motion": {"sample_count": 23}},
        }
    if selector == "inline":
        selection = {"inline": policy}
    env = tmp_path / "env.yaml"
    env.write_text("physics: default\n")
    deployment = tmp_path / "task.yaml"
    deployment.write_text(
        yaml.safe_dump(
            {
                "id": "test",
                "environment": {"component": str(env)},
                "task_program": {"execution_policy": selection},
            }
        )
    )
    output = tmp_path / "out"
    output.mkdir()
    prepared = json.loads(_prepare_deployment(deployment, output).read_text())
    result = yaml.safe_load(
        Path(prepared["task_program"]["execution_policy"]).read_text()
    )
    assert result["tracking"]["boundary_stop"] is True
    assert result["motion"]["sample_count"] == (
        23 if selector == "component" else policy["motion"]["sample_count"]
    )
    assert owned.read_bytes() == before
