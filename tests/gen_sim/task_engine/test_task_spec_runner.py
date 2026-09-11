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


"""CPU host regressions: final TaskSpec evidence precedes reset and submission."""

from __future__ import annotations

import json
from types import SimpleNamespace
import pytest
import torch

from embodichain.gen_sim.task_engine import _bundle_runner as runner
from embodichain.gen_sim.task_engine._task_spec import binding_for_graph
from embodichain.gen_sim.task_engine.semantic_planner import SemanticTaskPlanner
from embodichain.lab.gym.envs.demo import DemoSegment
from .test_task_spec_planning import planning_inputs


@pytest.mark.parametrize("mode", ["goal_failed", "execution_error", "succeeded"])
def test_runner_freezes_task_evidence_before_reset(tmp_path, monkeypatch, mode):
    import gymnasium
    from embodichain.lab.gym.utils import gym_utils, registration
    from embodichain.gen_sim.task_engine._task_program import assembly
    from embodichain.lab.task_program import language
    from embodichain.lab.sim.sim_manager import SimulationManager
    from embodichain.gen_sim.task_engine import _task_spec

    candidate, bindings, objects, template, _ = planning_inputs()
    graph = SemanticTaskPlanner().plan(candidate, bindings, objects)
    binding = binding_for_graph(template, graph)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "task_program").mkdir()
    for name in (
        "task_program_deployment.yaml",
        "task_program/program.yaml",
        "semantic_task_graph.json",
        "integration_fingerprint.json",
    ):
        (bundle / name).write_text("{}")
    output = tmp_path / "execution"
    pose = torch.eye(4).unsqueeze(0)
    pose[0, :3, :3] = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    calls = []

    class Env:
        num_envs = 1
        sim = SimpleNamespace(
            get_rigid_object=lambda uid: SimpleNamespace(
                get_local_pose=lambda **kwargs: pose
            )
        )

        def reset(self, **kwargs):
            calls.append(("reset", kwargs))
            if len([call for call in calls if call[0] == "reset"]) == 2:
                report = json.loads((output / "task_evaluation.json").read_text())
                assert report["accepted"] == [mode == "succeeded"]
                if mode != "succeeded":
                    assert kwargs["options"] == {"save_data": False}
                # Destroy old state; a post-reset observer would read this instead.
                pose[:] = float("nan")

        def create_demo_segments(self):
            if mode == "execution_error":
                raise RuntimeError("injected planning failure")
            yield DemoSegment(actions=(1,), name="recipe")

        def step(self, action):
            if mode == "succeeded":
                pose[0] = torch.eye(4)
            return (
                None,
                torch.zeros(1),
                torch.zeros(1, dtype=torch.bool),
                torch.zeros(1, dtype=torch.bool),
                {},
            )

        def is_task_success(self):
            return torch.ones(1, dtype=torch.bool)

        def _end_demo_episode_recording(self, result):
            calls.append(("metadata", result.success))
            assert (output / "task_evaluation.json").exists()
            assert result.success == (mode == "succeeded",)

        def close(self, **kwargs):
            pass

    catalog = SimpleNamespace(preflight=lambda program: None)
    deployment = SimpleNamespace(
        selection=None,
        integration=SimpleNamespace(registration=SimpleNamespace(catalog=catalog)),
    )
    monkeypatch.setattr(runner, "_verify_source", lambda root: None)
    monkeypatch.setattr(runner, "validate_semantic_task_graph", lambda value: graph)
    monkeypatch.setattr(_task_spec, "read_binding", lambda *args: binding)
    monkeypatch.setattr(
        runner, "_verify_integration_fingerprint", lambda *args: deployment
    )
    monkeypatch.setattr(runner, "_verify_program_projection", lambda *args: None)
    monkeypatch.setattr(
        runner, "load_config", lambda path: {"id": "test", "max_episode_steps": 10}
    )
    monkeypatch.setattr(assembly, "register_deployment", lambda *args, **kwargs: None)
    monkeypatch.setattr(registration, "discover_task_packages", lambda: None)
    monkeypatch.setattr(registration, "execute_init_hooks", lambda: None)
    monkeypatch.setattr(
        gym_utils,
        "build_env_cfg_from_args",
        lambda *args, **kwargs: (SimpleNamespace(), {"id": "test"}, {}),
    )
    monkeypatch.setattr(language, "load_task_program", lambda *args, **kwargs: None)
    monkeypatch.setattr(gymnasium, "make", lambda **kwargs: Env())
    monkeypatch.setattr(
        runner, "_preserve_failed_execution_recording", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(SimulationManager, "flush_cleanup_queue", lambda: None)
    monkeypatch.setattr(
        runner,
        "_build_execution_report",
        lambda *args, **kwargs: {
            "status": "succeeded" if mode == "succeeded" else "failed"
        },
    )
    monkeypatch.setattr(runner, "write_execution_report", lambda *args: None)
    assert runner.execute_bundle(bundle, execution_output=output) == (
        0 if mode == "succeeded" else 2
    )
    assert len([call for call in calls if call[0] == "reset"]) == 2
