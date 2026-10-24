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

"""Exercise the actual Gym sampling boundary and same-episode audit lifecycle."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from embodichain.gen_sim.task_engine._task_program import acceptance_audit as audit
from embodichain.gen_sim.task_engine._task_program import (
    acceptance_observation as observation,
)
from embodichain.lab.gym.envs.base_env import BaseEnv
from embodichain.utils.utility import save_config
from .test_task_spec_runtime import _placement_evidence


def test_installed_contact_observer_receives_gym_physics_steps_and_resets(monkeypatch):
    clock = SimpleNamespace(step_index=0, now=lambda: clock.step_index * 0.04)
    raw = {
        "is_valid": torch.tensor([[True]]),
        "user_ids": torch.tensor([[[1, 2]]]),
        "position": torch.zeros(1, 1, 3),
        "normal": torch.tensor([[[0.0, 0.0, 1.0]]]),
        "impulse": torch.tensor([[0.01]]),
    }
    sensor = SimpleNamespace(
        requires_substep_update=True,
        dropped_contacts=0,
        get_data=lambda: raw,
        get_actor_ids=lambda *args: (
            torch.tensor([[1]]) if len(args) == 1 else torch.tensor([[3, 4]])
        ),
    )
    sensor.begin_control_step = lambda: sensor.acceptance_reader.begin()
    sensor.update_physics_step = lambda dt: sensor.acceptance_reader.capture(
        dt, sensor.acceptance_reader.source_revision()
    )
    config = []
    sim = SimpleNamespace(
        SUPPORTED_SENSOR_TYPES={},
        spawn_result=SimpleNamespace(backend="dexsim", topology_revision=1),
    )
    sim.get_world = lambda: SimpleNamespace(
        get_physics_scene=lambda: SimpleNamespace(_get_contact_data_revision=lambda: 1)
    )
    sim.add_sensor = lambda cfg: (config.append(cfg), sensor)[1]
    sim.get_rigid_object = lambda uid: SimpleNamespace(
        cfg=SimpleNamespace(body_type="dynamic")
    )

    def update(dt, steps, *, after_substep, **kwargs):
        for index in range(steps):
            raw["impulse"].fill_((index + 1) * 0.01)
            after_substep(dt)

    sim.update = update
    env = SimpleNamespace(
        sim=sim,
        num_envs=1,
        sensors={},
        step_dt=0.04,
        physics_dt=0.01,
        cfg=SimpleNamespace(sim_steps_per_control=4),
        _active_task_program_bridge=SimpleNamespace(_clock=clock),
        robot=SimpleNamespace(
            uid="robot",
            link_names=[
                f"{side}_{number}_finger_pad"
                for side in ("left", "right")
                for number in (0, 1)
            ],
        ),
    )
    deployment = SimpleNamespace(
        integration=SimpleNamespace(
            registration=SimpleNamespace(
                scene_binding=SimpleNamespace(
                    rigid_objects=(
                        SimpleNamespace(entity_id="object", simulation_uid="native"),
                    )
                )
            )
        )
    )
    monkeypatch.setattr(observation, "_opening_reader", lambda *args: lambda *args: {})
    reader = observation.install_contact_timeline(
        env, deployment, {"task_groups": [{"task_type": "E5"}]}
    )
    assert env.sensors[observation.SENSOR_UID] is sensor
    assert config[0].filter_need_both_actor is False
    BaseEnv._advance_physics(env)
    clock.step_index = 1
    BaseEnv._advance_physics(env)
    assert list(reader.rows) == [1, 2]
    assert reader.rows[2]["time"] == 0.08
    assert len(reader.rows[2]["substeps"]) == 4
    assert (
        reader.rows[2]["substeps"][0]["rows"][0]["impulses"][0]
        < reader.rows[2]["substeps"][-1]["rows"][0]["impulses"][0]
    )
    assert reader.snapshot()["retained_contact_rows"] == 8
    reader.reset()
    assert reader.rows == {}
    assert reader.generation == 1


@pytest.mark.parametrize("mutate", [False, True])
def test_dual_observer_audit_preserves_runtime_and_exact_terminal_snapshot(
    tmp_path, monkeypatch, mutate
):
    plan, contract, runtime, program, constraints, env, deployment = (
        _placement_evidence(tmp_path)
    )
    save_config(tmp_path / "task_program/program.yaml", program)
    env.robot = SimpleNamespace(get_qpos=lambda: torch.zeros(1, 2))
    deployment.integration.registration.scene_binding.articulations = ()
    output = tmp_path / "audit"
    output.mkdir()
    original = audit.capture_final_goals

    def capture(*args, **kwargs):
        value = original(*args, **kwargs)
        if mutate and not kwargs.get("legacy_observer"):
            runtime["unexpected_mutation"] = True
        return value

    monkeypatch.setattr(audit, "capture_final_goals", capture)
    if mutate:
        with pytest.raises(ValueError, match="modified native state or runtime"):
            audit.evaluate_episode_observers(
                env, deployment, tmp_path, plan.graph, runtime, contract, 1, output
            )
        return
    current = audit.evaluate_episode_observers(
        env, deployment, tmp_path, plan.graph, runtime, contract, 1, output
    )
    legacy = json.loads((output / "task_spec_legacy_evaluation.json").read_text())
    comparison = json.loads((output / "task_spec_observer_comparison.json").read_text())
    terminal = json.loads((output / "task_spec_terminal_snapshot.json").read_text())
    assert current["environments"][0]["certificate_ready"] is True
    assert legacy["environments"][0]["certificate_ready"] is True
    assert comparison["native_state_unchanged"] is True
    assert comparison["legacy_affects_selected_verdict"] is False
    assert terminal["step_counts"] == env._traj_steps.tolist()
    with np.load(output / "task_spec_native_geometry.npz") as geometry:
        assert geometry["terminal_steps"].tolist() == env._traj_steps.tolist()
        assert geometry["rigid_0_terminal_pose"].shape == (1, 4, 4)
