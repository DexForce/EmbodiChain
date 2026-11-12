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

from collections.abc import Callable
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml

from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.sim import SimulationManager
import gymnasium as gym
import embodichain_tasks

pytestmark = [
    pytest.mark.subprocess_sim,
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="Cable routing requires CUDA"
    ),
    pytest.mark.usefixtures("requires_dexuni_two_way", "requires_dexuni_contact_query"),
]


def _material_runtime(env):
    return env._active_task_program_bridge.post_policy_port._cable_runtime


class _SpoolReleaseCUDAScenario:
    """Qualify the actual jaw sweep, rather than only cable goal predicates."""

    def setup_method(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("Cable routing requires CUDA")
        path = (
            Path(__file__).resolve().parents[4]
            / "extensions/embodichain_tasks/configs/tasks/manipulation/cable/route_the_cable/task.yam.yaml"
        )
        cfg = config_to_cfg(yaml.safe_load(path.read_text()), source_path=path)
        cfg.seed = 73
        cfg.sim_cfg.render_cfg.renderer = "no-render"
        cfg.sim_cfg.robot_ik_gizmo = None
        self.env = gym.make("RouteTheCable-YAM-v1", cfg=cfg).unwrapped

    def teardown_method(self) -> None:
        if hasattr(self, "env"):
            self.env.close(exit_process=False)
        SimulationManager.flush_cleanup_queue()

    def test_first_spool_release_preserves_seating_without_gripper_penetration(
        self,
        record_property: Callable[[str, float], None],
    ) -> None:
        env = self.env
        scene = env.sim.spawn_result
        handles = scene.handles
        robot_path = next(path for path in handles if path.endswith("/yam"))
        spool_paths = [path for path in handles if "/spool_" in path]
        query = scene.create_contact_query(
            [handles[robot_path], *(handles[path] for path in spool_paths)],
            match="all",
            capacity=4096,
            device=env.device,
        )
        spool_ids = torch.tensor(
            [actor.actor_id for actor in query.actors if actor.path in spool_paths],
            device=env.device,
        )
        robot_ids = torch.tensor(
            [actor.actor_id for actor in query.actors if actor.path == robot_path],
            device=env.device,
        )
        worst_distance = 0.0
        worst_segment = ""
        env.reset()
        for segment in env.create_demo_segments():
            for action in segment.actions:
                env.step(action)
                contacts = query.fetch_solver_report()
                assert contacts.dropped_count == 0
                pairs = contacts.actor_ids[: contacts.count]
                robot_spool = torch.isin(pairs, robot_ids).any(-1) & torch.isin(
                    pairs, spool_ids
                ).any(-1)
                distances = contacts.distance[: contacts.count][robot_spool]
                if distances.numel():
                    distance = float(distances.min())
                    if distance < worst_distance:
                        worst_distance, worst_segment = distance, segment.name
            assert bool(segment.validator().all()), segment.name
            if segment.name == "open_clear_of_spool_0":
                break

        # Finite cable states and seated cable alone previously accepted a jaw
        # sweep through the rim with about 16 mm reported penetration.
        assert worst_distance >= -0.001, (worst_segment, worst_distance)
        max_slip = float(_material_runtime(env).max_slip.max())
        record_property("max_grasp_slip_m", max_slip)
        assert max_slip <= 0.012
        assert bool((_material_runtime(env).metrics()["seat_0"] >= 2).all())
        assert bool((_material_runtime(env).tool("right_arm")[:, 2, 3] > 0.81).all())
        finger_ids = env.robot.get_joint_ids("right_eef")
        assert bool((env.robot.get_qpos()[:, finger_ids] > 0.045).all())

    def test_third_carry_preserves_grasp_and_clears_spools(
        self, record_property: Callable[[str, float], None]
    ) -> None:
        env = self.env
        scene = env.sim.spawn_result
        paths = [
            path for path in scene.handles if path.endswith("/yam") or "/spool_" in path
        ]
        query = scene.create_contact_query(
            [scene.handles[path] for path in paths],
            match="all",
            capacity=4096,
            device=env.device,
        )
        robot_ids = torch.tensor(
            [actor.actor_id for actor in query.actors if actor.path.endswith("/yam")],
            device=env.device,
        )
        spool_ids = torch.tensor(
            [actor.actor_id for actor in query.actors if "/spool_" in actor.path],
            device=env.device,
        )
        third_segments = {
            "raise_2",
            "carry_2",
            "lower_2",
            "release_2",
            "retreat_2",
            "open_clear_of_spool_2",
        }
        worst_distance = 0.0
        max_slip = 0.0
        raised = False
        env.reset()
        for segment in env.create_demo_segments():
            for action in segment.actions:
                env.step(action)
                if segment.name in third_segments:
                    max_slip = max(
                        max_slip,
                        float(_material_runtime(env).max_slip.max()),
                    )
                    contacts = query.fetch_solver_report()
                    assert contacts.dropped_count == 0
                    pairs = contacts.actor_ids[: contacts.count]
                    robot_spool = torch.isin(pairs, robot_ids).any(-1) & torch.isin(
                        pairs, spool_ids
                    ).any(-1)
                    distances = contacts.distance[: contacts.count][robot_spool]
                    if distances.numel():
                        worst_distance = min(worst_distance, float(distances.min()))
            assert bool(segment.validator().all()), segment.name
            if segment.name == "raise_2":
                raised = True
                assert _material_runtime(env).grasps["right_arm"][0] == 0.438
                # The gripped centre must clear the 23.5 mm rim plus cable radius.
                clearance_height = (
                    0.75 + 0.0235 + env.sim.get_cable_object("cable").cfg.radius
                )
                assert bool(
                    (
                        _material_runtime(env).point(0.438)[0][:, 2] > clearance_height
                    ).all()
                )
            if segment.name == "open_clear_of_spool_2":
                assert raised
                record_property("third_carry_slip_m", max_slip)
                record_property("third_carry_robot_spool_distance_m", worst_distance)
                assert max_slip <= 0.012
                assert worst_distance >= -0.001
                assert bool((_material_runtime(env).metrics()["seat_1"] >= 2).all())
                return
        pytest.fail("The expert did not complete the third carry")


def _run_spool_case(case_name: str, metrics_path: Path) -> None:
    """Execute one unchanged acceptance case in its own NoRender world."""
    scenario = _SpoolReleaseCUDAScenario()
    properties: dict[str, float] = {}
    try:
        scenario.setup_method()
        getattr(scenario, case_name)(properties.__setitem__)
    finally:
        try:
            scenario.teardown_method()
        finally:
            metrics_path.write_text(json.dumps(properties), encoding="utf-8")


def _assert_spool_case_in_subprocess(
    case_name: str,
    tmp_path: Path,
    record_property: Callable[[str, float], None],
) -> None:
    """Preserve acceptance metrics without initializing pytest's native world."""
    metrics_path = tmp_path / "acceptance_metrics.json"
    script = (
        "import runpy, sys; from pathlib import Path; "
        "module = runpy.run_path(sys.argv[1]); "
        "module['_run_spool_case'](sys.argv[2], Path(sys.argv[3]))"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(Path(__file__).resolve()),
            case_name,
            str(metrics_path),
        ],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if metrics_path.exists():
        for name, value in json.loads(metrics_path.read_text(encoding="utf-8")).items():
            record_property(name, value)
    assert result.returncode == 0, result.stdout + result.stderr


class TestSpoolReleaseCUDA:
    """Qualify the physical jaw sweep in a fresh NoRender process per case."""

    def test_first_spool_release_preserves_seating_without_gripper_penetration(
        self,
        tmp_path: Path,
        record_property: Callable[[str, float], None],
    ) -> None:
        _assert_spool_case_in_subprocess(
            "test_first_spool_release_preserves_seating_without_gripper_penetration",
            tmp_path,
            record_property,
        )

    def test_third_carry_preserves_grasp_and_clears_spools(
        self,
        tmp_path: Path,
        record_property: Callable[[str, float], None],
    ) -> None:
        _assert_spool_case_in_subprocess(
            "test_third_carry_preserves_grasp_and_clears_spools",
            tmp_path,
            record_property,
        )
