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

from pathlib import Path

import pytest
import torch
import yaml

from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.sim import SimulationManager
from embodichain_tasks.manipulation.cable.route_the_cable import RouteTheCableEnv

pytestmark = [pytest.mark.requires_sim, pytest.mark.gpu, pytest.mark.slow]


class TestSpoolReleaseCUDA:
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
        cfg.sim_cfg.render_cfg.renderer = "hybrid"
        self.env = RouteTheCableEnv(cfg)

    def teardown_method(self) -> None:
        if hasattr(self, "env"):
            self.env.close(exit_process=False)
        SimulationManager.flush_cleanup_queue()

    def test_first_spool_release_preserves_seating_without_gripper_penetration(
        self,
        record_property,
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
        for segment in env.create_demo_segments(milestone="first_spool"):
            for action in segment.actions:
                env.step(action)
                contacts = query.fetch()
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

        # Finite cable states and seated cable alone previously accepted a jaw
        # sweep through the rim with about 16 mm reported penetration.
        assert worst_distance >= -0.001, (worst_segment, worst_distance)
        max_slip = float(env._episode_slip.max())
        record_property("max_grasp_slip_m", max_slip)
        assert max_slip <= 0.012
        assert bool((env.route_metrics()["seat_0"] >= 2).all())
        assert bool((env._tool("right")[:, 2, 3] > 0.81).all())
        finger_ids = env.robot.get_joint_ids("right_eef")
        assert bool((env.robot.get_qpos()[:, finger_ids] > 0.045).all())

    def test_third_carry_preserves_grasp_and_clears_spools(
        self, record_property
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
                    max_slip = max(max_slip, float(env._max_slip.max()))
                    contacts = query.fetch()
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
                assert env._grasp[:2] == ("right", 0.438)
                # The gripped centre must clear the 23.5 mm rim plus cable radius.
                clearance_height = 0.75 + 0.0235 + env.cable.cfg.radius
                assert bool((env._point(0.438)[0][:, 2] > clearance_height).all())
            if segment.name == "open_clear_of_spool_2":
                assert raised
                record_property("third_carry_slip_m", max_slip)
                record_property("third_carry_robot_spool_distance_m", worst_distance)
                assert max_slip <= 0.012
                assert worst_distance >= -0.001
                assert bool((env.route_metrics()["seat_1"] >= 2).all())
                return
        pytest.fail("The expert did not complete the third carry")
