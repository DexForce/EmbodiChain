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

import os

import pytest
import torch

from embodichain.data import get_data_path
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.motion.solvers import PytorchSolverCfg
from embodichain.lab.sim.cfg import RobotCfg
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim.motion.solvers.qpos_seed_sel_sampler import (
    QposSeedSelSampler,
    _pose_error_twist,
)
from embodichain.utils.utility import reset_all_seeds

_DB_SIZE = 5000
_NUM_TARGETS = 24


def _rot_x(angle: float) -> torch.Tensor:
    pose = torch.eye(4)
    c, s = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
    pose[1, 1], pose[1, 2], pose[2, 1], pose[2, 2] = c, -s, s, c
    return pose


class TestPoseErrorTwist:
    """Rotation-error stability over the whole rotation range (no sim needed)."""

    def test_exact_pi_rotation(self):
        cur = torch.eye(4).unsqueeze(0)
        tgt = _rot_x(torch.pi).unsqueeze(0)
        twist = _pose_error_twist(cur, tgt)
        assert abs(twist[0, 3:].norm().item() - torch.pi) < 1e-5

    def test_near_pi_rotation(self):
        angle = torch.pi - 1e-3
        cur = torch.eye(4).unsqueeze(0)
        tgt = _rot_x(angle).unsqueeze(0)
        twist = _pose_error_twist(cur, tgt)
        assert abs(twist[0, 3:].norm().item() - angle) < 1e-4

    def test_small_and_moderate_rotations(self):
        for angle in (1e-4, 0.5, 2.0):
            cur = torch.eye(4).unsqueeze(0)
            tgt = _rot_x(angle).unsqueeze(0)
            twist = _pose_error_twist(cur, tgt)
            assert abs(twist[0, 3:].norm().item() - angle) < 1e-4

    def test_flipped_candidate_not_assigned_zero_cost(self):
        """A 180-degree-error candidate must rank behind an aligned one."""
        target = torch.eye(4).unsqueeze(0)
        aligned = torch.eye(4).unsqueeze(0)
        aligned[0, 0, 3] = 0.05  # small position offset, same orientation
        flipped = _rot_x(torch.pi).unsqueeze(0)  # same position, opposite pose
        cost_aligned = _pose_error_twist(aligned, target).norm().item()
        cost_flipped = _pose_error_twist(flipped, target).norm().item()
        assert cost_flipped > 3.0
        assert cost_flipped > cost_aligned


class TestQposSeedSelSampler:
    """Seed-selection sampler tests on the library DexforceW1 robot."""

    def setup_method(self):
        config = SimulationManagerCfg(headless=True, sim_device="cpu")
        self.sim = SimulationManager(config)

        urdf = get_data_path("DexforceW1V021/DexforceW1_v02_1.urdf")
        assert os.path.isfile(urdf)

        cfg_dict = {
            "fpath": urdf,
            "control_parts": {
                "left_arm": [f"LEFT_J{i+1}" for i in range(7)],
            },
            "solver_cfg": {
                "left_arm": {
                    "class_type": "PytorchSolver",
                    "end_link_name": "left_ee",
                    "root_link_name": "left_arm_base",
                    "num_samples": 30,
                },
            },
        }
        self.robot: Robot = self.sim.add_robot(cfg=RobotCfg.from_dict(cfg_dict))
        self.sim.prepare()
        self.solver = self.robot.get_solver("left_arm")
        self.lower = self.solver.lower_qpos_limits
        self.upper = self.solver.upper_qpos_limits
        self.dof = self.solver.dof

    def teardown_method(self):
        self.sim.destroy()
        SimulationManager.flush_cleanup_queue()

    # ------------------------------------------------------------------ helpers

    def _make_sampler(self, num_samples: int, **kwargs) -> QposSeedSelSampler:
        defaults = dict(
            fk_fn=self.solver.get_fk,
            jacobian_fn=self.solver.get_jacobian,
            db_size=_DB_SIZE,
            sobol_seed=0,
        )
        defaults.update(kwargs)
        return QposSeedSelSampler(
            num_samples=num_samples,
            dof=self.dof,
            device=self.solver.device,
            **defaults,
        )

    def _reachable_targets(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """FK poses of random valid joint configurations, shape ``(n, 4, 4)``."""
        reset_all_seeds(0)
        q_true = self.lower + torch.rand(n, self.dof) * (self.upper - self.lower)
        with torch.no_grad():
            return self.solver.get_fk(q_true), q_true

    # ------------------------------------------------------------------ contract

    def test_fallback_and_target_aware_contract(self):
        sampler = self._make_sampler(num_samples=6)
        seed = torch.zeros(self.dof)

        # Without a target: parent behaviour, no database build.
        out = sampler.sample(seed, self.lower, self.upper, batch_size=3)
        assert out.shape == (18, self.dof)
        grouped = out.view(3, 6, self.dof)
        assert torch.allclose(grouped[:, 0], seed.expand(3, self.dof))
        assert sampler.database_size == 0

        # With a target: same contract, slot 0 preserved, database built.
        target, _ = self._reachable_targets(3)
        out = sampler.sample(
            seed, self.lower, self.upper, batch_size=3, target_xpos=target
        )
        assert out.shape == (18, self.dof)
        grouped = out.view(3, 6, self.dof)
        assert torch.allclose(grouped[:, 0], seed.expand(3, self.dof))
        assert (grouped >= self.lower - 1e-6).all()
        assert (grouped <= self.upper + 1e-6).all()
        assert sampler.database_size == _DB_SIZE

    def test_retrieved_seeds_are_task_space_close(self):
        """Retrieved seeds must reach far closer to the target than random."""
        sampler = self._make_sampler(num_samples=8)
        target, _ = self._reachable_targets(_NUM_TARGETS)
        out = sampler.sample(
            torch.zeros(self.dof),
            self.lower,
            self.upper,
            batch_size=_NUM_TARGETS,
            target_xpos=target,
        )
        retrieved = out.view(_NUM_TARGETS, 8, self.dof)[:, 1:]

        with torch.no_grad():
            seed_pos = self.solver.get_fk(retrieved.reshape(-1, self.dof))[:, :3, 3]
        seed_pos = seed_pos.view(_NUM_TARGETS, 7, 3)
        target_pos = target[:, :3, 3].unsqueeze(1)
        db_dist = (seed_pos - target_pos).norm(dim=-1).mean()

        reset_all_seeds(1)
        rand_q = self.lower + torch.rand(_NUM_TARGETS * 7, self.dof) * (
            self.upper - self.lower
        )
        with torch.no_grad():
            rand_pos = self.solver.get_fk(rand_q)[:, :3, 3]
        rand_dist = (rand_pos.view(_NUM_TARGETS, 7, 3) - target_pos).norm(dim=-1).mean()

        assert db_dist < 0.5 * rand_dist

    # ------------------------------------------------------------------ end to end

    def test_cfg_enabled_seed_selection_improves_get_ik(self):
        """The configuration route must wire retrieval into the real get_ik."""
        target, _ = self._reachable_targets(_NUM_TARGETS)
        k = 4

        # Default-off solver: shipped random seeding, no sampler attached.
        assert self.solver._seed_sampler is None
        reset_all_seeds(0)
        ok_shipped, _ = self.solver.get_ik(target.clone(), num_samples=k)
        shipped = ok_shipped.float().mean().item()

        # Same chain with seed selection enabled through the config.
        urdf = get_data_path("DexforceW1V021/DexforceW1_v02_1.urdf")
        sel_solver = PytorchSolverCfg(
            urdf_path=urdf,
            end_link_name="left_ee",
            root_link_name="left_arm_base",
            # Standalone solvers do not get the robot-populated joint names
            # and default nearest weights; provide both explicitly.
            joint_names=[f"LEFT_J{i+1}" for i in range(7)],
            ik_nearest_weight=[1.0] * 7,
            num_samples=30,
            enable_seed_selection=True,
            seed_db_size=_DB_SIZE,
        ).init_solver(device=self.solver.device)
        assert sel_solver._seed_sampler is not None

        reset_all_seeds(0)
        ok_sel, _ = sel_solver.get_ik(target.clone(), num_samples=k)
        sel = ok_sel.float().mean().item()

        assert sel >= shipped
        assert sel >= 0.8
        # The database was built lazily by the first get_ik call.
        assert sel_solver._seed_sampler.database_size == _DB_SIZE

    # ------------------------------------------------------------------ capped retrieval

    def test_db_smaller_than_requested_seed_count(self):
        """Reviewer case: db_size=2, num_samples=5, batch=3 -> (15, dof)."""
        sampler = self._make_sampler(num_samples=5, db_size=2)
        target, _ = self._reachable_targets(3)
        seed = torch.zeros(self.dof)
        out = sampler.sample(
            seed, self.lower, self.upper, batch_size=3, target_xpos=target
        )
        assert out.shape == (15, self.dof)
        grouped = out.view(3, 5, self.dof)
        assert torch.allclose(grouped[:, 0], seed.expand(3, self.dof))
        # Padded slots must still be valid configurations.
        assert (grouped[:, 1:] >= self.lower - 1e-6).all()
        assert (grouped[:, 1:] <= self.upper + 1e-6).all()

    def test_num_samples_exceeds_k_max(self):
        """Retrieval capped by k_max must still honour the size contract."""
        sampler = self._make_sampler(num_samples=6, k_max=2)
        target, _ = self._reachable_targets(4)
        seed = torch.zeros(self.dof)
        out = sampler.sample(
            seed, self.lower, self.upper, batch_size=4, target_xpos=target
        )
        assert out.shape == (24, self.dof)
        grouped = out.view(4, 6, self.dof)
        assert torch.allclose(grouped[:, 0], seed.expand(4, self.dof))
        assert (grouped[:, 1:] >= self.lower - 1e-6).all()
        assert (grouped[:, 1:] <= self.upper + 1e-6).all()

    def test_capped_retrieval_without_caller_seed(self):
        """use_caller_seed=False must also honour the size contract."""
        sampler = self._make_sampler(num_samples=5, db_size=2, use_caller_seed=False)
        target, _ = self._reachable_targets(3)
        out = sampler.sample(
            torch.zeros(self.dof),
            self.lower,
            self.upper,
            batch_size=3,
            target_xpos=target,
        )
        assert out.shape == (15, self.dof)
        grouped = out.view(3, 5, self.dof)
        assert (grouped >= self.lower - 1e-6).all()
        assert (grouped <= self.upper + 1e-6).all()

    # ------------------------------------------------------------------ rebuild

    def test_limits_change_triggers_rebuild(self):
        sampler = self._make_sampler(num_samples=4)
        target, _ = self._reachable_targets(3)
        seed = torch.zeros(self.dof)
        sampler.sample(seed, self.lower, self.upper, batch_size=3, target_xpos=target)
        first_db = sampler._db_qpos

        # Same limits: no rebuild.
        sampler.sample(seed, self.lower, self.upper, batch_size=3, target_xpos=target)
        assert sampler._db_qpos is first_db

        # Narrowed limits: rebuild, and every seed obeys the new bounds.
        narrow_lo = self.lower * 0.5
        narrow_hi = self.upper * 0.5
        out = sampler.sample(
            seed, narrow_lo, narrow_hi, batch_size=3, target_xpos=target
        )
        assert sampler._db_qpos is not first_db
        retrieved = out.view(3, 4, self.dof)[:, 1:]
        assert (retrieved >= narrow_lo - 1e-6).all()
        assert (retrieved <= narrow_hi + 1e-6).all()

    # ------------------------------------------------------------------ validation

    def test_invalid_inputs_raise(self):
        sampler = self._make_sampler(num_samples=4)
        target, _ = self._reachable_targets(3)
        with pytest.raises(ValueError):
            sampler.sample(
                torch.zeros(self.dof),
                self.lower,
                self.upper,
                batch_size=3,
                target_xpos=torch.eye(4),
            )
        with pytest.raises(ValueError):
            sampler.sample(
                torch.zeros(5, self.dof + 1),
                self.lower,
                self.upper,
                batch_size=3,
                target_xpos=target,
            )
        with pytest.raises(ValueError):
            self._make_sampler(num_samples=4, db_size=0)
        with pytest.raises(ValueError):
            self._make_sampler(num_samples=4, rot_scale=0.0)
