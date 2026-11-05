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

"""Deterministic constrained candidates and source command/query facts."""

from __future__ import annotations
from pathlib import Path
from dataclasses import replace
import hashlib
import json
import torch
from ._ordinary_pick_fixture import ordinary_fixture, OrdinaryCandidates


class ConstrainedCandidates(OrdinaryCandidates):
    def __init__(self, mode="safe"):
        super().__init__()
        self.mode = mode

    def get_valid_grasp_poses(self, *, obj_poses, **kwargs):
        values = []
        for obj in obj_poses:
            poses = obj[None].repeat(3, 1, 1)
            poses[:, :3, :3] = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
            poses[:, 2, 3] += torch.tensor([-0.01, 0.01, 0.02])
            poses[:, 0, 3] += torch.tensor([0.0, 0.0, 0.005])
            costs = torch.tensor([0.0, 0.1, 0.2])
            if self.mode == "direction_fallback" and torch.equal(
                kwargs["approach_direction"].reshape(-1, 3),
                torch.tensor([[0.0, 0.0, -1.0]]),
            ):
                costs[:] = torch.inf
            values.append((poses, costs))
        return values


def constrained_fixture(directory: Path, mode: str):
    gen, context, sem, calls, ik = ordinary_fixture(directory)
    if mode.startswith("raw_"):
        # The coarse endpoint-built path leaves the declared joint range. The
        # source's continuous raw candidate paths use IK directly and remain
        # inside it. Keep velocity limits high so the exact position-limit gate
        # (rather than an unrelated speed failure) rejects the coarse plan.
        path = Path(gen.robot.cfg.fpath)
        path.write_text(path.read_text().replace('velocity="1"', 'velocity="100"'))
        gen.robot.get_qvel_limits.side_effect = lambda name=None, **kw: torch.full(
            (1, len(kw.get("joint_ids", gen.robot.get_joint_ids(name=name)))), 100.0
        )
        original_generate = gen._generate_once

        def unsafe_coarse_path(targets, *, options):
            result = original_generate(targets, options=options)
            positions = result.positions.clone()
            positions[:, 1:, 0] += 3.0
            return replace(result, positions=positions)

        gen._generate_once = unsafe_coarse_path
    if mode in {"opening", "opening_prefix"}:
        qpos = context.robot.qpos.clone()
        qpos[:, 3:] = -0.01
        context = replace(context, robot=replace(context.robot, qpos=qpos))
        gen.robot.get_qpos.return_value = qpos
    if mode == "initial_failed":
        original = gen._generate_once

        def generate(targets, *, options):
            result = original(targets, options=options)
            if len(calls) == 1:
                return replace(result, success=torch.tensor([False]))
            return result

        gen._generate_once = generate
    if mode.startswith("cartesian_"):
        original_generate = gen._generate_once

        def raw(targets, *, options):
            torch.rand(())
            result = original_generate(targets, options=options)
            if (
                mode == "cartesian_exhausted"
                or mode == "cartesian_dense"
                and options.sample_count < 260
            ):
                return replace(result, success=torch.tensor([False]))
            return result

        gen._generate_once = raw
    inspections = []

    def inspect(robot, positions, part):
        if mode == "start_rejected":
            valid = False
        elif mode == "unavailable":
            result = {"valid": False, "unavailable": "controlled_geometry_unavailable"}
            inspections.append(result)
            return result
        elif mode == "raw_exhausted" and len(positions) > 1:
            valid = False
        elif mode in {"alternate", "exhausted"} and len(positions) > 1:
            valid = mode == "alternate" and float(positions[-1, 0]) > 0.203
        else:
            valid = True
        result = (
            {"valid": True, "frames": len(positions), "scope": "discrete_robot_meshes"}
            if valid
            else {
                "valid": False,
                "frame": 0,
                "kind": "self",
                "pairs": [["arm", "passive"]],
                "scope": "discrete_robot_meshes",
            }
        )
        inspections.append(dict(result))
        return result

    return gen, context, sem, calls, ik, inspect, inspections


def record(plan, calls, ik, inspections):
    trajectory = plan.joint_trajectory
    digest = lambda value: hashlib.sha256(
        value.detach().cpu().numpy().tobytes()
    ).hexdigest()
    return dict(
        success=plan.plan_success.tolist(),
        positions=digest(trajectory.positions),
        dt=digest(trajectory.dt),
        segments=[(s.name, s.start, s.stop) for s in plan.segments],
        metadata=plan.diagnostics.metadata,
        failure=(
            None
            if plan.diagnostics.failure is None
            else dict(
                code=plan.diagnostics.failure.code,
                retryable=plan.diagnostics.failure.retryable,
            )
        ),
        calls=[
            {
                **{key: value for key, value in call.items() if key != "targets"},
                "target_count": len(call["targets"]),
                "targets_sha256": hashlib.sha256(
                    json.dumps(
                        call["targets"],
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode()
                ).hexdigest(),
            }
            for call in calls
        ],
        ik=ik,
        inspections=inspections,
        rng=digest(torch.get_rng_state()),
    )


def force_opening_jump(action, request, context, plan):
    trajectory = plan.joint_trajectory
    positions = trajectory.positions.clone()
    positions[:, : next(s.stop for s in plan.segments if s.name == "approach"), 3:] = (
        0.04
    )
    trajectory = replace(trajectory, positions=positions)
    commands = action._joint_command_sequence(
        request, trajectory, active_mask=plan.plan_success
    )
    return replace(
        plan,
        joint_trajectory=trajectory,
        commands=commands,
        tracking=action._tracking_sequence(request, commands),
    )
