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
"""Diagnose native peg reachability through real joint-position actions.

Run: python -m scripts.benchmark.assembly.peg_reachability --mode oracle --num-envs 64

The oracle uses privileged part/fixture poses and payload properties. It is a
control upper-bound diagnostic, never a learned-policy or paper reproduction
score. No object or joint physical state is written after the initial reset;
all subsequent control uses the task's ordinary seven-dimensional action.
The nominal reset distribution, contacts, grasp and success guards are retained.
Private solver inspection is read-only and confined to this benchmark.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import inspect
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import time
import traceback
from typing import Any

import numpy as np
import torch
import warp as wp

__all__: list[str] = []


def _synchronize(device: torch.device) -> None:
    """Complete participating Warp and Torch streams."""
    wp.synchronize_device(str(device))
    torch.cuda.synchronize(device)


def _memory(device: torch.device) -> dict[str, int | None]:
    """Sample process RSS and total CUDA-device usage, including other users."""
    try:
        import psutil

        rss = psutil.Process().memory_info().rss
    except ImportError:
        rss = None
    free, total = torch.cuda.mem_get_info(device)
    return {"process_rss_bytes": rss, "total_device_used_bytes": total - free}


class _PhysicsHealth:
    """Observe native health at substeps or before control-boundary reset."""

    def __init__(
        self,
        env: Any,
        *,
        scope: str = "substep",
        strict: bool = False,
        include_world_flags: bool = True,
    ) -> None:
        if scope not in {"control", "substep"}:
            raise ValueError("Health scope must be control or substep.")
        self._scope = scope
        self._strict = strict
        self._include_world_flags = include_world_flags
        self._clock_samples = 0
        self._configured_physics_dt = getattr(env, "physics_dt", None)
        self.values: dict[str, Any] = {
            "observations": 0,
            "control_boundaries_observed": 0,
            "physical_substeps_observed": 0,
            "physics_substeps_requested_in_completed_controls": 0,
            "controls_without_explicit_substep_count": 0,
            "overflow_available": False,
            "overflow_seen": False,
            "capacity_overflow_seen": False,
            "convergence_budget_exhausted_seen": False,
            "unclassified_overflow_seen": False,
            "max_nonzero_overflow_flags": 0,
            "overflow_flags": {},
            "flagged_world_observation_counts": {},
            "contact_maxima": {},
            "scope": (
                "after sim.update returns, before BaseEnv auto-reset; sticky overflow flags cover the full control interval, contact maxima sample its last physical substep"
                if scope == "control"
                else "read-only solver buffers after each physics substep; synchronous host copies included in diagnostic timing"
            ),
            "strict": strict,
            "control_clock_observations": [],
            "control_clock_scope": "clock read immediately before/after the first two completed sim.update calls; no individual substep clock read is claimed",
            "native_state": {
                "fields": ["qpos", "qvel", "qacc"],
                "scope": (
                    "complete native qpos/qvel/qacc arrays after sim.update returns and before BaseEnv auto-reset; states between control boundaries are not sampled"
                    if scope == "control"
                    else "complete native qpos/qvel/qacc arrays after each observed physical substep and before BaseEnv auto-reset"
                ),
                "readback": "synchronous native array copies followed by CPU finite reduction; summaries retain at most 16 first bad world IDs per field",
                "available": False,
                "missing_fields": [],
                "observations": 0,
                "complete_observations": 0,
                "nonfinite_seen": False,
                "all_samples_finite": None,
                "first_nonfinite": None,
                "field_summaries": {},
            },
        }
        try:
            part = env.sim.get_rigid_object("peg")
            self.solver = part._entities[0]._physics_binding._runtime.solver
            self.data = self.solver.mjw_data
            from mujoco_warp._src.types import OverflowType

            self._overflow_names = {
                name: int(value) for name, value in OverflowType.__members__.items()
            }
            convergence = {"ITERATIONS", "LS_ITERATIONS"}
            self._convergence_mask = sum(
                value
                for name, value in self._overflow_names.items()
                if name in convergence
            )
            self._capacity_mask = sum(
                value
                for name, value in self._overflow_names.items()
                if name not in convergence
            )
            self.values["overflow_masks"] = {
                "capacity": self._capacity_mask,
                "convergence_budget": self._convergence_mask,
                "meaning": "ITERATIONS/LS_ITERATIONS are convergence budgets; other installed enum bits indicate capacity/collision loss. Flags are sticky until native reset.",
            }
            self.values["actual_capacities"] = {
                name: int(getattr(self.data, name))
                for name in ("naconmax", "njmax", "njmax_nnz")
                if getattr(self.data, name, None) is not None
            }
            self.values["counter_scope"] = {
                "nacon": "global batch contact count; naconmax is global allocation, not a hard per-world limit",
                "ncollision": "global broadphase candidate count",
                "nefc": "constraints per world; njmax is a hard per-world limit",
            }
            options = getattr(getattr(self.solver, "mjw_model", None), "opt", None)
            actual_options = {}

            def compact_option(value: Any) -> Any:
                array = np.asarray(value.numpy() if hasattr(value, "numpy") else value)
                if array.size == 1:
                    return array.item()
                unique = np.unique(array)
                if unique.size == 1:
                    return {"shape": list(array.shape), "constant": unique.item()}
                return {
                    "shape": list(array.shape),
                    "distinct_values": int(unique.size),
                    "first_distinct_values": unique[:16].tolist(),
                    "minimum": array.min().item(),
                    "maximum": array.max().item(),
                }

            for name in (
                "cone",
                "impratio_invsqrt",
                "iterations",
                "ls_iterations",
                "ccd_iterations",
                "timestep",
                "integrator",
                "tolerance",
            ):
                value = getattr(options, name, None)
                if value is not None:
                    actual_options[name] = compact_option(value)
                    if name == "impratio_invsqrt":
                        array = np.asarray(
                            value.numpy() if hasattr(value, "numpy") else value
                        )
                        actual_options["impratio_from_invsqrt"] = compact_option(
                            1.0 / np.square(array)
                        )
            self._compact_option = compact_option
            self.values["prepared_solver_options"] = actual_options
            self.values["prepared_solver_options_scope"] = (
                "captured before the first live sim.update; timestep may still be the solver preparation default and is not evidence of the effective step duration"
            )
            path = Path(inspect.getfile(type(self.solver)))
            self.values.update(
                private_path="peg._entities[0]._physics_binding._runtime.solver.mjw_data",
                solver_source=str(path),
                solver_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        except (AttributeError, ImportError, IndexError, TypeError) as error:
            self.data = None
            self.values["unavailable_reason"] = str(error)
        if strict and (self.data is None or not hasattr(self.data, "overflow")):
            raise RuntimeError(
                "Strict native training health requires readable MuJoCo Warp overflow flags."
            )
        native_state = self.values["native_state"]
        native_state["missing_fields"] = [
            name
            for name in native_state["fields"]
            if getattr(self.data, name, None) is None
        ]
        native_state["available"] = not native_state["missing_fields"]
        if strict and native_state["missing_fields"]:
            raise RuntimeError(
                "Strict native health requires finite-state fields: "
                f"{native_state['missing_fields']}"
            )
        self._original_update = env.sim.update

        def observed_update(*args: Any, **kwargs: Any) -> None:
            before_clock = self._native_clock() if self._clock_samples < 2 else None
            if scope == "control":
                self._original_update(*args, **kwargs)
                requested_steps = kwargs.get("step", args[1] if len(args) > 1 else None)
                if requested_steps is None:
                    self.values["controls_without_explicit_substep_count"] += 1
                else:
                    self.values[
                        "physics_substeps_requested_in_completed_controls"
                    ] += int(requested_steps)
                self._record_clock(before_clock, args, kwargs)
                self.observe()
                return
            after_substep = kwargs.get("after_substep")

            def observe(dt: float) -> None:
                if after_substep is not None:
                    after_substep(dt)
                self.observe()

            kwargs["after_substep"] = observe
            self._original_update(*args, **kwargs)
            self._record_clock(before_clock, args, kwargs)

        env.sim.update = observed_update

    def _native_clock(self) -> np.ndarray | None:
        """Copy a native clock only for bounded initial control diagnostics."""
        value = getattr(self.data, "time", None)
        if value is None:
            return None
        return np.asarray(value.numpy() if hasattr(value, "numpy") else value).copy()

    def _record_clock(
        self,
        before: np.ndarray | None,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        """Separate prepared dt from two effective control-clock measurements."""
        if self._clock_samples >= 2:
            return
        self._clock_samples += 1
        after = self._native_clock()
        options = getattr(
            getattr(getattr(self, "solver", None), "mjw_model", None), "opt", None
        )
        timestep = getattr(options, "timestep", None)
        if timestep is not None and self._clock_samples == 1:
            self.values["effective_solver_timestep_after_first_update"] = (
                self._compact_option(timestep)
            )
        requested_steps = kwargs.get("step", args[1] if len(args) > 1 else None)
        requested_dt = kwargs.get("physics_dt", args[0] if args else None)
        if requested_dt is None:
            requested_dt = self._configured_physics_dt
        expected = (
            float(requested_dt) * int(requested_steps)
            if requested_dt is not None and requested_steps is not None
            else None
        )
        sample = {
            "completed_update": self._clock_samples,
            "explicit_step_count": (
                int(requested_steps) if requested_steps is not None else None
            ),
            "requested_physics_dt_seconds": (
                float(requested_dt) if requested_dt is not None else None
            ),
            "expected_elapsed_seconds_from_request": expected,
            "native_clock_available": before is not None and after is not None,
            "individual_substep_clock_measured": False,
        }
        if before is not None and after is not None:
            delta = after.astype(np.float64) - before.astype(np.float64)
            sample.update(
                native_clock_before=self._compact_option(before),
                native_clock_after=self._compact_option(after),
                observed_control_elapsed_seconds=self._compact_option(delta),
                all_world_clock_deltas_match_request=(
                    bool(np.allclose(delta, expected, rtol=1e-5, atol=1e-7))
                    if expected is not None
                    else None
                ),
            )
        self.values["control_clock_observations"].append(sample)

    def _observe_native_state(self) -> None:
        """Record bounded finite-state diagnostics without changing native arrays."""
        health = self.values["native_state"]
        health["observations"] += 1
        missing = []
        for name in health["fields"]:
            value = getattr(self.data, name, None)
            if value is None:
                missing.append(name)
                continue
            try:
                array = np.asarray(value.numpy() if hasattr(value, "numpy") else value)
                if array.ndim != 2:
                    raise ValueError(
                        f"Expected a world-major matrix; got {array.shape}."
                    )
                finite = np.isfinite(array)
            except Exception as error:
                missing.append(name)
                health.setdefault("first_read_errors", {}).setdefault(name, str(error))
                continue
            bad_worlds = np.flatnonzero(~finite.all(axis=1))
            count = int(bad_worlds.size)
            summary = health["field_summaries"].setdefault(
                name,
                {
                    "shape": list(array.shape),
                    "observations": 0,
                    "nonfinite_world_observations": 0,
                    "nonfinite_value_observations": 0,
                    "max_nonfinite_worlds": 0,
                    "first_bad_world_ids": [],
                },
            )
            summary["observations"] += 1
            summary["nonfinite_world_observations"] += count
            summary["nonfinite_value_observations"] += int(np.count_nonzero(~finite))
            summary["max_nonfinite_worlds"] = max(
                summary["max_nonfinite_worlds"], count
            )
            if count:
                health["nonfinite_seen"] = True
                if not summary["first_bad_world_ids"]:
                    summary["first_bad_world_ids"] = bad_worlds[:16].tolist()
                if health["first_nonfinite"] is None:
                    health["first_nonfinite"] = {
                        "observation": self.values["observations"],
                        "field": name,
                        "bad_world_count": count,
                        "first_bad_world_ids": bad_worlds[:16].tolist(),
                    }
        health["missing_fields"] = sorted(set(health["missing_fields"]) | set(missing))
        health["available"] = not health["missing_fields"]
        if not missing:
            health["complete_observations"] += 1
        if health["nonfinite_seen"]:
            health["all_samples_finite"] = False
        elif health["available"] and health["complete_observations"]:
            health["all_samples_finite"] = True
        else:
            health["all_samples_finite"] = None

    def observe(self) -> None:
        """Read diagnostic counters/state; never mutate native model or data."""
        self.values["observations"] += 1
        counter = (
            "control_boundaries_observed"
            if self._scope == "control"
            else "physical_substeps_observed"
        )
        self.values[counter] += 1
        if self.data is None:
            return
        self._observe_native_state()
        if hasattr(self.data, "overflow"):
            flags = np.asarray(self.data.overflow.numpy())
            count = int(np.count_nonzero(flags))
            self.values["overflow_available"] = True
            self.values["overflow_seen"] |= count > 0
            self.values["capacity_overflow_seen"] |= bool(
                np.any(flags & self._capacity_mask)
            )
            self.values["convergence_budget_exhausted_seen"] |= bool(
                np.any(flags & self._convergence_mask)
            )
            self.values["unclassified_overflow_seen"] |= bool(
                np.any(flags & ~(self._capacity_mask | self._convergence_mask))
            )
            if self._include_world_flags:
                self.values["last_overflow_flags_per_world"] = flags.tolist()
            else:
                flagged = np.flatnonzero(flags)
                self.values["last_flagged_world_count"] = int(flagged.size)
                self.values["first_flagged_world_ids"] = flagged[:16].tolist()
            for name, bit in self._overflow_names.items():
                world_count = int(np.count_nonzero(flags & bit))
                self.values["overflow_flags"][name] = max(
                    self.values["overflow_flags"].get(name, 0), world_count
                )
                observations = self.values["flagged_world_observation_counts"]
                observations[name] = observations.get(name, 0) + world_count
            self.values["max_nonzero_overflow_flags"] = max(
                count, self.values["max_nonzero_overflow_flags"]
            )
        for name in ("nacon", "ncollision", "nefc"):
            value = getattr(self.data, name, None)
            if value is None:
                continue
            array = np.asarray(value.numpy() if hasattr(value, "numpy") else value)
            if array.size:
                old = self.values["contact_maxima"].setdefault(
                    name, {"max_element": 0, "max_sum": 0, "shape": list(array.shape)}
                )
                old["max_element"] = max(old["max_element"], int(array.max()))
                old["max_sum"] = max(old["max_sum"], int(array.sum()))
        native_state = self.values["native_state"]
        if self._strict and not native_state["available"]:
            raise RuntimeError(
                "Strict native health cannot read required finite-state fields before auto-reset: "
                f"{native_state['missing_fields']}"
            )
        if self._strict and native_state["nonfinite_seen"]:
            raise RuntimeError(
                "Strict native health detected nonfinite qpos/qvel/qacc before auto-reset: "
                f"{native_state['first_nonfinite']}"
            )
        if self._strict and (
            self.values["capacity_overflow_seen"]
            or self.values["unclassified_overflow_seen"]
        ):
            raise RuntimeError(
                "Strict native training health detected capacity/collision loss "
                "or an unclassified overflow bit before auto-reset: "
                f"{self.values['overflow_flags']}"
            )

    def restore(self, env: Any) -> None:
        """Remove the diagnostic observer before ordinary teardown."""
        env.sim.update = self._original_update


class _Oracle:
    """Lift, align, insert and hold using privileged pose feedback and DLS."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self.phase = torch.ones(env.num_envs, dtype=torch.long, device=env.device)
        self.alignment_hold = torch.zeros_like(self.phase)
        self.part = env.sim.get_rigid_object("peg")
        self.fixture = env.sim.get_rigid_object("plate")
        self.base_rotation = env._assembly_robot_base[:3, :3]
        self.identity = torch.eye(6, device=env.device)[None]
        self.thickness = float(env.assembly_asset.config["plate_thickness"])
        self.mass = float(env._moving_definition.mass)
        self.com = torch.tensor(
            env._moving_definition.com, dtype=torch.float32, device=env.device
        )
        self.gravity = torch.tensor(
            env.sim_cfg.physics_cfg.gravity, dtype=torch.float32, device=env.device
        )
        if self.gravity.ndim == 0:
            self.gravity = torch.tensor(
                [0.0, 0.0, float(self.gravity)], device=env.device
            )
        self.stiffness = float(getattr(env, "joint_stiffness", 80.0))

    def action(self) -> torch.Tensor:
        """Produce target increments; source physical states remain read-only."""
        env = self.env
        part = self.part.get_local_pose(to_matrix=True)
        fixture = self.fixture.get_local_pose(to_matrix=True)
        relative = torch.linalg.solve(fixture, part)
        bottom = (
            env._assembly_bottom_vertices[None] @ relative[:, :3, :3].transpose(1, 2)
            + relative[:, None, :3, 3]
        )
        low = bottom[:, :, 2].amin(dim=-1) - env._fixture_zmax
        high = bottom[:, :, 2].amax(dim=-1) - env._fixture_zmax
        center_height = (low + high) / 2
        radial = torch.linalg.vector_norm(relative[:, :2, 3], dim=-1)
        rotation_error = fixture[:, :3, :3] @ part[:, :3, :3].transpose(1, 2)
        omega = 0.5 * torch.stack(
            [
                rotation_error[:, 2, 1] - rotation_error[:, 1, 2],
                rotation_error[:, 0, 2] - rotation_error[:, 2, 0],
                rotation_error[:, 1, 0] - rotation_error[:, 0, 1],
            ],
            dim=-1,
        )
        sine = torch.linalg.vector_norm(omega, dim=-1)
        cosine = ((rotation_error.diagonal(dim1=1, dim2=2).sum(-1) - 1) / 2).clamp(
            -1, 1
        )
        angle = torch.atan2(sine, cosine)
        omega = omega * (angle / sine.clamp_min(1e-8))[:, None]
        self.phase = torch.where((self.phase == 1) & (low >= 0.00125), 2, self.phase)
        aligned = (radial < 0.00025) & (angle < 0.015) & (high - low < 0.00075)
        self.alignment_hold = torch.where(aligned, self.alignment_hold + 1, 0)
        self.phase = torch.where(
            (self.phase == 2) & (self.alignment_hold >= 3), 3, self.phase
        )
        within_target = ((low + self.thickness).abs() < 0.0006) & (
            (high + self.thickness).abs() < 0.0006
        )
        self.phase = torch.where((self.phase == 3) & within_target, 4, self.phase)
        dp = torch.zeros_like(relative[:, :3, 3])
        dp[:, :2] = torch.where(
            (self.phase >= 2)[:, None], -relative[:, :2, 3], 0
        ).clamp(-0.0006, 0.0006)
        above = self.phase <= 2
        vertical = torch.where(above, 0.0015 - low, -self.thickness - center_height)
        descend_ok = (radial < 0.0005) & (angle < 0.025) & (high - low < 0.0015)
        vertical = torch.where(above | descend_ok | (vertical > 0), vertical, 0)
        dp[:, 2] = vertical.clamp(-0.0003, 0.0003)
        omega = omega * (0.008 / angle.clamp_min(0.008))[:, None]
        q = env.robot.get_qpos(name="arm")
        tcp = (
            env._assembly_robot_base
            @ env._assembly_chain.forward_kinematics(q).get_matrix()
        )
        # Rotating around the part center also moves the TCP lever arm.
        # Without this term orientation correction itself destroys XY alignment.
        world_dp = (fixture[:, :3, :3] @ dp[:, :, None]).squeeze(-1)
        world_dp += torch.cross(omega, tcp[:, :3, 3] - part[:, :3, 3], dim=-1)
        error = torch.cat(
            [world_dp @ self.base_rotation, omega @ self.base_rotation], dim=-1
        )
        jacobian = env._assembly_chain.jacobian(q)
        dq = (
            jacobian.transpose(1, 2)
            @ torch.linalg.solve(
                jacobian @ jacobian.transpose(1, 2) + self.identity * 1e-5,
                error[:, :, None],
            )
        ).squeeze(-1)
        # The environment compensates robot gravity, but not the free peg.
        # Support its weight through a PD target offset, never an extra force.
        com = (part[:, :3, :3] @ self.com[:, None]).squeeze(-1) + part[:, :3, 3]
        support = (-self.mass * self.gravity).expand(env.num_envs, -1)
        moment = torch.cross(com - tcp[:, :3, 3], support, dim=-1)
        wrench = torch.cat(
            [support @ self.base_rotation, moment @ self.base_rotation], dim=-1
        )
        payload_bias = (jacobian.transpose(1, 2) @ wrench[:, :, None]).squeeze(
            -1
        ) / self.stiffness
        action = ((6.0 * dq + payload_bias) / 0.02).clamp(-5.0, 5.0)
        action[env._elapsed_steps < int(getattr(env, "grasp_warmup_steps", 10))] = 0
        return action


class _GripHold(_Oracle):
    """Solve one 5 mm lift, then hold fixed joint targets from encoders only."""

    def __init__(self, env: Any, hold_seconds: float) -> None:
        super().__init__(env)
        self.hold_seconds = hold_seconds
        q = env.robot.get_qpos(name="arm").clone()
        initial_tcp = (
            env._assembly_robot_base
            @ env._assembly_chain.forward_kinematics(q).get_matrix()
        )
        initial_part = self.part.get_local_pose(to_matrix=True)
        initial_inhand = torch.linalg.solve(initial_tcp, initial_part)
        self.initial_inhand_z = initial_inhand[:, 2, 3].clone()
        self.assumed_com_in_tcp = (
            initial_inhand[:, :3, :3] @ self.com[:, None]
        ).squeeze(-1) + initial_inhand[:, :3, 3]
        self.target_tcp = initial_tcp.clone()
        self.target_tcp[:, 2, 3] += 0.005
        limits = env.robot.get_qpos_limits(name="arm")
        base_rotation = self.base_rotation
        # Preparation-only IK, with no simulator or object-state writes.
        # Once solved this joint target is immutable for the whole hold.
        for _ in range(10):
            tcp = (
                env._assembly_robot_base
                @ env._assembly_chain.forward_kinematics(q).get_matrix()
            )
            dr = self.target_tcp[:, :3, :3] @ tcp[:, :3, :3].transpose(1, 2)
            omega = 0.5 * torch.stack(
                [
                    dr[:, 2, 1] - dr[:, 1, 2],
                    dr[:, 0, 2] - dr[:, 2, 0],
                    dr[:, 1, 0] - dr[:, 0, 1],
                ],
                dim=-1,
            )
            error = torch.cat(
                [
                    (self.target_tcp[:, :3, 3] - tcp[:, :3, 3]) @ base_rotation,
                    omega @ base_rotation,
                ],
                dim=-1,
            )
            jacobian = env._assembly_chain.jacobian(q)
            dq = (
                jacobian.transpose(1, 2)
                @ torch.linalg.solve(
                    jacobian @ jacobian.transpose(1, 2) + self.identity * 1e-5,
                    error[:, :, None],
                )
            ).squeeze(-1)
            q = (q + dq.clamp(-0.05, 0.05)).clamp(limits[:, :, 0], limits[:, :, 1])
        self.target_q = q.detach().clone()
        solved = (
            env._assembly_robot_base
            @ env._assembly_chain.forward_kinematics(q).get_matrix()
        )
        self.ik_error = torch.linalg.vector_norm(
            solved[:, :3, 3] - self.target_tcp[:, :3, 3], dim=-1
        )
        self.hold_start = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )
        self.first_grasp_loss = torch.full_like(self.hold_start, -1)
        self.elapsed_hold = torch.zeros(env.num_envs, device=env.device)
        self.maximum_slip = torch.zeros_like(self.elapsed_hold)
        self.maximum_hand_error = torch.zeros_like(self.elapsed_hold)
        self.latest_slip = torch.zeros_like(self.elapsed_hold)

    def action(self) -> torch.Tensor:
        """Hold the solved target; never read the part or fixture after setup."""
        env = self.env
        q = env.robot.get_qpos(name="arm")
        tcp = (
            env._assembly_robot_base
            @ env._assembly_chain.forward_kinematics(q).get_matrix()
        )
        jacobian = env._assembly_chain.jacobian(q)
        # Payload support assumes the *initial* in-hand COM remains fixed.
        # Actual slip is observed only in the separate diagnostic below.
        lever = (tcp[:, :3, :3] @ self.assumed_com_in_tcp[:, :, None]).squeeze(-1)
        support = (-self.mass * self.gravity).expand(env.num_envs, -1)
        moment = torch.cross(lever, support, dim=-1)
        wrench = torch.cat(
            [support @ self.base_rotation, moment @ self.base_rotation], dim=-1
        )
        bias = (jacobian.transpose(1, 2) @ wrench[:, :, None]).squeeze(
            -1
        ) / self.stiffness
        action = ((self.target_q - q + bias) / 0.02).clamp(-5.0, 5.0)
        action[env._elapsed_steps < int(getattr(env, "grasp_warmup_steps", 10))] = 0
        return action

    def observe(self, info: dict[str, Any], step: int) -> None:
        """Measure actual slip without sending it to the hold controller."""
        env = self.env
        q = env.robot.get_qpos(name="arm")
        tcp = (
            env._assembly_robot_base
            @ env._assembly_chain.forward_kinematics(q).get_matrix()
        )
        relative = torch.linalg.solve(tcp, self.part.get_local_pose(to_matrix=True))
        self.latest_slip = relative[:, 2, 3] - self.initial_inhand_z
        self.maximum_slip = torch.maximum(self.maximum_slip, self.latest_slip.abs())
        hand_error = torch.linalg.vector_norm(
            tcp[:, :3, 3] - self.target_tcp[:, :3, 3], dim=-1
        )
        self.maximum_hand_error = torch.maximum(self.maximum_hand_error, hand_error)
        reached = (hand_error <= 0.0005) & (
            env._elapsed_steps >= int(getattr(env, "grasp_warmup_steps", 10))
        )
        self.hold_start = torch.where(
            (self.hold_start < 0) & reached, step, self.hold_start
        )
        loss = ~info["metrics"]["grasped"] & (env._elapsed_steps > 25)
        self.first_grasp_loss = torch.where(
            (self.first_grasp_loss < 0) & loss, step, self.first_grasp_loss
        )
        self.elapsed_hold = torch.where(
            self.hold_start >= 0, (step - self.hold_start) * env.step_dt, 0
        )

    def finished(self) -> torch.Tensor:
        """Return completed holds or independently recorded grasp failures."""
        return (self.elapsed_hold >= self.hold_seconds) | (self.first_grasp_loss >= 0)

    def summary(self) -> dict[str, Any]:
        """Report grip retention separately from task success or policy scores."""
        retained = (self.elapsed_hold >= self.hold_seconds) & (
            self.first_grasp_loss < 0
        )
        return {
            "scope": "5 mm one-time encoder-FK/DLS lift, then immutable joint targets; initial COM support only; measured part slip never feeds control",
            "hold_seconds_requested": self.hold_seconds,
            "lift_m": 0.005,
            "target_reached_position_tolerance_m": 0.0005,
            "target_q": self.target_q.tolist(),
            "target_tcp": self.target_tcp.tolist(),
            "preparation_ik_translation_error_m": self.ik_error.tolist(),
            "hold_start_control": self.hold_start.tolist(),
            "hold_seconds_observed": self.elapsed_hold.tolist(),
            "first_grasp_loss_control": self.first_grasp_loss.tolist(),
            "retained_grasp_for_requested_hold": retained.tolist(),
            "retention_fraction": float(retained.float().mean()),
            "latest_inhand_axial_slip_m": self.latest_slip.tolist(),
            "maximum_absolute_inhand_axial_slip_m": self.maximum_slip.tolist(),
            "maximum_hand_target_position_error_m": self.maximum_hand_error.tolist(),
        }


def _grip_snapshot(env: Any, initial_inhand_z: torch.Tensor) -> dict[str, Any]:
    """Sum every valid part-pad contact, with tangential vectors on each pad."""
    q = env.robot.get_qpos(name="arm")
    tcp = (
        env._assembly_robot_base
        @ env._assembly_chain.forward_kinematics(q).get_matrix()
    )
    part = env.sim.get_rigid_object("peg").get_local_pose(to_matrix=True)
    inhand = torch.linalg.solve(tcp, part)
    sensor = env.get_sensor("grasp_contacts")
    data = sensor.get_data()
    pairs, valid = data["user_ids"], data["is_valid"]
    pads = env._assembly_pad_actor_ids
    part_ids = env._assembly_part_actor_ids
    normal, tangential, tangential_norm, counts = [], [], [], []
    for side in range(pads.shape[1]):
        pad_first = (pairs[:, :, 0] == pads[:, side, None]) & (
            pairs[:, :, 1] == part_ids[:, None]
        )
        pad_second = (pairs[:, :, 1] == pads[:, side, None]) & (
            pairs[:, :, 0] == part_ids[:, None]
        )
        match = valid & (pad_first | pad_second)
        counts.append(match.sum(dim=-1))
        normal.append(torch.where(match, data["impulse"], 0).sum(dim=-1))
        on_pad = torch.where(pad_first[:, :, None], data["friction"], -data["friction"])
        on_pad = torch.where(match[:, :, None], on_pad, 0)
        tangential.append(on_pad.sum(dim=1))
        tangential_norm.append(torch.linalg.vector_norm(on_pad, dim=-1).sum(dim=-1))
    return {
        "scope": "last physical substep; all valid actor-ID-matched part-pad contacts; impulses in N s, not forces or control-interval sums",
        "pad_order": (
            ["right", "left"] if pads.shape[1] == 2 else list(range(pads.shape[1]))
        ),
        "inhand_position_m": inhand[:, :3, 3].tolist(),
        "inhand_axial_slip_m": (inhand[:, 2, 3] - initial_inhand_z).tolist(),
        "jaw_qpos": env.robot.get_qpos(name="eef").tolist(),
        "jaw_target_qpos": env.robot.get_qpos(name="eef", target=True).tolist(),
        "valid_contacts_per_pad": torch.stack(counts, dim=1).tolist(),
        "normal_impulse_sum_per_pad_Ns": torch.stack(normal, dim=1).tolist(),
        "tangential_impulse_vector_sum_on_pad_Ns": torch.stack(
            tangential, dim=1
        ).tolist(),
        "tangential_impulse_norm_sum_per_pad_Ns": torch.stack(
            tangential_norm, dim=1
        ).tolist(),
    }


def _versions() -> dict[str, Any]:
    """Record package and source identities without invoking a native World."""
    result: dict[str, Any] = {"python": platform.python_version()}
    for name in (
        "embodichain",
        "dexsim-engine",
        "torch",
        "warp-lang",
        "newton",
        "mujoco",
        "mujoco-warp",
        "mjfit",
        "pytorch-kinematics",
    ):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    for name, command in (
        ("git_head", ["git", "rev-parse", "HEAD"]),
        ("git_dirty", ["git", "status", "--porcelain"]),
    ):
        try:
            value = subprocess.check_output(command, text=True).strip()
            result[name] = bool(value) if name == "git_dirty" else value
        except (OSError, subprocess.SubprocessError):
            result[name] = None
    return result


def _run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute one seeded first episode per arena, with no subsequent reset."""
    import embodichain_tasks  # noqa: F401 -- official registration
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg
    from embodichain.lab.gym.utils.registration import make
    from embodichain.lab.sim import SimulationManager
    from embodichain.utils.utility import load_config

    device = torch.device(args.device)
    env, health = None, None
    result: dict[str, Any] = {"status": "failed", "rows": [], "finite": True}
    wp.init()
    _synchronize(device)
    baseline = _memory(device)
    peak_total = baseline["total_device_used_bytes"]
    torch.cuda.reset_peak_memory_stats(device)
    torch_baseline = torch.cuda.memory_allocated(device)
    try:
        root = Path(__file__).resolve().parents[3]
        path = (
            root
            / "extensions/embodichain_tasks/configs/tasks/manipulation/assembly/peg_insertion/task.study.nominal.yaml"
        )
        raw = load_config(str(path))
        cfg = config_to_cfg(raw, source_path=path)
        cfg.num_envs, cfg.seed = args.num_envs, args.seed
        cfg.sim_cfg.headless, cfg.sim_cfg.device = True, str(device)
        cfg.sim_cfg.render_cfg.renderer = "no-render"
        cfg.sim_cfg.physics_config.solver_cfg["nconmax"] = args.nconmax
        cfg.sim_cfg.physics_config.solver_cfg["njmax"] = args.njmax
        cfg.extensions["assembly_contact_capacity"] = args.assembly_contact_capacity
        cfg.extensions["grasp_contact_capacity"] = args.grasp_contact_capacity
        control_budget = args.steps
        if args.mode == "grip_hold":
            # Only this separate grip diagnostic extends the 5 s task budget:
            # allow the requested approach budget, then a full measured hold.
            control_dt = cfg.sim_steps_per_control * cfg.sim_cfg.physics_dt
            control_budget += math.ceil(args.hold_seconds / control_dt) + 1
            cfg.max_episode_steps = control_budget
            result["diagnostic_episode_duration_override"] = {
                "reason": "separate static-grip hold, not task/policy evaluation",
                "approach_budget_controls": args.steps,
                "max_episode_controls": control_budget,
                "max_episode_seconds": control_budget * control_dt,
            }
        started = time.perf_counter()
        env = make(raw["id"], cfg=cfg)
        env._replay_no_auto_reset = True
        obs, _ = env.reset(seed=args.seed)
        _synchronize(device)
        result["setup_including_generation_ms"] = (time.perf_counter() - started) * 1000
        result.update(
            task_config=str(path),
            fidelity=env.assembly_fidelity,
            physics_dt=env.physics_dt,
            controls_dt=env.step_dt,
            max_episode_steps=env.max_episode_steps,
            asset_sha256=env.assembly_asset.content_hash,
            asset_generator_revision=env.assembly_asset.generator_revision,
            asset_converter_version=env.assembly_asset.converter_version,
            initial_part_pose=env.sim.get_rigid_object("peg")
            .get_local_pose(to_matrix=True)
            .tolist(),
            initial_fixture_pose=env.sim.get_rigid_object("plate")
            .get_local_pose(to_matrix=True)
            .tolist(),
        )
        setup_memory = _memory(device)
        peak_total = max(peak_total, setup_memory["total_device_used_bytes"])
        result["memory_after_setup"] = setup_memory
        health = _PhysicsHealth(env)
        oracle = _Oracle(env) if args.mode == "oracle" else None
        hold = _GripHold(env, args.hold_seconds) if args.mode == "grip_hold" else None
        initial_tcp = (
            env._assembly_robot_base
            @ env._assembly_chain.forward_kinematics(
                env.robot.get_qpos(name="arm")
            ).get_matrix()
        )
        initial_inhand_z = torch.linalg.solve(
            initial_tcp, env.sim.get_rigid_object("peg").get_local_pose(to_matrix=True)
        )[:, 2, 3].clone()
        generator = torch.Generator(device=device).manual_seed(args.seed + 100000)
        done = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
        success = torch.zeros_like(done)
        captured = torch.zeros_like(done)
        rewards = torch.zeros(args.num_envs, device=device)
        terminal_step = torch.full(
            (args.num_envs,), -1, device=device, dtype=torch.long
        )
        samples: list[float] = []
        active_episode_controls = 0
        started = time.perf_counter()
        for index in range(control_budget):
            _synchronize(device)
            start = time.perf_counter()
            with torch.no_grad():
                if oracle is not None:
                    action = oracle.action()
                elif hold is not None:
                    action = hold.action()
                elif args.mode == "random":
                    action = (
                        torch.randn(
                            (args.num_envs, 7), device=device, generator=generator
                        )
                        * 0.5
                    )
                else:
                    action = torch.zeros((args.num_envs, 7), device=device)
                if hold is None:
                    action[done] = 0
                obs, reward, terminated, truncated, info = env.step(action)
            _synchronize(device)
            samples.append((time.perf_counter() - start) * 1000)
            active = ~done
            active_episode_controls += int(active.sum())
            success |= active & info["success"]
            captured |= active & info["metrics"]["captured"]
            rewards += reward.reshape(-1) * active
            newly_done = active & (terminated | truncated)
            terminal_step[newly_done] = index + 1
            done |= newly_done
            if hold is not None:
                hold.observe(info, index + 1)
                result["grip_hold"] = hold.summary()
            finite = all(
                bool(torch.isfinite(value).all())
                for value in (
                    obs["proprioception"],
                    obs["privileged"],
                    env.robot.get_qpos(),
                    env.robot.get_qvel(),
                )
            )
            result["finite"] &= finite
            memory = _memory(device)
            peak_total = max(peak_total, memory["total_device_used_bytes"])
            if (
                index % args.trace_every == 0
                or index == control_budget - 1
                or (hold is None and bool(done.all()))
                or (hold is not None and bool(hold.finished().all()))
            ):
                row = {
                    "step": index + 1,
                    "reward": rewards.tolist(),
                    "success": success.tolist(),
                    "done": done.tolist(),
                    "metrics": {
                        name: value.tolist() for name, value in info["metrics"].items()
                    },
                }
                if oracle is not None:
                    row["oracle_phase"] = oracle.phase.tolist()
                row["grip"] = _grip_snapshot(env, initial_inhand_z)
                if hold is not None:
                    row["hold_seconds_observed"] = hold.elapsed_hold.tolist()
                result["rows"].append(row)
                if hold is None:
                    print(
                        f"step={index+1} success={success.sum().item()}/{args.num_envs} captured={captured.sum().item()}/{args.num_envs}",
                        flush=True,
                    )
                else:
                    print(
                        f"step={index+1} grip_retention_for_requested_hold={result['grip_hold']['retention_fraction']:.3f} (not task success)",
                        flush=True,
                    )
            result.update(
                completed_controls=index + 1,
                completed_arena_controls=(index + 1) * args.num_envs,
                active_first_episode_controls=active_episode_controls,
                success_rate=float(success.float().mean()) if hold is None else None,
                capture_rate=float(captured.float().mean()) if hold is None else None,
                first_episode_success=success.tolist(),
                terminal_step=terminal_step.tolist(),
                first_episode_reward=rewards.tolist(),
            )
            if not finite:
                raise RuntimeError("Nonfinite observations or robot state.")
            if health.values["capacity_overflow_seen"]:
                raise RuntimeError("Native solver contact/constraint overflow.")
            if (hold is None and bool(done.all())) or (
                hold is not None and bool(hold.finished().all())
            ):
                break
        _synchronize(device)
        duration = time.perf_counter() - started
        steady = samples[min(args.warmup_controls, len(samples) - 1) :]
        result.update(
            status="completed",
            elapsed_control_diagnostic_s=duration,
            completed_arena_controls_per_second=result["completed_arena_controls"]
            / duration,
            cost_time_ms=statistics.median(steady),
            samples_ms=samples,
            timed_samples=len(steady),
        )
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        if health is not None:
            result["solver_health"] = health.values
            health.restore(env)
        if env is not None:
            result["sensor_dropped_contacts"] = {
                name: int(env.get_sensor(name).dropped_contacts)
                for name in ("assembly_contacts", "grasp_contacts")
            }
        _synchronize(device)
        final = _memory(device)
        peak_total = max(peak_total, final["total_device_used_bytes"])
        result["memory"] = {
            "baseline": baseline,
            "final_before_teardown": final,
            "sampled_peak_total_device_bytes": peak_total,
            "sampled_peak_total_device_delta_bytes": peak_total
            - baseline["total_device_used_bytes"],
            "torch_baseline_allocated_bytes": torch_baseline,
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "torch_peak_allocated_delta_bytes": torch.cuda.max_memory_allocated(device)
            - torch_baseline,
            "torch_final_allocated_delta_bytes": torch.cuda.memory_allocated(device)
            - torch_baseline,
            "scope": "cold scene capacity diagnostic; total-device usage sampled before/after setup and each control (not transient peak); includes other processes; Torch allocator peak excludes native/Warp allocations",
        }
        if env is not None:
            env.close(exit_process=False)
        SimulationManager.flush_cleanup_queue()
    return result


def _write_report(output: Path, document: dict[str, Any]) -> None:
    """Write raw data and exactly three Markdown metric tables."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(document, indent=2) + "\n")
    result = document["result"]
    memory = result.get("memory", {})
    baseline = memory.get("baseline", {})
    final = memory.get("final_before_teardown", {})
    mib = 1024**2

    def megabytes(value: int | None) -> str:
        return "unmeasured" if value is None else f"{value / mib:.3f}"

    cpu_delta = None
    if (
        baseline.get("process_rss_bytes") is not None
        and final.get("process_rss_bytes") is not None
    ):
        cpu_delta = final["process_rss_bytes"] - baseline["process_rss_bytes"]
    total_delta = None
    if (
        baseline.get("total_device_used_bytes") is not None
        and final.get("total_device_used_bytes") is not None
    ):
        total_delta = (
            final["total_device_used_bytes"] - baseline["total_device_used_bytes"]
        )
    mode = document["arguments"]["mode"]
    success = result.get("success_rate")
    success = "not evaluated" if success is None else success
    health = result.get("solver_health", {})
    overflow = (
        health.get("capacity_overflow_seen")
        if health.get("overflow_available")
        else "unmeasured"
    )
    lines = [
        "# Native peg reachability diagnostic",
        "",
        document["qualification"],
        "",
        "## Time & Memory",
        "",
        "| mode | cost_time_ms | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb | completed_arena_controls_per_second |",
        "|---|---:|---:|---:|---:|---:|",
        f"| {mode} | {result.get('cost_time_ms', 'unmeasured')} | {megabytes(cpu_delta)} | {megabytes(total_delta)} | {megabytes(memory.get('sampled_peak_total_device_bytes'))} | {result.get('completed_arena_controls_per_second', 'unmeasured')} |",
        "",
        "## Success & Other Metrics",
        "",
        "| mode | status | success_rate | capture_rate | grip_retention_fraction | finite | native_capacity_loss | convergence_budget_exhausted |",
        "|---|---|---:|---:|---:|---|---|---|",
        f"| {mode} | {result['status']} | {success} | {result.get('capture_rate', 'not evaluated')} | {result.get('grip_hold', {}).get('retention_fraction', 'not evaluated')} | {result.get('finite', 'unmeasured')} | {overflow} | {health.get('convergence_budget_exhausted_seen', 'unmeasured')} |",
        "",
        "## Leaderboard",
        "",
        "| rank | algorithm | overall_success_rate | qualification |",
        "|---:|---|---:|---|",
        f"| 1 | {mode} | {success} | diagnostic first episodes; not RL |",
        "",
        "Timing includes controller, completed native stepping and synchronous private health reads. The median excludes the first requested warmup controls; throughput includes trace and diagnostic collection. Memory is sampled cold-scene total-device usage, not a separate warmed allocator benchmark. Other processes can change device usage.",
        "",
        "The oracle reads part/fixture poses and payload mass/COM. It stages lift, XY/orientation alignment, insertion and hold, with DLS and payload-support bias through joint-position targets. It does not receive a policy score. Zero and random use the unchanged task interface. No pose/state writes or further resets occur after initial reset. Nominal geometry, reset support and episode limit are unchanged. Three-frame hold, bilateral grasp and penetration guards remain project additions to public geometric success.",
        "",
        "Raw JSON contains exact arguments, software/hardware, source identities, all initial poses, per-arena outcomes and sampled traces. Native health remains unverified if private overflow buffers are unavailable.",
        "Grip-hold mode alone extends the episode budget to allow approach followed by six seconds of hold. It solves one 5 mm lift with encoder FK/DLS and holds the resulting immutable joint target, with payload support based solely on the initial grasp. Actual part poses feed diagnostic slip measurements only. Task failure guards are still evaluated and latched; fixed hold commands continue for inspection. Its reported retention is a grip diagnostic, never a task success score.",
        "",
    ]
    output.with_suffix(".md").write_text("\n".join(lines))
    print(f"Markdown report saved: {output.with_suffix('.md')}", flush=True)


def main() -> None:
    """Select a bounded diagnostic without changing nominal randomization."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mode", choices=("zero", "random", "oracle", "grip_hold"), default="oracle"
    )
    parser.add_argument("--hold-seconds", type=float, default=6.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--nconmax", type=int, default=256)
    parser.add_argument("--njmax", type=int, default=1024)
    parser.add_argument("--assembly-contact-capacity", type=int, default=256)
    parser.add_argument("--grasp-contact-capacity", type=int, default=64)
    parser.add_argument("--trace-every", type=int, default=10)
    parser.add_argument("--warmup-controls", type=int, default=10)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/benchmarks/peg_reachability")
    )
    args = parser.parse_args()
    for name in (
        "num_envs",
        "steps",
        "threads",
        "nconmax",
        "njmax",
        "assembly_contact_capacity",
        "grasp_contact_capacity",
        "trace_every",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.warmup_controls < 0:
        parser.error("--warmup-controls must be nonnegative.")
    if not math.isfinite(args.hold_seconds) or args.hold_seconds <= 0:
        parser.error("--hold-seconds must be finite and positive.")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    document: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": (
            "Static grip diagnostic with one encoder-FK lift and fixed targets; no task success score."
            if args.mode == "grip_hold"
            else "Privileged action-only reachability/control diagnostic; not learned-policy evaluation or paper reproduction."
        ),
        "arguments": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(args).items()
        },
        "software": _versions(),
        "torch_cuda_version": torch.version.cuda,
        "platform": platform.platform(),
        "dtype": "float32",
        "torch_threads": torch.get_num_threads(),
        "synchronization": "Warp owning device and Torch CUDA before/after completed control; native health readback each substep",
    }
    if device.type != "cuda" or not torch.cuda.is_available():
        document["result"] = {
            "status": "skipped",
            "reason": "Native Newton/Warp diagnostic requires CUDA.",
        }
    else:
        document["gpu"] = torch.cuda.get_device_name(device)
        try:
            document["nvidia_driver_inventory"] = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=uuid,name,driver_version,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            document["nvidia_driver_inventory"] = None
        try:
            document["result"] = _run(args)
        except Exception:
            document["result"] = {"status": "failed", "error": traceback.format_exc()}
    _write_report(args.output, document)
    if document["result"]["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
