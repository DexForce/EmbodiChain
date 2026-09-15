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

"""Contact-aware, full-state validation for one fixed-scene cuboid PickUp."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from embodichain.utils import configclass
from embodichain.lab.sim.motion.expansion import (
    CandidateTrajectoryBatch,
    ExpertEpisode,
    MotionSnapshot,
    ValidationCheck,
    ValidationResult,
)
from ._collision import _FullStateCollisionWorld

if TYPE_CHECKING:
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.sim.objects import Robot

__all__ = ["PickUpContactProfile", "PickUpMotionValidator"]


def _result(name, passed, detail="", **metrics):
    return ValidationResult(
        (ValidationCheck(name, "passed" if passed else "failed", detail, metrics),)
    )


@configclass
class PickUpContactProfile:
    """Explicit permissions and measured-grasp tolerances for a single PickUp.

    Finger/object contacts are allowed in the declared final approach region
    and from ``close`` onwards. Object/support contact is allowed through
    initial lift-off; robot/support contact
    is restricted to the declared fixed mounting links. Self pairs within two
    URDF kinematic hops follow the existing cuRobo structural exclusion policy.
    All other robot/world, object/world and nonexcluded self collisions reject.
    """

    object_id: str = "cube"
    support_id: str = "bench"
    finger_links: tuple[str, ...] = ("gripper_finger1_link_1", "gripper_finger2_link_1")
    mounting_links: tuple[str, ...] = ("arm_base_link",)
    tcp_link: str = "ee_link"
    tcp_offset: tuple[tuple[float, ...], ...] = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.15),
        (0.0, 0.0, 0.0, 1.0),
    )
    min_lift: float = 0.12
    max_relative_translation: float = 0.01
    max_relative_rotation: float = 0.15
    max_tcp_distance: float = 0.06
    approach_contact_distance: float = 0.025
    max_penetration: float = 0.002
    min_contact_fraction: float = 0.95
    min_hold_seconds: float = 1.0
    measured_joint_tolerance: float = 1e-4
    max_joint_step: float = 0.02
    max_validation_samples: int = 4096
    max_contacts_per_step: int = 4096

    def __post_init__(self) -> None:
        for name in ("object_id", "support_id", "tcp_link"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a nonempty name")
        if self.object_id == self.support_id:
            raise ValueError("PickUp target and support must differ")
        for name in ("finger_links", "mounting_links"):
            values = getattr(self, name)
            if (
                isinstance(values, str)
                or any(not isinstance(v, str) or not v for v in values)
                or len(set(values)) != len(values)
            ):
                raise ValueError(f"{name} must contain unique names")
            setattr(self, name, tuple(values))
        if len(self.finger_links) != 2:
            raise ValueError(
                "This PickUp profile requires two distinct gripping fingers"
            )
        from embodichain.lab.sim.motion.expansion.contracts import _pose

        _pose(torch.tensor(self.tcp_offset, dtype=torch.float64), "tcp_offset")
        for name in (
            "min_lift",
            "max_relative_translation",
            "max_relative_rotation",
            "max_tcp_distance",
            "approach_contact_distance",
            "max_penetration",
            "min_contact_fraction",
            "min_hold_seconds",
            "measured_joint_tolerance",
            "max_joint_step",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.min_contact_fraction > 1:
            raise ValueError("min_contact_fraction must be <= 1")
        for name in ("max_validation_samples", "max_contacts_per_step"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 2:
                raise ValueError(f"{name} must be an integer >= 2")


@dataclass
class _Evidence:
    steps: int = 0
    hold_steps: int = 0
    contact_steps: list[int] = field(default_factory=lambda: [0, 0])
    first_relative: np.ndarray | None = None
    minimum_lift: float = math.inf
    max_translation: float = 0.0
    max_rotation: float = 0.0
    max_distance: float = 0.0
    failure: str | None = None


class PickUpMotionValidator:
    """Validate compiled PickUp plans and actual CPU-physics contact evidence.

    This bounded integration accepts an unscaled, fixed-base URDF robot and
    cuboid rigid objects. Planned checks use conservative convex hulls of URDF
    *collision* shapes, full joint FK including mimic geometry, a TCP-relative
    lifted object, and densified joint samples. Actual checks use measured full
    joints and object poses; native contacts are also checked after every physics
    substep. No continuous-collision guarantee is claimed.

    The same instance must be attached to QposRolloutExecutor and GenerationRunner.
    Only the target object and physically observed mimic joints may move outside
    the active controller layout. Gym is deliberately unsupported until its
    execution loop exposes equivalent physical-substep evidence.

    Args:
        sim: Manually stepped CPU-physics simulation owning every row.
        robot: The sole fixed-base robot, with initialized full-articulation FK.
        profile: Explicit collision permissions and measured task thresholds.
    """

    def __init__(
        self,
        sim: SimulationManager,
        robot: Robot,
        *,
        profile: PickUpContactProfile | None = None,
    ) -> None:
        if torch.device(sim.sim_config.sim_device).type != "cpu":
            raise ValueError("PickUp contact evidence currently requires CPU physics")
        self.sim, self.robot = sim, robot
        self.profile = (profile or PickUpContactProfile()).copy()
        self.profile.__post_init__()
        self.world = _FullStateCollisionWorld(sim, robot)
        p = self.profile
        if {p.object_id, p.support_id} - set(self.world.entity_ids):
            raise ValueError(
                "PickUp object and support must be registered collision entities"
            )
        if (
            set((*p.finger_links, *p.mounting_links)) - set(self.world.links)
            or p.tcp_link not in robot.link_names
        ):
            raise ValueError(
                "PickUp link permissions must resolve to actual collision links"
            )
        self.tcp_offset = torch.tensor(
            p.tcp_offset, dtype=torch.float32, device=robot.device
        )
        if set(p.mounting_links) - self.world.fixed_links:
            raise ValueError(
                "Only links fixed to the robot root may contact its mounting support"
            )
        self.mimic_ids = tuple(robot.mimic_ids)
        self.dynamic_entity_ids = (p.object_id,)
        import dexsim

        self._physics = dexsim.default_world().get_physics_scene()
        self._physics.enable_contact_data_update_on_cpu(True)
        self._users = {}
        for link in robot.link_names:
            self._register_users(link, robot.get_user_ids(link))
        for uid in self.world.entity_ids:
            self._register_users(uid, sim.get_rigid_object(uid).get_user_ids())
        self._rows = ()
        self._snapshots = ()
        self._evidence = []

    @property
    def collision_world_entity_ids(self) -> tuple[str, ...]:
        """Physical rigid UIDs represented by this complete collision model."""
        return self.world.entity_ids

    def _register_users(self, name, values):
        ids = values.detach().cpu().reshape(-1).tolist()
        if len(ids) != self.robot.num_instances:
            raise ValueError("Contact user IDs must map every physical row")
        for row, uid in enumerate(ids):
            if uid in self._users and self._users[uid] != (row, name):
                raise ValueError("Ambiguous native contact body ID")
            self._users[uid] = (row, name)

    def _phases(self, phases, length):
        expected = ("transit", "approach", "close", "lift", "hold")
        if tuple(phase.phase_id for phase in phases) != expected:
            raise ValueError(
                "PickUp requires explicit transit/approach/close/lift/hold phases"
            )
        end = 0
        kinds = ("free", "contact", "contact", "contact", "hold")
        for phase, kind in zip(phases, kinds):
            if phase.start_index != end or phase.kind != kind:
                raise ValueError("PickUp phases must cover the trajectory in order")
            if phase.phase_id != "transit" and phase.allowed_operators:
                raise ValueError("PickUp contact and approach phases must be protected")
            end = phase.stop_index
        if end != length:
            raise ValueError("PickUp phases must cover every trajectory sample")
        return phases

    def _allowed(self, first, second, phase, lift, grasp_distance=math.inf):
        p = self.profile
        pair = frozenset((first, second))
        if pair in self.world.adjacent_pairs:
            return True
        if p.support_id in pair and any(link in pair for link in p.mounting_links):
            return True
        if p.object_id in pair:
            other = second if first == p.object_id else first
            if other in p.finger_links and (
                phase in ("close", "lift", "hold")
                or (
                    phase == "approach"
                    and grasp_distance <= p.approach_contact_distance
                )
            ):
                return True
            if other == p.support_id and (
                phase in ("transit", "approach", "close")
                or (phase == "lift" and lift <= 0.005)
            ):
                return True
        return False

    def _validate_path(self, qpos, phases, snapshot, row, *, measured_objects=None):
        p = self.profile
        qpos = qpos.detach().cpu()
        self._phases(phases, len(qpos))
        if not torch.allclose(
            qpos[0], snapshot.joint_positions.cpu().to(qpos), atol=1e-6, rtol=0
        ):
            raise ValueError("PickUp path must begin at the captured initial joints")
        limits = self.robot.get_qpos_limits()[row].detach().cpu().to(qpos)
        tolerance = 0.0 if measured_objects is None else p.measured_joint_tolerance
        if bool(
            (
                (qpos < limits[:, 0] - tolerance) | (qpos > limits[:, 1] + tolerance)
            ).any()
        ):
            return _result(
                "path_collision",
                False,
                "Full-joint position limits exceeded",
                maximum_joint_limit_violation=float(
                    torch.maximum(limits[:, 0] - qpos, qpos - limits[:, 1])
                    .clamp(min=0)
                    .max()
                ),
            )
        if set(snapshot.entity_poses) != set(self.world.entity_ids):
            raise ValueError("PickUp snapshots must cover the complete collision world")
        if measured_objects is None:
            # Planned geometry must model passive mimic motion, even though the
            # drive only receives targets for active joints.
            for child, parent, multiplier, offset in zip(
                self.robot.mimic_ids,
                self.robot.mimic_parents,
                self.robot.mimic_multipliers,
                self.robot.mimic_offsets,
            ):
                if not torch.allclose(
                    qpos[:, child],
                    qpos[:, parent] * multiplier + offset,
                    atol=1e-6,
                    rtol=0,
                ):
                    raise ValueError(
                        "Planned mimic coordinates disagree with the robot coupling"
                    )
            increments = qpos.diff(dim=0).abs().amax(dim=1)
            counts = torch.clamp(
                torch.ceil(increments / p.max_joint_step).long(), min=1
            )
            if int(counts.sum()) + 1 > p.max_validation_samples:
                raise ValueError("PickUp collision sampling exceeds the bounded budget")
            samples, source_indices = [qpos[:1]], [0]
            for index, count in enumerate(counts.tolist(), 1):
                alpha = torch.arange(1, count + 1, dtype=qpos.dtype)[:, None] / count
                samples.append(torch.lerp(qpos[index - 1], qpos[index], alpha))
                source_indices.extend([index] * count)
            values = torch.cat(samples)
        else:
            values = qpos
            source_indices = list(range(len(qpos)))
            if len(values) > p.max_validation_samples or measured_objects.shape != (
                len(values),
                4,
                4,
            ):
                raise ValueError(
                    "Actual collision evidence has incompatible shape or exceeds its budget"
                )
        poses = self.world.link_poses(values, snapshot.root_pose, (p.tcp_link,))
        tcp = poses[p.tcp_link] @ self.tcp_offset.cpu().double().numpy()
        lift_start = phases[3].start_index
        anchor = next(
            i for i, index in enumerate(source_indices) if index >= lift_start
        )
        initial_object = snapshot.entity_poses[p.object_id].cpu().double().numpy()
        relative = np.linalg.inv(tcp[anchor]) @ initial_object
        for sample, index in enumerate(source_indices):
            phase = next(
                phase.phase_id
                for phase in phases
                if phase.start_index <= index < phase.stop_index
            )
            objects = {
                uid: pose.cpu().double().numpy()
                for uid, pose in snapshot.entity_poses.items()
            }
            if measured_objects is not None:
                objects[p.object_id] = measured_objects[sample].cpu().double().numpy()
            elif index >= lift_start:
                objects[p.object_id] = tcp[sample] @ relative
            height = objects[p.object_id][2, 3] - initial_object[2, 3]
            collision = self.world.collisions(
                {link: poses[link][sample] for link in self.world.links},
                objects,
                p.object_id,
                lambda a, b: self._allowed(
                    a,
                    b,
                    phase,
                    height,
                    float(
                        np.linalg.norm(tcp[sample, :3, 3] - objects[p.object_id][:3, 3])
                    ),
                ),
            )
            if collision:
                return _result(
                    "path_collision",
                    False,
                    f"Forbidden collision at sample {index} ({phase}): {collision}",
                    checked_samples=sample + 1,
                )
        return _result(
            "path_collision",
            True,
            "Conservative full-joint URDF hulls and object/world samples",
            checked_samples=len(values),
        )

    def validate_qpos(
        self, batch: CandidateTrajectoryBatch, snapshots: Sequence[MotionSnapshot]
    ) -> tuple[ValidationResult, ...]:
        """Check planned full-state paths, phase permissions and lifted geometry.

        Args:
            batch: Full-joint candidates with explicit physical source rows.
            snapshots: Initial states in physical environment order.

        Returns:
            One path validation result per candidate, in candidate order.
        """
        if (
            len(snapshots) != self.robot.num_instances
            or batch.source_row_indices is None
            or batch.joint_names != tuple(self.robot.joint_names)
        ):
            raise ValueError(
                "PickUp validation requires full ordered joints and real source rows"
            )
        results = []
        for index, row in enumerate(batch.source_row_indices.tolist()):
            if row >= len(snapshots):
                raise ValueError("Source row is outside the physical batch")
            snapshot = snapshots[row]
            if snapshot.joint_names != batch.joint_names:
                raise ValueError(
                    "PickUp snapshots must use the full ordered robot joints"
                )
            identity = batch.identities[index]
            if (identity.scene_case_id, identity.initial_state_id) != (
                snapshot.scene_case.scene_case_id,
                snapshot.scene_case.initial_state_id,
            ):
                raise ValueError(
                    "PickUp candidate identity does not match its snapshot"
                )
            length = int(batch.valid_length[index])
            results.append(
                self._validate_path(
                    batch.positions[index, :length], batch.phases[index], snapshot, row
                )
            )
        return tuple(results)

    def validate_episode(
        self, episode: ExpertEpisode, snapshot: MotionSnapshot, *, row: int
    ) -> ValidationResult:
        """Recheck measured finger geometry and measured target poses after rollout.

        Args:
            episode: Frozen observations containing full joints and object poses.
            snapshot: Initial scene state belonging to the episode's case.
            row: Physical environment row used during execution.

        Returns:
            The measured path's joint-limit and collision validation result.
        """
        return self._validate_path(
            episode.observations["joint_positions"],
            episode.phases,
            snapshot,
            row,
            measured_objects=episode.observations["object_pose"],
        )

    def observations(self) -> dict[str, torch.Tensor]:
        """Return current physical object and TCP poses in each local arena.

        Returns:
            Object and TCP transforms, each shaped ``(num_envs, 4, 4)``.
        """
        return {
            "object_pose": self.sim.get_rigid_object(
                self.profile.object_id
            ).get_local_pose(to_matrix=True),
            "tcp_pose": self.robot.get_link_pose(self.profile.tcp_link, to_matrix=True)
            @ self.tcp_offset,
        }

    def begin_rollout(
        self,
        candidates: Sequence[CandidateTrajectoryBatch | None],
        snapshots: Sequence[MotionSnapshot],
    ) -> None:
        """Reset bounded per-row evidence before the first physical command.

        Args:
            candidates: One candidate per physical row, or ``None`` for idle rows.
            snapshots: Initial states in the same physical row order.
        """
        if len(candidates) != self.robot.num_instances or len(snapshots) != len(
            candidates
        ):
            raise ValueError(
                "Contact monitoring must cover the complete physical batch"
            )
        for candidate in candidates:
            if candidate is not None:
                self._phases(candidate.phases[0], int(candidate.valid_length[0]))
        self._rows, self._snapshots = tuple(candidates), tuple(snapshots)
        self._evidence = [_Evidence() for _ in candidates]

    def observe_substep(
        self, sample_index: int, active: Sequence[bool], *, physics_dt: float
    ) -> None:
        """Accumulate native contacts and held-object stability after one physics step.

        The CPU contact buffer is read before the next physics update. Unknown
        bodies, cross-row contacts, excessive penetration and buffer budget
        overflow cannot produce accepted evidence. No contact-history arrays grow.

        Args:
            sample_index: Command sample whose physics substep just completed.
            active: Whether each physical row is still executing its candidate.
            physics_dt: Duration of the completed physics substep in seconds.
        """
        p = self.profile
        data, users = self._physics.get_cpu_contact_buffer()
        data, users = np.asarray(data), np.asarray(users)
        if (
            data.ndim != 2
            or data.shape[1] != 11
            or users.shape != (len(data), 2)
            or not np.isfinite(data).all()
            or len(data) > p.max_contacts_per_step
        ):
            raise ValueError(
                "Native contact evidence is malformed or exceeds its bounded budget"
            )
        observation = {
            key: value.detach().cpu().double().numpy()
            for key, value in self.observations().items()
        }
        finger_contacts = [set() for _ in active]
        phases = []
        for row, candidate in enumerate(self._rows):
            phases.append(
                None
                if candidate is None or not active[row]
                else next(
                    phase.phase_id
                    for phase in candidate.phases[0]
                    if phase.start_index <= sample_index < phase.stop_index
                )
            )
        for contact, pair in zip(data, users):
            first, second = self._users.get(int(pair[0])), self._users.get(int(pair[1]))
            if first is None and second is None:
                continue
            known = first or second
            row = known[0]
            if first is not None and second is not None and first[0] != second[0]:
                for owner in (first[0], second[0]):
                    if active[owner]:
                        self._evidence[owner].failure = (
                            self._evidence[owner].failure or "Cross-row native contact"
                        )
                continue
            if not active[row]:
                continue
            evidence = self._evidence[row]
            if first is None or second is None:
                evidence.failure = (
                    evidence.failure or "Unmapped body or cross-row native contact"
                )
                continue
            a, b = first[1], second[1]
            height = observation["object_pose"][row, 2, 3] - float(
                self._snapshots[row].entity_poses[p.object_id][2, 3]
            )
            distance = float(
                np.linalg.norm(
                    observation["tcp_pose"][row, :3, 3]
                    - observation["object_pose"][row, :3, 3]
                )
            )
            if not self._allowed(a, b, phases[row], height, distance):
                evidence.failure = (
                    evidence.failure
                    or f"Forbidden physical contact during {phases[row]}: {a} / {b}"
                )
            if contact[10] < -p.max_penetration:
                evidence.failure = (
                    evidence.failure
                    or f"Physical penetration exceeds tolerance: {a} / {b}"
                )
            if contact[9] > 1e-7 and p.object_id in (a, b):
                other = b if a == p.object_id else a
                if other in p.finger_links:
                    finger_contacts[row].add(other)
        for row, phase in enumerate(phases):
            if phase is None:
                continue
            evidence = self._evidence[row]
            evidence.steps += 1
            if phase not in ("lift", "hold"):
                continue
            if phase == "hold":
                evidence.hold_steps += 1
                for index, link in enumerate(p.finger_links):
                    evidence.contact_steps[index] += link in finger_contacts[row]
            obj, tcp = observation["object_pose"][row], observation["tcp_pose"][row]
            relative = np.linalg.inv(tcp) @ obj
            if evidence.first_relative is None:
                evidence.first_relative = relative.copy()
            delta = np.linalg.inv(evidence.first_relative) @ relative
            if phase == "hold":
                evidence.minimum_lift = min(
                    evidence.minimum_lift,
                    obj[2, 3]
                    - float(self._snapshots[row].entity_poses[p.object_id][2, 3]),
                )
            evidence.max_translation = max(
                evidence.max_translation, float(np.linalg.norm(delta[:3, 3]))
            )
            angle = math.acos(float(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1)))
            evidence.max_rotation = max(evidence.max_rotation, angle)
            evidence.max_distance = max(
                evidence.max_distance, float(np.linalg.norm(relative[:3, 3]))
            )
        self._physics_dt = physics_dt

    def rollout_validation(self, row: int) -> ValidationResult:
        """Freeze contact coverage and stable-grasp gates for one completed row.

        Args:
            row: Physical environment row monitored by this validator.

        Returns:
            Mandatory native-contact and held-object stability checks.
        """
        e, p = self._evidence[row], self.profile
        duration = e.hold_steps * getattr(self, "_physics_dt", 0.0)
        fractions = [value / max(1, e.hold_steps) for value in e.contact_steps]
        lift = float(e.minimum_lift) if math.isfinite(e.minimum_lift) else 0.0
        held = (
            duration >= p.min_hold_seconds
            and lift >= p.min_lift
            and e.max_translation <= p.max_relative_translation
            and e.max_rotation <= p.max_relative_rotation
            and e.max_distance <= p.max_tcp_distance
            and min(fractions) >= p.min_contact_fraction
        )
        return ValidationResult(
            (
                ValidationCheck(
                    "physical_contacts",
                    "passed" if e.steps and e.failure is None else "failed",
                    e.failure
                    or "Native contacts checked after every CPU physics substep",
                    {"physics_samples": e.steps},
                ),
                ValidationCheck(
                    "held_object_stability",
                    "passed" if held else "failed",
                    "Both fingers must maintain physical contact with a lifted, stable object",
                    {
                        "hold_seconds": duration,
                        "minimum_lift_m": lift,
                        "max_relative_translation_m": e.max_translation,
                        "max_relative_rotation_rad": e.max_rotation,
                        "max_tcp_distance_m": e.max_distance,
                        "finger_0_contact_fraction": fractions[0],
                        "finger_1_contact_fraction": fractions[1],
                    },
                ),
            )
        )
