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

"""Immutable physical state services, independent of Gym and Task Program."""

from __future__ import annotations
from dataclasses import dataclass, field
import math
from numbers import Real
import torch
from .query import SceneView, _backend_tensor

__all__ = [
    "EntityState",
    "SceneState",
    "capture_state",
    "restore_state",
    "restore_render_state",
]


@dataclass(frozen=True)
class EntityState:
    """Immutable single-arena rigid or articulated physical state in Z-up/xyzw."""

    kind: str
    uid: str
    body_type: str
    joint_names: tuple[str, ...]
    pose: tuple[float, ...]
    velocity: tuple[float, ...]
    qpos: tuple[float, ...] = ()
    qvel: tuple[float, ...] = ()
    target_qpos: tuple[float, ...] = ()
    target_qvel: tuple[float, ...] = ()
    qf: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in ("rigid_object", "robot", "articulation"):
            raise ValueError("Unknown snapshot physical entity kind")
        if type(self.uid) is not str or not self.uid or self.uid != self.uid.strip():
            raise ValueError("Snapshot entity UID must be a nonempty stripped string")
        names = tuple(self.joint_names)
        if any(
            type(name) is not str or not name or name != name.strip() for name in names
        ) or len(set(names)) != len(names):
            raise ValueError("Snapshot joint_names must contain unique nonempty names")
        if self.kind == "rigid_object":
            if names or self.body_type not in ("static", "dynamic", "kinematic"):
                raise ValueError("Rigid snapshot body type/joint names are invalid")
        elif self.body_type:
            raise ValueError("Articulation snapshots cannot declare rigid body_type")
        object.__setattr__(self, "joint_names", names)
        for name, width in (
            ("pose", 7),
            ("velocity", 6),
            *(
                (name, len(names))
                for name in ("qpos", "qvel", "target_qpos", "target_qvel", "qf")
            ),
        ):
            values = tuple(getattr(self, name))
            if len(values) != width or any(
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                for value in values
            ):
                raise ValueError(
                    f"Snapshot {name} must contain {width} finite real values"
                )
            object.__setattr__(self, name, tuple(map(float, values)))
        if not math.isclose(
            sum(value * value for value in self.pose[3:]), 1.0, abs_tol=1e-5
        ):
            raise ValueError("Snapshot pose quaternion must be normalized")
        if (
            self.kind == "rigid_object"
            and self.body_type != "dynamic"
            and any(self.velocity)
        ):
            raise ValueError("Moving non-dynamic rigid bodies cannot be restored")


@dataclass(frozen=True)
class SceneState:
    """Fixed-topology physical state with an explicit environment dimension.

    Args:
        physics_backend: Backend identity, required for restore.
        env_ids: Arena selection corresponding to each batch.
        batches: Immutable numeric entity rows, one tuple per selected arena.

    This is a portable physical-state representation, not a backend checkpoint.
    The owner and binding tokens restrict restoration to the original simulation.
    """

    physics_backend: str
    env_ids: tuple[int, ...]
    batches: tuple[tuple[EntityState, ...], ...]
    _owner: object = field(repr=False, compare=False)
    _topology: tuple[tuple[object, ...], ...] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.physics_backend, str)
            or not self.physics_backend.strip()
        ):
            raise ValueError("SceneState physics_backend must be nonempty text")
        batches = tuple(tuple(batch) for batch in self.batches)
        if (
            not self.env_ids
            or any(type(i) is not int or i < 0 for i in self.env_ids)
            or len(set(self.env_ids)) != len(self.env_ids)
            or len(batches) != len(self.env_ids)
        ):
            raise ValueError(
                "SceneState requires one batch per unique nonnegative env_id"
            )
        for batch in batches:
            if any(not isinstance(row, EntityState) for row in batch) or len(
                {row.uid for row in batch}
            ) != len(batch):
                raise ValueError("SceneState requires unique validated entity rows")
        object.__setattr__(self, "env_ids", tuple(self.env_ids))
        object.__setattr__(self, "batches", batches)


@dataclass(frozen=True)
class _CapturedState:
    """Owned CPU tensor fields from one physical boundary in view row order."""

    view: SceneView
    entities: tuple[tuple[str, str, str, tuple[str, ...]], ...]
    fields: dict[tuple[str, str], torch.Tensor]
    rigid: dict[str, torch.Tensor]

    @property
    def env_ids(self) -> tuple[int, ...]:
        return self.view.env_ids

    def to_scene_state(self) -> SceneState:
        values = {}
        for (uid, name), tensor in self.fields.items():
            values.setdefault(uid, {})[name] = tensor.tolist()
        batches = []
        for row in range(len(self.env_ids)):
            batches.append(
                tuple(
                    EntityState(
                        kind,
                        uid,
                        body_type,
                        names,
                        **{name: rows[row] for name, rows in values[uid].items()},
                    )
                    for kind, uid, body_type, names in self.entities
                )
            )
        return SceneState(
            self.view.sim.physics_backend,
            self.env_ids,
            tuple(batches),
            self.view.sim,
            self.view.topology,
        )


def _capture_state_fields(view: SceneView) -> _CapturedState:
    """Read each entity once before materializing portable Python values."""
    view.check_topology()
    fields, rigid, entities = {}, {}, []
    rows = list(view.env_ids)

    def select(values, widths, uid):
        for value, width in zip(values, widths, strict=True):
            if (
                not isinstance(value, torch.Tensor)
                or not value.is_floating_point()
                or value.shape != (view._num_envs, width)
            ):
                raise ValueError(f"{uid} has invalid physical state field shape")
        if len({(value.device, value.dtype) for value in values}) == 1:
            source = values[0] if len(values) == 1 else torch.cat(values, dim=-1)
            packed = source.detach()[rows].cpu()
            if not bool(torch.isfinite(packed).all()):
                raise ValueError(f"{uid} state must be finite in selected rows")
            return packed.split(widths, dim=-1)
        selected = tuple(value.detach()[rows].cpu() for value in values)
        if not all(bool(torch.isfinite(value).all()) for value in selected):
            raise ValueError(f"{uid} state must be finite in selected rows")
        return selected

    for kind, uid, entity in view.entities():
        if kind == "rigid_object":
            (state,) = select((entity.body_state,), (13,), uid)
            pose, velocity = state[:, :7], state[:, 7:]
            if entity.body_type != "dynamic" and bool(velocity.any()):
                raise ValueError("Moving non-dynamic rigid bodies cannot be restored")
            rigid[uid] = state
            values = {"pose": pose, "velocity": velocity}
            entities.append((kind, uid, entity.body_type, ()))
        else:
            names = tuple(entity.joint_names)
            live = entity.body_data.fetch_state()
            # Some backends own and reuse observation buffers behind their getters.
            live = {key: value.clone() for key, value in live.items()}
            values = dict(
                zip(
                    (
                        "pose",
                        "velocity",
                        "qpos",
                        "qvel",
                        "target_qpos",
                        "target_qvel",
                        "qf",
                    ),
                    select(
                        (
                            live["root_pose"],
                            torch.cat((live["root_lin_vel"], live["root_ang_vel"]), -1),
                            live["qpos"],
                            live["qvel"],
                            entity.get_qpos(target=True),
                            entity.get_qvel(target=True),
                            entity.get_qf(),
                        ),
                        (7, 6, *(len(names),) * 5),
                        uid,
                    ),
                    strict=True,
                )
            )
            entities.append((kind, uid, "", names))
        quat_norm2 = values["pose"][:, 3:].double().square().sum(dim=-1)
        if bool(((quat_norm2 - 1.0).abs() > 1e-5).any()):
            raise ValueError("Snapshot pose quaternion must be normalized")
        fields.update({(uid, name): value for name, value in values.items()})
    return _CapturedState(view, tuple(entities), fields, rigid)


def capture_state(view: SceneView) -> SceneState:
    """Capture all physical fields without stepping or retaining live buffers.

    Args:
        view: Fixed-topology scene selection with explicit ordered environment
            indices. Only physical entities are captured; pure visual group
            members retain their independent render state.
    Returns:
        Immutable arena-local poses, velocities, joints, targets and efforts.
    """
    return _capture_state_fields(view).to_scene_state()


def _preflight_restore(
    view: SceneView,
    snapshot: SceneState,
    *,
    qpos_tolerance: float = 0.0,
    validate_joint_limits: bool = True,
) -> dict[str, tuple[object, dict[str, torch.Tensor]]]:
    if (
        isinstance(qpos_tolerance, bool)
        or not isinstance(qpos_tolerance, Real)
        or not math.isfinite(qpos_tolerance)
        or qpos_tolerance < 0
    ):
        raise ValueError("qpos_tolerance must be finite and nonnegative")
    view.check_topology()
    if not isinstance(snapshot, SceneState):
        raise TypeError("snapshot must be a SceneState")
    if snapshot._owner is not view.sim:
        raise ValueError("Snapshot belongs to another scene host")
    if (
        snapshot._topology != view.topology
        or snapshot.physics_backend != view.sim.physics_backend
        or snapshot.env_ids != view.env_ids
    ):
        raise ValueError("Snapshot topology, backend or environment selection differs")
    for batch in snapshot.batches:
        recorded = tuple(
            (row.kind, row.uid, row.body_type, row.joint_names) for row in batch
        )
        if recorded != tuple(
            description[:4]
            for description in view.topology
            if description[0] != "visual_object"
        ):
            raise ValueError(
                "Snapshot entity set, body types or joint names differ from the scene"
            )
    entities = view.entities()
    for kind, uid, entity in entities:
        methods = (
            ("clear_dynamics", "set_local_pose")
            if kind == "rigid_object"
            else ("set_state",)
        )
        if kind == "rigid_object" and entity.body_type == "dynamic":
            methods += ("set_velocity",)
        if any(not callable(getattr(entity, method, None)) for method in methods):
            raise ValueError(f"Entity {uid!r} does not support physical restoration")

    bindings = {uid: entity for _, uid, entity in entities}
    prepared = {}
    for column, state in enumerate(snapshot.batches[0]):
        states = [batch[column] for batch in snapshot.batches]
        values = {
            "pose": [row.pose for row in states],
            "velocity": [row.velocity for row in states],
        }
        if state.kind != "rigid_object":
            values.update(
                {
                    name: [getattr(row, name) for row in states]
                    for name in ("qpos", "qvel", "target_qpos", "target_qvel", "qf")
                }
            )
        entity = bindings[state.uid]
        tensors = {
            name: _backend_tensor(value, device=view.sim.device).squeeze(0)
            for name, value in values.items()
        }
        if state.kind != "rigid_object" and validate_joint_limits:
            # Articulation.set_state clips positions and targets to current
            # limits. Reject before any writes instead of silently changing
            # measured snapshot fields during restoration.
            limits = torch.as_tensor(
                entity.body_data.qpos_limits, device=view.sim.device
            )[list(view.env_ids)]
            if (
                limits.shape != (len(view.env_ids), len(state.joint_names), 2)
                or bool(torch.isnan(limits).any())
                or bool((limits[..., 0] > limits[..., 1]).any())
            ):
                raise ValueError(
                    f"Entity {state.uid!r} has invalid joint position limits"
                )
            for name in ("qpos", "target_qpos"):
                tolerance = qpos_tolerance if name == "qpos" else 0.0
                if bool(
                    (
                        (limits[..., 0] - tensors[name] > tolerance)
                        | (tensors[name] - limits[..., 1] > tolerance)
                    ).any()
                ):
                    raise ValueError(
                        f"Snapshot {state.uid}.{name} exceeds current joint position limits"
                    )
        prepared[state.uid] = (entity, tensors)
    return prepared


def restore_state(
    view: SceneView, snapshot: SceneState, *, qpos_tolerance: float = 0.0
) -> None:
    """Restore a captured state without stepping, after complete preflight.

    Args:
        view: Original scene selection with unchanged bindings.
        snapshot: Owned state captured from the same simulation.
        qpos_tolerance: Maximum allowed measured joint-position limit excursion.
            Defaults to strict rejection. An explicit tolerance permits the
            articulation writer to clip measured positions by that amount;
            controller targets remain strict. Callers must verify the actual
            restored state against their acceptance tolerances.
    Raises:
        ValueError: For unsupported topology, foreign state, or saved joint
            positions/targets outside current limits, before any writes.
        RuntimeError: For native writes that failed; partial writes are not rolled back.
    """
    prepared = _preflight_restore(view, snapshot, qpos_tolerance=qpos_tolerance)

    try:
        for state in snapshot.batches[0]:
            entity, tensors = prepared[state.uid]
            if state.kind == "rigid_object":
                entity.clear_dynamics(env_ids=list(view.env_ids))
                entity.set_local_pose(tensors["pose"], env_ids=list(view.env_ids))
                if state.body_type == "dynamic":
                    entity.set_velocity(
                        lin_vel=tensors["velocity"][:, :3],
                        ang_vel=tensors["velocity"][:, 3:],
                        env_ids=list(view.env_ids),
                    )
            else:
                entity.set_state(
                    env_ids=list(view.env_ids),
                    root_pose=tensors["pose"],
                    root_velocity=tensors["velocity"],
                    qpos=tensors["qpos"],
                    qvel=tensors["qvel"],
                    target_qpos=tensors["target_qpos"],
                    target_qvel=tensors["target_qvel"],
                    qf=tensors["qf"],
                    clear_dynamics=True,
                )
        view.sync_render_state()
    except Exception as exc:
        raise RuntimeError(
            "Scene restore failed during native writes; scene may be partially restored"
        ) from exc


def restore_render_state(view: SceneView, snapshot: SceneState) -> None:
    """Publish measured state for rendering with physics disabled.

    Args:
        view: Original fixed-topology world selection.
        snapshot: Complete measured state with matching owner and bindings.
    Raises:
        ValueError: If physics is enabled or the snapshot cannot be represented.
        RuntimeError: For native publication failure; callers must discard the host.

    Measured joint positions are published exactly, including finite solver
    excursions beyond command limits. Controller targets and efforts are never
    written. This separate render-only path cannot be used for physical reset.
    """
    if getattr(view.sim, "physics_enabled", True):
        raise ValueError("render-state publication requires physics disabled")
    prepared = _preflight_restore(view, snapshot, validate_joint_limits=False)
    for state in snapshot.batches[0]:
        entity, _ = prepared[state.uid]
        if state.kind != "rigid_object" and not callable(
            getattr(
                getattr(entity.body_data, "articulation_view", None),
                "apply_state",
                None,
            )
        ):
            raise ValueError(f"Entity {state.uid!r} has no native state publisher")
    try:
        for state in snapshot.batches[0]:
            entity, tensors = prepared[state.uid]
            if state.kind == "rigid_object":
                entity.set_local_pose(tensors["pose"], env_ids=list(view.env_ids))
                if state.body_type == "dynamic":
                    entity.set_velocity(
                        lin_vel=tensors["velocity"][:, :3],
                        ang_vel=tensors["velocity"][:, 3:],
                        env_ids=list(view.env_ids),
                    )
            else:
                # Native Default also clamps measured writes to joint limits.
                # Temporarily admit finite solver excursions while dynamics is
                # disabled, then restore the exact physical limits before return.
                limits = entity.get_qpos_limits(env_ids=list(view.env_ids)).clone()
                expanded = limits.clone()
                expanded[..., 0] = torch.minimum(limits[..., 0], tensors["qpos"] - 1e-6)
                expanded[..., 1] = torch.maximum(limits[..., 1], tensors["qpos"] + 1e-6)
                changed = not torch.equal(limits, expanded)
                try:
                    if changed:
                        entity.set_qpos_limits(expanded, env_ids=list(view.env_ids))
                    entity.body_data.articulation_view.apply_state(
                        list(view.env_ids),
                        None,
                        root_pose=tensors["pose"],
                        root_velocity=tensors["velocity"],
                        qpos=tensors["qpos"],
                        qvel=tensors["qvel"],
                        clear_dynamics=False,
                    )
                    if getattr(entity, "device", None) == torch.device(
                        "cpu"
                    ) and not getattr(entity.body_data, "is_newton_backend", False):
                        entity._world.update(0.0)
                finally:
                    if changed:
                        entity.set_qpos_limits(limits, env_ids=list(view.env_ids))
        view.sync_render_state()
    except Exception as error:
        raise RuntimeError(
            "render-state publication failed; world may be partially modified"
        ) from error
