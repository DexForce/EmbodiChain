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

"""Measured, selective rigid-body physics changes at host-owned batch boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
import torch
from .query import SceneView

__all__ = [
    "capture_physics_properties",
    "RigidPhysicsState",
    "capture_rigid_physics",
    "apply_rigid_physics",
    "restore_rigid_physics",
]


@dataclass(frozen=True)
class RigidPhysicsState:
    """Owned mass, principal inertia and friction values for selected dynamic bodies.

    Args:
        env_ids: Ordered physical arena indices.
        bodies: Tuples of UID, masses, diagonal inertias and friction coefficients.
        backend: Physical backend identity.

    Capture through :func:`capture_rigid_physics`; restoration requires unchanged
    bindings in the original world. This is not a dynamics solver checkpoint.
    """

    env_ids: tuple[int, ...]
    bodies: tuple[
        tuple[str, tuple[float, ...], tuple[tuple[float, ...], ...], tuple[float, ...]],
        ...,
    ]
    backend: str
    _owner: object = field(repr=False, compare=False)
    _topology: tuple[tuple[object, ...], ...] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        ids = tuple(self.env_ids)
        if (
            not ids
            or any(type(i) is not int or i < 0 for i in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValueError("Physics snapshot needs unique nonnegative env_ids")
        owned = []
        for uid, masses, inertias, friction in self.bodies:
            if not isinstance(uid, str) or not uid.strip():
                raise ValueError("Physics body UID must be nonempty")
            masses, friction = tuple(masses), tuple(friction)
            inertias = tuple(tuple(row) for row in inertias)
            if not (len(masses) == len(friction) == len(inertias) == len(ids)):
                raise ValueError("Physics parameters must align with env_ids")
            if any(
                type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                for v in masses
            ):
                raise ValueError("Mass must be finite and positive")
            if any(
                type(v) not in (int, float) or not math.isfinite(v) or v < 0
                for v in friction
            ):
                raise ValueError("Friction must be finite and nonnegative")
            for row in inertias:
                if len(row) != 3 or any(
                    type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                    for v in row
                ):
                    raise ValueError(
                        "Principal inertia must have three finite positive values"
                    )
                if 2 * max(row) > sum(row) * (1 + 1e-5):
                    raise ValueError(
                        "Principal inertia violates the triangle inequality"
                    )
            owned.append((uid, masses, inertias, friction))
        if not owned or len({row[0] for row in owned}) != len(owned):
            raise ValueError("Physics snapshot needs unique body UIDs")
        object.__setattr__(self, "env_ids", ids)
        object.__setattr__(self, "bodies", tuple(owned))


def _bodies(view: SceneView, uids: tuple[str, ...]) -> dict[str, object]:
    view.check_topology()
    if view.sim.physics_backend not in {"default", "newton"}:
        raise ValueError("Rigid physics changes require a supported physical backend")
    if not uids or len(set(uids)) != len(uids):
        raise ValueError("uids must contain unique dynamic bodies")
    bindings = {uid: (kind, entity) for kind, uid, entity in view.entities()}
    result = {}
    for uid in uids:
        kind, entity = bindings.get(uid, (None, None))
        if kind != "rigid_object" or entity.body_type != "dynamic":
            raise ValueError(
                f"Physics variation requires a dynamic rigid object: {uid!r}"
            )
        data = getattr(entity, "body_data", None)
        if data is None or not data.body_view.is_ready:
            raise ValueError(f"Body {uid!r} needs a ready physical view")
        result[uid] = entity
    return result


def _values(
    value: object, shape: tuple[int, ...], *, device: object, positive: bool
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device).clone()
    if tensor.shape != shape or not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"Physical values must be finite with shape {shape}")
    if bool((tensor <= 0 if positive else tensor < 0).any()):
        raise ValueError(
            "Physical values must be positive"
            if positive
            else "Physical values must be nonnegative"
        )
    return tensor


def capture_rigid_physics(view: SceneView, uids: tuple[str, ...]) -> RigidPhysicsState:
    """Capture actual backend parameters without stepping or retaining buffers.

    Args:
        view: Selected world rows.
        uids: Dynamic rigid bodies whose parameters the host owns.
    Returns:
        Immutable CPU parameter values in selection order.
    """
    bindings = _bodies(view, uids)
    bodies = []
    n = len(view.env_ids)
    for uid, entity in bindings.items():
        mass = _values(
            entity.get_mass(env_ids=list(view.env_ids)),
            (n,),
            device="cpu",
            positive=True,
        )
        inertia = _values(
            entity.get_inertia(env_ids=list(view.env_ids)),
            (n, 3),
            device="cpu",
            positive=True,
        )
        friction = _values(
            entity.get_friction(env_ids=list(view.env_ids)),
            (n,),
            device="cpu",
            positive=False,
        )
        bodies.append(
            (
                uid,
                tuple(mass.tolist()),
                tuple(tuple(row) for row in inertia.tolist()),
                tuple(friction.tolist()),
            )
        )
    return RigidPhysicsState(
        view.env_ids, tuple(bodies), view.sim.physics_backend, view.sim, view.topology
    )


def apply_rigid_physics(
    view: SceneView,
    baseline: RigidPhysicsState,
    *,
    mass_scale: Mapping[str, torch.Tensor] | None = None,
    friction: Mapping[str, torch.Tensor] | None = None,
) -> RigidPhysicsState:
    """Apply absolute parameters derived from a captured baseline and verify reads.

    Args:
        view: Original selected world rows, with unchanged topology.
        baseline: Canonical parameter baseline; repeated calls do not accumulate.
        mass_scale: Positive per-row mass multipliers by UID. Inertia scales
            by the same factor, preserving the body's shape and mass frame.
        friction: Nonnegative absolute per-row friction coefficients by UID.
    Returns:
        Actual parameter snapshot after writes and successful round-trip checks.
    Raises:
        ValueError: For invalid input, before any native write.
        RuntimeError: For native or readback failure; the host must stop the batch.

    Only mass/inertia and friction vary here. Damping, geometry, gravity, solver
    settings and the time step are outside this operation's capability contract.
    No call advances physics or claims task success.
    """
    view.check_topology()
    if (
        not isinstance(baseline, RigidPhysicsState)
        or baseline._owner is not view.sim
        or baseline._topology != view.topology
        or baseline.env_ids != view.env_ids
        or baseline.backend != view.sim.physics_backend
    ):
        raise ValueError(
            "Physics baseline ownership, topology, backend or selection differs"
        )
    uids = tuple(row[0] for row in baseline.bodies)
    bindings = _bodies(view, uids)
    scales, frictions = dict(mass_scale or {}), dict(friction or {})
    if set(scales).union(frictions) - set(uids):
        raise ValueError("Physics changes name bodies outside the captured baseline")
    prepared = {}
    n = len(view.env_ids)
    for uid, masses, inertias, coefficients in baseline.bodies:
        scale = _values(
            scales.get(uid, torch.ones(n)), (n,), device=view.sim.device, positive=True
        )
        mass = _values(masses, (n,), device=view.sim.device, positive=True) * scale
        inertia = (
            _values(inertias, (n, 3), device=view.sim.device, positive=True)
            * scale[:, None]
        )
        mu = _values(
            frictions.get(uid, coefficients),
            (n,),
            device=view.sim.device,
            positive=False,
        )
        _values(mass, (n,), device=view.sim.device, positive=True)
        _values(inertia, (n, 3), device=view.sim.device, positive=True)
        prepared[uid] = (mass, inertia, mu)
    try:
        for uid, (mass, inertia, mu) in prepared.items():
            entity = bindings[uid]
            entity.set_mass(mass, env_ids=list(view.env_ids))
            entity.set_inertia(inertia, env_ids=list(view.env_ids))
            entity.set_friction(mu, env_ids=list(view.env_ids))
        actual = capture_rigid_physics(view, uids)
        for uid, mass, inertia, mu in actual.bodies:
            for observed, expected in zip((mass, inertia, mu), prepared[uid]):
                if not torch.allclose(
                    torch.tensor(observed, device=expected.device),
                    expected,
                    atol=1e-7,
                    rtol=1e-5,
                ):
                    raise RuntimeError(f"Physical readback differs for {uid!r}")
        return actual
    except Exception as exc:
        raise RuntimeError(
            "Physics update failed; world may be partially modified"
        ) from exc


def restore_rigid_physics(view: SceneView, baseline: RigidPhysicsState) -> None:
    """Restore and verify the baseline without stepping.

    Args:
        view: Original world selection.
        baseline: Captured canonical parameters.
    """
    apply_rigid_physics(view, baseline)


def capture_physics_properties(view: SceneView) -> dict[str, torch.Tensor]:
    """Capture selected-row mass properties, material parameters and joint drives.

    Args:
        view: Existing scene selection with unchanged bindings.
    Returns:
        Owned tensors keyed by entity UID and property name. Limits may contain
        infinities; all other properties must be finite. Static rigid material
        reads follow the backend's public getter contract.
    """
    view.check_topology()
    properties = {}
    for kind, uid, entity in view.entities():
        # Static bodies deliberately have no RigidBodyData. Their material
        # parameters remain measurable, but dynamic mass properties do not.
        values = (
            {}
            if kind == "rigid_object" and entity.body_type == "static"
            else {
                name: getattr(entity.body_data, name)
                for name in ("mass", "inertia", "com_pose")
            }
        )
        if kind == "rigid_object":
            values.update(
                friction=entity.get_friction(env_ids=list(range(view._num_envs))),
                damping=entity.get_damping(env_ids=list(range(view._num_envs))),
            )
        else:
            values.update(
                {
                    name: getattr(entity.body_data, name)
                    for name in (
                        "joint_stiffness",
                        "joint_damping",
                        "joint_friction",
                        "joint_armature",
                        "qpos_limits",
                        "qvel_limits",
                    )
                }
            )
        for name, value in values.items():
            if (
                not isinstance(value, torch.Tensor)
                or not value.is_floating_point()
                or value.ndim == 0
                or value.shape[0] != view._num_envs
                or bool(torch.isnan(value).any())
                or (
                    not name.endswith("limits")
                    and not bool(torch.isfinite(value).all())
                )
            ):
                raise ValueError(f"Cannot measure physical parameter {uid}.{name}")
            properties[f"{uid}.{name}"] = value.detach()[list(view.env_ids)].clone()
    return properties
