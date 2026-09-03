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
"""Newton-specific hooks used by the backend-neutral Scene batch views."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
from dexsim.scene import Scene

if TYPE_CHECKING:
    from dexsim.scene import RigidBodyBatch

__all__ = ["is_newton_scene"]


def is_newton_scene(scene: object) -> bool:
    """Return whether an object is a current DexSim Newton Scene."""
    return isinstance(scene, Scene) and scene.backend == "newton"


def _create_newton_standalone_state_sync(
    model: Any,
    body_ids: Sequence[int],
) -> Any:
    """Create DexSim's reusable FREE-joint synchronization selection."""
    from dexsim.engine.newton_physics.rigid_body.state_sync import (
        StandaloneRigidStateSync,
    )

    return StandaloneRigidStateSync.from_body_ids(model, body_ids)


def _synchronize_standalone_rigid_body_state(
    scene: Scene,
    batch: RigidBodyBatch,
    cached: tuple[int, Any, Any] | None,
) -> tuple[int, Any, Any]:
    """Synchronize Newton FREE-joint state after a Scene batch write."""
    topology_revision = int(scene.topology_revision)
    if cached is None or cached[0] != topology_revision:
        # Accessing ``_binding`` refreshes a stale stable batch. DexSim
        # currently exposes neither the Newton runtime nor this required
        # synchronization through the public Batch API.
        binding = batch._binding
        runtime = getattr(binding, "_runtime", None)
        indices = getattr(binding, "_indices", None)
        if runtime is None or indices is None:
            raise RuntimeError(
                "Newton rigid-body batch has no finalized runtime selection."
            )
        state_sync = _create_newton_standalone_state_sync(
            runtime.model,
            indices.detach().cpu().tolist(),
        )
        cached = (topology_revision, runtime, state_sync)

    _, runtime, state_sync = cached
    state_sync.synchronize((runtime.current_state, runtime.other_state))
    return cached


_DEFAULT_MIMIC_NATURAL_FREQUENCY = 1.0e3
_DEFAULT_MIMIC_DAMPING_RATIO = 1.0e1
_MIMIC_FOLLOWER_TARGET_GAIN_RATIO = 1.0e-2


def _default_mujoco_mimic_solref(physics_dt: float, num_substeps: int) -> np.ndarray:
    """Approximate Default's mimic compliance with MuJoCo solref.

    Positive MuJoCo solref uses (timeconst, dampratio) and therefore retains
    the effective-mass scaling of PhysX articulation mimic joints. MuJoCo's
    reference-safety rule clamps timeconst to twice the solver timestep, so
    apply the same bound explicitly.
    """
    if not np.isfinite(physics_dt) or physics_dt <= 0.0:
        raise ValueError("Newton physics_dt must be finite and positive.")
    if num_substeps <= 0:
        raise ValueError("Newton num_substeps must be positive.")

    solver_dt = physics_dt / num_substeps
    natural_time_constant = 1.0 / (
        _DEFAULT_MIMIC_NATURAL_FREQUENCY * _DEFAULT_MIMIC_DAMPING_RATIO
    )
    return np.asarray(
        (
            max(natural_time_constant, 2.0 * solver_dt),
            _DEFAULT_MIMIC_DAMPING_RATIO,
        ),
        dtype=np.float32,
    )


def _configure_newton_mimic_compliance(
    *,
    result: Scene | None,
    entities: Sequence[object],
    state_joint_names: Sequence[str],
    mimic_ids: Sequence[int],
    mimic_parents: Sequence[int],
) -> bool:
    """Tune native MuJoCo-Warp mimic constraints toward Default behavior.

    MuJoCo's default equality solref is underdamped relative to Default's
    articulation mimic. Map Default's natural-frequency and damping-ratio
    parameters to MuJoCo's mass-scaled positive convention as a stable
    approximation. A follower drive with one percent of its leader's gains also
    tracks the leader's target relation between solver updates. Keeping the
    native equality rows enabled preserves mechanical force coupling; the drive
    is only a stabilizer and never mirrors measured follower state.
    """
    if result is None or result.backend != "newton" or not mimic_ids:
        return False

    from dexsim.engine.newton_physics.backend_registry import get_newton_backend

    backend = get_newton_backend(result.world)
    if (
        backend is None
        or backend.solver_type != "mujoco_warp"
        or backend.cfg.requires_grad
        or (backend.model is not None and backend.model.requires_grad)
    ):
        return False

    relation_names = [
        (state_joint_names[child_id], state_joint_names[parent_id])
        for child_id, parent_id in zip(mimic_ids, mimic_parents, strict=True)
    ]
    first_binding = getattr(entities[0], "_physics_binding", None)
    runtime = getattr(first_binding, "_runtime", None)
    if runtime is None:
        raise RuntimeError("Newton Scene articulation has no finalized runtime.")

    model = runtime.model
    target_ke = np.asarray(model.joint_target_ke.numpy()).reshape(-1)
    target_kd = np.asarray(model.joint_target_kd.numpy()).reshape(-1)
    target_mode = np.asarray(model.joint_target_mode.numpy()).reshape(-1)
    expected_pairs: set[tuple[int, int]] = set()
    for entity in entities:
        binding = getattr(entity, "_physics_binding", None)
        if binding is None or getattr(binding, "_runtime", None) is not runtime:
            raise RuntimeError(
                "Newton mimic configuration requires one shared finalized runtime."
            )
        runtime_joints = {joint.name: joint for joint in binding.joints}
        follower_ke: list[float] = []
        follower_kd: list[float] = []
        follower_mode: list[int] = []
        for child_name, parent_name in relation_names:
            try:
                child = runtime_joints[child_name]
                parent = runtime_joints[parent_name]
            except KeyError as error:
                raise RuntimeError(
                    "Newton mimic metadata references a missing runtime joint."
                ) from error
            if int(child.qd_size) != 1 or int(parent.qd_size) != 1:
                raise NotImplementedError(
                    "MuJoCo-Warp mimic compliance requires scalar joints."
                )
            expected_pairs.add((int(child.joint_id), int(parent.joint_id)))
            parent_dof = int(parent.qd_start)
            follower_ke.append(
                float(target_ke[parent_dof]) * _MIMIC_FOLLOWER_TARGET_GAIN_RATIO
            )
            follower_kd.append(
                float(target_kd[parent_dof]) * _MIMIC_FOLLOWER_TARGET_GAIN_RATIO
            )
            follower_mode.append(int(target_mode[parent_dof]))

        configured = entity.set_newton_drive(
            joint_ids=np.asarray(mimic_ids, dtype=np.int32),
            target_ke=np.asarray(follower_ke, dtype=np.float32),
            target_kd=np.asarray(follower_kd, dtype=np.float32),
            target_mode=np.asarray(follower_mode, dtype=np.int32),
        )
        if configured != len(mimic_ids):
            raise RuntimeError(
                "Newton failed to configure every mimic follower stabilizer."
            )

    mimic_joint0 = np.asarray(model.constraint_mimic_joint0.numpy()).reshape(-1)
    mimic_joint1 = np.asarray(model.constraint_mimic_joint1.numpy()).reshape(-1)
    row_by_pair = {
        (int(child), int(parent)): row
        for row, (child, parent) in enumerate(
            zip(mimic_joint0, mimic_joint1, strict=True)
        )
    }
    try:
        constraint_rows = np.asarray(
            [row_by_pair[pair] for pair in expected_pairs], dtype=np.int32
        )
    except KeyError as error:
        raise RuntimeError(
            f"Newton model has no mimic constraint for joint pair {error.args[0]}."
        ) from error

    solver = runtime.solver
    mapping = getattr(solver, "mjc_eq_to_newton_mimic", None)
    mjw_model = getattr(solver, "mjw_model", None)
    if mapping is None or mjw_model is None:
        raise RuntimeError("MuJoCo-Warp did not materialize Newton mimic rows.")

    mapping_values = np.asarray(mapping.numpy())
    selected = np.isin(mapping_values, constraint_rows)
    if int(selected.sum()) != len(constraint_rows):
        raise RuntimeError(
            "MuJoCo-Warp mimic row mapping does not match the articulation."
        )
    eq_solref = np.asarray(mjw_model.eq_solref.numpy()).copy()
    mimic_solref = _default_mujoco_mimic_solref(
        float(backend.cfg.dt),
        int(backend.cfg.num_substeps),
    )
    eq_solref[selected] = mimic_solref
    mjw_model.eq_solref.assign(eq_solref)

    # Keep the optional CPU mirror coherent for debugging and CPU execution.
    mj_model = getattr(solver, "mj_model", None)
    if mj_model is not None and len(eq_solref) > 0:
        mj_model.eq_solref[:] = eq_solref[0]
    return True
