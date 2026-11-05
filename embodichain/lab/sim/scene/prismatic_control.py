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

"""Source prismatic topology, limit-cache and placement qualification."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING
import numpy as np
import torch
from .articulation_inputs import PrismaticBinding, handle_mesh

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Articulation, RigidObject

__all__ = [
    "synchronize_prismatic_joint_limits",
    "validate_prismatic_placement",
    "measure_native_support_top",
]


def synchronize_prismatic_joint_limits(
    binding: PrismaticBinding, art: Articulation
) -> None:
    """Validate native ownership and refresh only the stale public limit cache.

    Args:
        binding: Qualified source joint identity and scaled limit pair.
        art: Existing native articulation; no simulator is created here.
    """
    if art is None or binding.joint not in art.joint_names:
        raise ValueError(
            "Native prismatic joint identity differs from its declaration."
        )
    joint = next(
        j for j in art.get_parent_joint_chain(binding.link) if j.joint_type != "fixed"
    )
    if (
        joint.name != binding.joint
        or joint.joint_type != "prismatic"
        or joint.parent_link_name != binding.parent
    ):
        raise ValueError(
            "The native prismatic joint does not own the declared handle link."
        )
    if joint.joint_limits is None:
        raise ValueError("Native prismatic joint limits are unavailable.")
    joint_index = art.joint_names.index(binding.joint)
    cached_limits = art.get_qpos_limits()
    selected_limits = cached_limits[:, joint_index, :]
    native_pair = selected_limits.new_tensor(joint.joint_limits)
    declared_pair = selected_limits.new_tensor(binding.limits)
    unscaled_pair = declared_pair / binding.scale
    native_is_unscaled = bool(binding.part_id) and torch.allclose(
        native_pair, unscaled_pair, atol=1e-6, rtol=1e-5
    )
    if (
        not torch.allclose(native_pair, declared_pair, atol=1e-6, rtol=1e-5)
        and not native_is_unscaled
    ):
        raise ValueError(
            "Native prismatic limits differ from the scaled asset binding: "
            f"native={native_pair.detach().cpu().tolist()}, "
            f"declared={declared_pair.detach().cpu().tolist()}."
        )
    resolved_pair = declared_pair if native_is_unscaled else native_pair
    if not torch.allclose(selected_limits, resolved_pair, atol=1e-6, rtol=1e-5):
        # Other columns can still contain pre-scale cache values. Never write
        # them back while synchronizing this joint's already-active limits.
        values = resolved_pair.expand(cached_limits.shape[0], 1, 2).clone()
        art.set_qpos_limits(values, joint_ids=[joint_index])


def validate_prismatic_placement(
    binding: PrismaticBinding, config: dict[str, Any], table_top_z: float | None
) -> None:
    """Qualify authored horizontal prismatic geometry above measured support.

    Args:
        binding: Qualified source joint/handle identity and body scale.
        config: Source articulation geometry and declared world pose.
        table_top_z: Explicit measured support height, never inferred by a name.
    """
    from .articulation_geometry import _root_pose, read_articulation_geometry

    if table_top_z is None or not np.isfinite(table_top_z):
        raise ValueError("E6 placement requires a measured tabletop height.")
    geometry = read_articulation_geometry(config["fpath"])
    if geometry.root_link != binding.parent:
        raise ValueError("E6 geometry root differs from the declared fixed parent.")
    vertices = geometry.world_vertices(config)
    if vertices[:, 2].min() < table_top_z - 0.002:
        raise ValueError(
            f"E6 articulation {binding.object_id!r} intersects the tabletop: "
            f"bottom={vertices[:, 2].min():.6f}, table={table_top_z:.6f}. "
            "Repair the source scene scale/placement before execution."
        )
    handle, _ = handle_mesh(binding, config["fpath"])
    link_pose = geometry.link_poses[binding.link]
    handle = handle @ link_pose[:3, :3].T + link_pose[:3, 3] * binding.scale
    pose = _root_pose(config)
    slide_axis_world = pose[:3, :3] @ (
        link_pose[:3, :3] @ np.asarray(binding.axis, dtype=float)
    )
    slide_axis_world /= np.linalg.norm(slide_axis_world)
    if abs(float(slide_axis_world[2])) > np.sin(np.deg2rad(1.0)):
        raise ValueError(
            "GenSim E6 requires a passive slide axis within one degree of "
            "horizontal; repair the source articulation layout before execution."
        )
    handle = handle @ pose[:3, :3].T + pose[:3, 3]
    if handle[:, 2].min() <= table_top_z:
        raise ValueError("E6 handle is not clear of the tabletop.")


def measure_native_support_top(support: RigidObject) -> float:
    """Measure the source first-row native mesh top without stepping.

    Args:
        support: Explicit native rigid support owner.

    Returns:
        Highest input vertex in world metres using the observed rigid pose.
    """
    pose = support.get_local_pose(to_matrix=True)[0]
    vertices = support.get_vertices(scale=True)[0]
    return float((vertices @ pose[:3, :3].T + pose[:3, 3])[:, 2].max())
