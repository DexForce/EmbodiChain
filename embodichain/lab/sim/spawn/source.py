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

"""Resolve Newton articulation metadata before its first physics build."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np
from dexsim.spawn import ArticulationDesc

if TYPE_CHECKING:
    from dexsim.spawn import SceneBuilder

__all__ = ["resolve_articulation_source"]


def resolve_articulation_source(
    builder: SceneBuilder,
    desc: ArticulationDesc,
) -> ArticulationDesc:
    """Populate exact URDF metadata without building a Newton model.

    DexSim 0.4.3 removed its public source-resolution phase while retaining
    the same URDF-to-descriptor translator inside the Newton adapter. This
    compatibility boundary invokes that translator with a disposable
    render-only skeleton, allowing name-dependent EmbodiChain overlays to be
    authored before :meth:`SceneBuilder.finalize`.

    Args:
        builder: Scene builder that owns the target arena layout.
        desc: Articulation descriptor to resolve in place.

    Returns:
        The resolved descriptor.
    """
    signature = _source_signature(desc)
    previous = getattr(desc, "_embodichain_source_signature", None)
    if previous == signature:
        return desc

    if desc.urdf_path is None:
        setattr(desc, "_embodichain_source_signature", signature)
        return desc

    if previous is not None:
        desc.links = []
        desc.joints = []
        desc.root_link_name = None

    arena = _source_arena(builder, desc)
    temp_name = f"__embodichain_resolve__{desc.name.replace('/', '__')}__{id(desc)}"
    skeleton = arena.create_skeleton("skeleton")
    if skeleton is None:
        raise RuntimeError(f"Failed to create a source resolver for {desc.name!r}.")
    skeleton.set_name(temp_name)
    skeleton.detach_parent()
    try:
        scale = np.asarray(desc.body_scale, dtype=np.float32).reshape(3)
        load_result = skeleton.load_urdf(os.path.abspath(desc.urdf_path), scale)
        if load_result != 0:
            raise RuntimeError(
                f"Skeleton.load_urdf({desc.urdf_path!r}) failed: {load_result}"
            )

        # DexSim currently exposes no public metadata-only resolver. Reuse the
        # adapter's source translator so its retained descriptor semantics stay
        # identical to the subsequent Newton build.
        from dexsim.spawn.adapters.newton_articulation_adapter import (
            _translate_urdf_articulation,
        )

        _translate_urdf_articulation(skeleton, desc)
    finally:
        # Drop the wrapper before deleting its Arena-owned native object.
        skeleton = None
        arena.remove_skeleton(temp_name)

    setattr(desc, "_embodichain_source_signature", signature)
    return desc


def _source_signature(desc: ArticulationDesc) -> tuple[object, ...]:
    if desc.urdf_path is None:
        return "explicit", id(desc)
    return (
        "urdf",
        os.path.abspath(desc.urdf_path),
        tuple(float(value) for value in np.asarray(desc.body_scale).reshape(3)),
    )


def _source_arena(builder: SceneBuilder, desc: ArticulationDesc) -> Any:
    if desc.per_env and builder.replicate_plan is not None:
        arenas = builder.prepare_arenas()
        if not arenas:
            raise RuntimeError(
                f"No replicated Arena is available to resolve {desc.name!r}."
            )
        return arenas[0]
    return builder.world.get_env()
