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
"""Adapt public mjfit/MuJoCo interfaces into portable assembly assets."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import metadata
import json
from pathlib import Path
import subprocess
from typing import Any
from types import ModuleType
from urllib.parse import unquote, urlparse

import numpy as np

from .model import AssemblyAsset, AssemblyBody, AssemblyGeometry, AssemblyJoint

__all__ = [
    "MJFIT_REVISION",
    "build_assembly",
    "import_mjcf",
    "sample_assemblies",
    "load_mjfit",
]

MJFIT_REVISION = "5833f465896e1e06bc4be4f003be847f5183a4db"


def load_mjfit() -> tuple[ModuleType, str]:
    """Load the optional mjfit provider and verify its pinned provenance.

    Returns:
        Public generator module and its verified source revision.

    Raises:
        ImportError: If the optional generator is unavailable.
        RuntimeError: If its installed source revision does not match the pin.
    """
    try:
        import mjfit
    except ImportError as error:
        raise ImportError(
            "Assembly generation requires the pinned mjfit provider: "
            f"pip install 'mjfit @ git+https://github.com/younghyopark/mjfit.git@{MJFIT_REVISION}'."
        ) from error
    revision = "unverified"
    try:
        direct = json.loads(
            metadata.distribution("mjfit").read_text("direct_url.json") or "{}"
        )
        revision = direct.get("vcs_info", {}).get("commit_id", revision)
        source = urlparse(direct.get("url", ""))
        if revision == "unverified" and source.scheme == "file":
            checkout = Path(unquote(source.path))
            if (checkout / ".git").exists():
                revision = subprocess.check_output(
                    ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
                ).strip()
    except (
        metadata.PackageNotFoundError,
        ValueError,
        OSError,
        subprocess.CalledProcessError,
    ):
        pass
    if revision != MJFIT_REVISION:
        raise RuntimeError(
            f"mjfit source revision is {revision!r}; expected {MJFIT_REVISION}. "
            "Install the pinned mjfit provider so generated assets have verified provenance."
        )
    return mjfit, revision


def build_assembly(
    config: str | Path | Mapping[str, Any],
    *,
    seed: int = 0,
    cache_dir: str | Path | None = None,
    **overrides: Any,
) -> AssemblyAsset:
    """Generate one asset from a bundled name, YAML or public mjfit config.

    The seed identifies an already-sampled configuration; fixed configurations
    are deterministic. Use :func:`sample_assemblies` for randomized pegs.
    Existing self-contained manifests can be loaded without this dependency.
    """
    mjfit, revision = load_mjfit()
    cfg = {
        **mjfit.load(str(config) if isinstance(config, Path) else config),
        **overrides,
    }
    # Include standalone solver options for the reference. The adapter discards
    # worldbody decorations and floor, retaining only named physical bodies.
    xml = mjfit.build(cfg, standalone=True)
    asset = import_mjcf(xml, config=cfg, seed=seed, generator_revision=revision)
    asset.generator_version = mjfit.__version__
    if cache_dir is not None:
        asset.save(Path(cache_dir) / f"{asset.content_hash}.json")
    return asset


def sample_assemblies(
    count: int, *, seed: int = 0, family: str = "convex", **parameters: Any
) -> list[AssemblyAsset]:
    """Draw mjfit's stable seeded peg distribution using its public sampler."""
    if count < 0:
        raise ValueError("Assembly sample count must be non-negative.")
    mjfit, _ = load_mjfit()
    return [
        build_assembly(config, seed=seed)
        for config in mjfit.sample_pegs(count, seed=seed, family=family, **parameters)
    ]


def import_mjcf(
    xml: str,
    *,
    config: Mapping[str, Any] | None = None,
    seed: int = 0,
    generator_revision: str = "external",
) -> AssemblyAsset:
    """Import the bounded MJCF subset emitted by mjfit, without private APIs.

    Supported parts attach directly to world and have zero joints, one free
    joint, or one unlimited hinge. Unsupported constraints fail explicitly.
    Compiled mesh vertices already include MuJoCo's principal-axis change;
    compiled geom poses transform those vertices into the owning body frame.
    """
    try:
        import mujoco
    except ImportError as error:
        raise ImportError(
            "MJCF conversion requires MuJoCo; manifest loading does not."
        ) from error
    from scipy.spatial.transform import Rotation

    model = mujoco.MjModel.from_xml_string(xml)
    if model.neq or model.ntendon:
        raise ValueError(
            "Assembly importer does not support equality constraints or tendons."
        )
    joint_kinds = {
        int(mujoco.mjtJoint.mjJNT_FREE): "free",
        int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
    }
    kinds = {
        int(mujoco.mjtGeom.mjGEOM_MESH): "mesh",
        int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
        int(mujoco.mjtGeom.mjGEOM_BOX): "box",
        int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
    }
    bodies = []
    for body_id in range(1, model.nbody):
        if model.body_parentid[body_id] != 0 or model.body_jntnum[body_id] > 1:
            raise ValueError(
                "Only world-attached mjfit parts with at most one joint are supported."
            )
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if not name:
            raise ValueError("Assembly physical bodies must have names.")
        joint = AssemblyJoint()
        if model.body_jntnum[body_id]:
            jid = int(model.body_jntadr[body_id])
            kind = joint_kinds.get(int(model.jnt_type[jid]))
            if kind is None or (kind == "hinge" and model.jnt_limited[jid]):
                raise ValueError("Only free joints and unlimited hinges are supported.")
            dof = int(model.jnt_dofadr[jid])
            joint = AssemblyJoint(
                name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                or f"{name}_joint",
                kind=kind,
                position=model.jnt_pos[jid].tolist(),
                axis=model.jnt_axis[jid].tolist(),
                damping=float(model.dof_damping[dof]),
                armature=float(model.dof_armature[dof]),
                frictionloss=float(model.dof_frictionloss[dof]),
            )
            motors = [a for a in range(model.nu) if model.actuator_trnid[a, 0] == jid]
            if len(motors) > 1:
                raise ValueError(
                    "Only one simple motor per assembly hinge is supported."
                )
            if motors:
                aid = motors[0]
                if int(model.actuator_trntype[aid]) != int(mujoco.mjtTrn.mjTRN_JOINT):
                    raise ValueError("Assembly motor must use a joint transmission.")
                if model.actuator_dyntype[aid] or model.actuator_biastype[aid]:
                    raise ValueError(
                        "Assembly actuators must be stateless torque motors."
                    )
                joint.motor_gear = float(model.actuator_gear[aid, 0])
                joint.control_range = model.actuator_ctrlrange[aid].tolist()
        geometries = []
        for gid in range(
            int(model.body_geomadr[body_id]),
            int(model.body_geomadr[body_id] + model.body_geomnum[body_id]),
        ):
            kind = kinds.get(int(model.geom_type[gid]))
            if kind is None:
                raise ValueError(
                    f"Unsupported assembly geom type {model.geom_type[gid]}."
                )
            masks = (int(model.geom_contype[gid]), int(model.geom_conaffinity[gid]))
            if masks not in {(0, 0), (1, 1)}:
                raise ValueError(
                    "Assembly importer supports only the mjfit collision masks (0,0) and (1,1)."
                )
            collision = masks == (1, 1)
            geom = AssemblyGeometry(
                name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
                or f"{name}_geom_{gid}",
                kind=kind,
                position=model.geom_pos[gid].tolist(),
                quaternion=model.geom_quat[gid][[1, 2, 3, 0]].tolist(),
                size=model.geom_size[gid].tolist(),
                rgba=model.geom_rgba[gid].tolist(),
                collision=collision,
                visible=int(model.geom_group[gid]) != 3,
                friction=model.geom_friction[gid].tolist(),
                condim=int(model.geom_condim[gid]),
                priority=int(model.geom_priority[gid]),
                solmix=float(model.geom_solmix[gid]),
                solref=model.geom_solref[gid].tolist(),
                solimp=model.geom_solimp[gid].tolist(),
            )
            if kind == "mesh":
                mid = int(model.geom_dataid[gid])
                va, vn = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
                fa, fn = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
                geom.vertices = model.mesh_vert[va : va + vn].astype(float).tolist()
                geom.triangles = model.mesh_face[fa : fa + fn].astype(int).tolist()
            geometries.append(geom)
        inertia_rotation = Rotation.from_quat(
            model.body_iquat[body_id][[1, 2, 3, 0]]
        ).as_matrix()
        inertia = (
            inertia_rotation @ np.diag(model.body_inertia[body_id]) @ inertia_rotation.T
        )
        bodies.append(
            AssemblyBody(
                name=name,
                position=model.body_pos[body_id].tolist(),
                quaternion=model.body_quat[body_id][[1, 2, 3, 0]].tolist(),
                mass=float(model.body_mass[body_id]),
                com=model.body_ipos[body_id].tolist(),
                inertia=inertia.tolist(),
                geometries=geometries,
                joint=joint,
            )
        )
    return AssemblyAsset(
        generator_revision=generator_revision,
        source_mjcf=xml,
        config=dict(config or {}),
        seed=int(seed),
        bodies=bodies,
        source_options={
            "mujoco_version": mujoco.__version__,
            "timestep": float(model.opt.timestep),
            "solver": int(model.opt.solver),
            "integrator": int(model.opt.integrator),
            "gravity": model.opt.gravity.tolist(),
        },
    )
