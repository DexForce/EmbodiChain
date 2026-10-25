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
"""Independent geometry acceptance and matched passive assembly probes.

These probes report measurements and missing backend capabilities explicitly.
A successful drop does not qualify lateral/tilt/yaw clearance accuracy.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
import time
from typing import Any

import numpy as np

from embodichain.utils import configclass
from .model import AssemblyAsset

__all__ = ["PrecisionRunCfg", "compare_usd_geometry", "run_passive_probe"]


@configclass
class PrecisionRunCfg:
    """A passive probe's reproducible workload, route and physical settings."""

    backend: str = "mujoco_cpu"
    route: str = "direct"
    num_envs: int = 1
    duration: float = 0.2
    timestep: float = 0.001
    spin_rate: float = 15.0
    contact_offset: float = 0.0
    rest_offset: float = 0.0
    solver_iterations: int = 100
    solver_tolerance: float | None = None
    """Explicit MuJoCo stopping tolerance; None preserves each route's default."""
    enable_multiccd: bool | None = None
    """Explicit MuJoCo MultiCCD selection; None preserves each route's default."""
    contact_capacity: int = 4096
    source_contact: bool = True
    solver: str | None = None
    """Explicit solver comparison; None preserves the source solver."""
    use_mujoco_cpu: bool = False
    """Use Newton's MuJoCo CPU route; never an implicit Warp fallback."""
    warp_reference_initialization: str = "cpu_compiled"
    """Reference constants: cpu_compiled, or explicit Warp set_const_0 recomputation."""
    work_dir: str = "outputs/assembly_precision"


def compare_usd_geometry(asset: AssemblyAsset, usd_file: str | Path) -> dict[str, Any]:
    """Measure local vertices and authored inertia against independent USD I/O.

    This checks the authored stage rather than its backend cooking. Collider
    ownership is also checked through DexSim's public USD parser when present.
    """
    from pxr import Usd, UsdGeom, UsdPhysics
    from scipy.spatial.transform import Rotation

    stage = Usd.Stage.Open(str(usd_file))
    if stage is None:
        raise ValueError(f"Cannot open assembly USD {usd_file}.")
    max_error = 0.0
    mass_error = 0.0
    inertia_error = 0.0
    com_error = 0.0
    checked_colliders = 0
    xform_cache = UsdGeom.XformCache()
    for body in asset.bodies:
        base = f"/Assembly/{body.name}"
        if body.joint.kind == "hinge":
            base += f"/{body.name}_link"
        prim = stage.GetPrimAtPath(base)
        if not prim:
            raise ValueError(f"Missing body root {base}.")
        for index, geom in enumerate(body.geometries):
            geom_prim = stage.GetPrimAtPath(f"{base}/geom_{index}")
            if not geom_prim:
                raise ValueError(f"USD lost geometry {body.name}/{index}.")
            if geom.collision:
                checked_colliders += 1
                if not geom_prim.HasAPI(UsdPhysics.CollisionAPI):
                    raise ValueError("USD lost a collision shape.")
            world = np.asarray(
                xform_cache.GetLocalToWorldTransform(geom_prim), dtype=float
            ).T
            source_rotation = Rotation.from_quat(geom.quaternion).as_matrix()
            if geom.kind == "mesh":
                source_points = np.asarray(geom.vertices)
                points = np.asarray(
                    UsdGeom.Mesh(geom_prim).GetPointsAttr().Get(), dtype=float
                )
            elif geom.kind == "box":
                from itertools import product

                corners = np.asarray(list(product((-1.0, 1.0), repeat=3)))
                source_points = corners * geom.size[:3]
                points = (
                    corners * float(UsdGeom.Cube(geom_prim).GetSizeAttr().Get()) / 2
                )
            elif geom.kind == "sphere":
                directions = np.concatenate((np.eye(3), -np.eye(3)))
                source_points = directions * geom.size[0]
                points = directions * float(
                    UsdGeom.Sphere(geom_prim).GetRadiusAttr().Get()
                )
            elif geom.kind == "cylinder":
                angles = np.arange(8) * np.pi / 4
                directions = np.concatenate(
                    [
                        np.column_stack((np.cos(angles), np.sin(angles), np.full(8, z)))
                        for z in (-1.0, 1.0)
                    ]
                )
                source_points = directions * [geom.size[0], geom.size[0], geom.size[1]]
                cylinder = UsdGeom.Cylinder(geom_prim)
                radius = float(cylinder.GetRadiusAttr().Get())
                height = float(cylinder.GetHeightAttr().Get())
                points = directions * [radius, radius, height / 2]
                axis = cylinder.GetAxisAttr().Get()
                if axis == "X":
                    points = points[:, [2, 1, 0]]
                elif axis == "Y":
                    points = points[:, [0, 2, 1]]
                elif axis != "Z":
                    raise ValueError(f"Unsupported USD cylinder axis {axis!r}.")
            else:
                raise ValueError(f"Unsupported assembly geometry {geom.kind!r}.")
            source_body_points = source_points @ source_rotation.T + geom.position
            body_rotation = Rotation.from_quat(body.quaternion).as_matrix()
            source = source_body_points @ body_rotation.T + body.position
            target = points @ world[:3, :3].T + world[:3, 3]
            if source.shape != target.shape:
                raise ValueError("USD changed the geometry vertex count.")
            max_error = max(
                max_error, float(np.max(np.linalg.norm(source - target, axis=1)))
            )
        if body.body_type == "dynamic":
            mass = UsdPhysics.MassAPI(prim)
            mass_error = max(
                mass_error, abs(float(mass.GetMassAttr().Get()) - body.mass) / body.mass
            )
            com_error = max(
                com_error,
                float(
                    np.linalg.norm(
                        np.asarray(mass.GetCenterOfMassAttr().Get()) - body.com
                    )
                ),
            )
            q = mass.GetPrincipalAxesAttr().Get()
            rotation = Rotation.from_quat([*q.GetImaginary(), q.GetReal()]).as_matrix()
            target = (
                rotation @ np.diag(mass.GetDiagonalInertiaAttr().Get()) @ rotation.T
            )
            source = np.asarray(body.inertia)
            inertia_error = max(
                inertia_error,
                float(np.linalg.norm(target - source) / np.linalg.norm(source)),
            )
    ownership = "unmeasured"
    try:
        from dexsim.kit.usd import parse_usd
    except ImportError:
        pass
    else:
        parsed = parse_usd(str(usd_file), root_name="assembly")
        counts = [
            len(body.collisions) for body in parsed.mesh_objects if body.collisions
        ]
        counts += [
            sum(len(link.collisions) for link in art.links)
            for art in parsed.articulations
        ]
        expected = [sum(g.collision for g in body.geometries) for body in asset.bodies]
        if sorted(counts) != sorted(expected):
            raise ValueError(
                f"USD parser changed body/collider ownership: {counts} != {expected}."
            )
        ownership = "pass"
    return {
        "geometry_error_m": max_error,
        "mass_relative_error": mass_error,
        "inertia_relative_error": inertia_error,
        "com_error_m": com_error,
        "colliders": checked_colliders,
        "body_ownership": ownership,
        "passed": max_error <= 1e-6
        and mass_error <= 1e-5
        and inertia_error <= 1e-5
        and com_error <= 1e-6
        and ownership == "pass",
    }


def run_passive_probe(asset: AssemblyAsset, cfg: PrecisionRunCfg) -> dict[str, Any]:
    """Execute physical falling, screwing, or contact-driven gear rotation.

    Only angular speed is prescribed for the screw; axial motion comes from
    contact. Gear followers receive no control. Invalid numeric/overflow runs
    fail, and unavailable dependencies return an explicit skipped result.
    """
    if cfg.backend not in {"mujoco_cpu", "mujoco_warp", "newton", "default"}:
        raise ValueError(f"Unknown precision backend {cfg.backend!r}.")
    if cfg.route not in {"direct", "usd"}:
        raise ValueError("Assembly route must be direct or usd.")
    if cfg.backend in {"mujoco_cpu", "mujoco_warp"} and cfg.route != "direct":
        raise ValueError(
            "Native MuJoCo references consume source MJCF and require the direct route."
        )
    if cfg.num_envs <= 0 or cfg.timestep <= 0 or cfg.duration <= 0:
        raise ValueError(
            "Probe environment count, timestep and duration must be positive."
        )
    if round(cfg.duration / cfg.timestep) < 1:
        raise ValueError("Probe duration must include at least one physical timestep.")
    if cfg.solver_iterations <= 0 or cfg.contact_capacity <= 0:
        raise ValueError("Solver iterations and contact capacity must be positive.")
    if cfg.solver not in {None, "pgs", "cg", "newton"}:
        raise ValueError("Solver must be pgs, cg, newton, or None for the source.")
    if cfg.use_mujoco_cpu and cfg.backend != "newton":
        raise ValueError("use_mujoco_cpu selects the Newton CPU integration only.")
    if cfg.use_mujoco_cpu and cfg.num_envs != 1:
        raise ValueError("Newton CPU precision probes currently support one world.")
    _validate_solver_overrides(cfg)
    if not (math.isfinite(cfg.contact_offset) and math.isfinite(cfg.rest_offset)) or (
        cfg.contact_offset < 0 or cfg.contact_offset < cfg.rest_offset
    ):
        raise ValueError(
            "Contact offset must be finite, non-negative, and no smaller than rest offset."
        )
    result: dict[str, Any] = {
        "backend": cfg.backend,
        "route": cfg.route,
        "asset_sha256": asset.content_hash,
        "config": cfg.to_dict(),
        "status": "unverified",
        "metrics": {},
    }
    try:
        measured = (
            _run_mujoco(asset, cfg)
            if cfg.backend in {"mujoco_cpu", "mujoco_warp"}
            else _run_embodichain(asset, cfg)
        )
    except ImportError as error:
        result.update(status="skipped", reason=str(error))
    except Exception as error:
        # This is an experiment boundary: retain unsupported meshes and native
        # failures in the matrix instead of losing the entire measurement file.
        result.update(status="failed", reason=f"{type(error).__name__}: {error}")
    else:
        result.update(measured)
    return _json_measurement(result)


def _validate_solver_overrides(cfg: PrecisionRunCfg) -> None:
    """Reject calibration settings that would otherwise be ignored."""
    if cfg.warp_reference_initialization not in {"cpu_compiled", "warp_recomputed"}:
        raise ValueError(
            "Warp reference initialization must be cpu_compiled or warp_recomputed."
        )
    if (
        cfg.warp_reference_initialization != "cpu_compiled"
        and cfg.backend != "mujoco_warp"
    ):
        raise ValueError(
            "Recomputed reference initialization is only for MuJoCo Warp references."
        )
    if cfg.solver_tolerance is not None and (
        not math.isfinite(cfg.solver_tolerance) or cfg.solver_tolerance < 0
    ):
        raise ValueError("Solver tolerance must be finite and non-negative.")
    if cfg.enable_multiccd is not None and not isinstance(cfg.enable_multiccd, bool):
        raise ValueError("enable_multiccd must be a bool or None.")
    if cfg.backend == "default" and any(
        value is not None
        for value in (cfg.solver, cfg.solver_tolerance, cfg.enable_multiccd)
    ):
        raise ValueError(
            "Default does not accept MuJoCo solver or calibration overrides."
        )


def _json_measurement(result: dict[str, Any]) -> dict[str, Any]:
    """Keep failed numeric measurements representable in strict JSON."""
    nonfinite = 0

    def convert(value: Any) -> Any:
        nonlocal nonfinite
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            nonfinite += 1
            return None
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value

    measured = convert(result)
    if nonfinite:
        measured.update(
            status="failed",
            nonfinite_measurement_count=nonfinite,
            numeric_encoding="Nonfinite values are null; this run failed.",
        )
    return measured


def _score(
    asset: AssemblyAsset,
    initial: dict[str, np.ndarray],
    final: dict[str, np.ndarray],
    cfg: PrecisionRunCfg,
) -> dict[str, Any]:
    kind = asset.config.get("kind")
    metrics: dict[str, Any] = {}
    finite = all(np.isfinite(value).all() for value in final.values())
    if kind == "many_pegs":
        from .many import _score_many_pegs

        metrics = _score_many_pegs(asset, initial, final)
    elif kind == "peg":
        fell = initial["peg"][..., 2] - final["peg"][..., 2]
        scale = float(asset.config.get("scale", 1.0))
        # mjfit's body origins are the plate/insertion-section midpoints.
        # Their relative height also accounts for a raised blind-hole bottom.
        # Defaults match the pinned public peg() generator.
        thickness = float(asset.config.get("plate_thickness", 0.01)) * scale
        length = float(asset.config.get("peg_length", 0.03)) * scale
        clearance = float(asset.config.get("clearance", 1e-4)) * scale
        tolerance = min(5e-6 * scale, 0.01 * thickness)
        relative = final["peg"][..., :3] - final["plate"][..., :3]
        insertion = 0.5 * (thickness + length) - relative[..., 2]
        lateral = np.max(np.abs(relative[..., :2]), axis=-1)
        # Falling onto the rim (the usual 5 mm spawn gap), or falling outside
        # the plate, is not insertion. This is a coarse endpoint check only;
        # clearance/tilt/contact accuracy is qualified by separate probes.
        inserted = (insertion >= thickness - tolerance) & (
            lateral <= clearance + tolerance
        )
        metrics = {
            "fall_m": float(np.mean(fell)),
            "insertion_depth_m": float(np.mean(insertion)),
            "required_insertion_depth_m": thickness,
            "lateral_error_m": float(np.max(lateral)),
            "seating_tolerance_m": tolerance,
            "success_rate": float(np.mean(inserted)),
        }
    elif kind == "thread":
        advance = initial["bolt"][..., 2] - final["bolt"][..., 2]
        ideal = (
            cfg.spin_rate
            * round(cfg.duration / cfg.timestep)
            * cfg.timestep
            / (2 * math.pi)
            * float(asset.config["pitch"])
            * int(asset.config.get("n_starts", 1))
        )
        relative = np.abs(advance - ideal) / max(abs(ideal), 1e-12)
        metrics = {
            "advance_m": float(np.mean(advance)),
            "ideal_advance_m": ideal,
            "lead_relative_error": float(np.max(relative)),
            "success_rate": float(np.mean(relative <= 0.05)),
        }
    elif kind == "gear":
        if "gear_b" not in final:
            raise ValueError("Gear-ratio probe requires a gear pair.")
        a = final["gear_a"][..., 0] - initial["gear_a"][..., 0]
        b = final["gear_b"][..., 0] - initial["gear_b"][..., 0]
        teeth = asset.config.get("teeth", [18, 24])
        ideal = -float(teeth[0]) / float(teeth[1])
        relative = np.abs(b / np.where(np.abs(a) > 1e-9, a, np.nan) - ideal) / abs(
            ideal
        )
        valid = np.isfinite(relative) & (np.abs(a) > math.pi) & (relative <= 0.005)
        metrics = {
            "ratio_relative_error": (
                float(np.nanmax(relative)) if np.isfinite(relative).any() else None
            ),
            "driver_angle_rad": float(np.mean(a)),
            "success_rate": float(np.mean(valid)),
        }
    else:
        raise ValueError(
            "Passive assembly probes require config kind peg, many_pegs, thread or gear."
        )
    metrics["finite"] = finite
    return metrics


def _run_mujoco(
    asset: AssemblyAsset,
    cfg: PrecisionRunCfg,
    *,
    on_step: Callable[[Any, Any, int], None] | None = None,
) -> dict[str, Any]:
    _validate_solver_overrides(cfg)
    import mujoco

    # Warp's get_data_into reconstructs a dense CPU Jacobian even for a sparse
    # source model. Give its diagnostic copy enough arena storage; this changes
    # allocation only, never source geometry or physical solver settings.
    import xml.etree.ElementTree as ET

    source_xml = asset.source_mjcf
    if cfg.backend == "mujoco_warp":
        xml = ET.fromstring(source_xml)
        size = xml.find("size")
        if size is None:
            size = ET.SubElement(xml, "size")
        for legacy in ("nconmax", "njmax"):
            size.attrib.pop(legacy, None)
        size.set("memory", "512M")
        source_xml = ET.tostring(xml, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(source_xml)
    model.opt.timestep = cfg.timestep
    model.opt.iterations = cfg.solver_iterations
    if cfg.solver_tolerance is not None:
        model.opt.tolerance = cfg.solver_tolerance
    if cfg.enable_multiccd is not None:
        if cfg.enable_multiccd:
            model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
        else:
            model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MULTICCD
    if cfg.solver is not None:
        model.opt.solver = getattr(mujoco.mjtSolver, f"mjSOL_{cfg.solver.upper()}")
    model.geom_margin[:] = cfg.contact_offset
    model.geom_gap[:] = cfg.contact_offset - cfg.rest_offset
    effective_solver_options = {
        "solver": int(model.opt.solver),
        "tolerance": float(model.opt.tolerance),
        "disableflags": int(model.opt.disableflags),
        "multiccd": not bool(
            model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_MULTICCD
        ),
    }
    constant_initialization = {
        "protocol": "cpu_compiled",
        "body_invweight0_before": model.body_invweight0.tolist(),
        "body_invweight0_after": model.body_invweight0.tolist(),
    }
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    kind = asset.config["kind"]
    ids = {b.name: model.body(b.name).id for b in asset.bodies}

    def snapshot(d: Any) -> dict[str, np.ndarray]:
        return {
            b.name: (
                np.array([d.qpos[model.joint(b.joint.name).qposadr[0]]])
                if kind == "gear"
                else d.xpos[ids[b.name]].copy()
            )
            for b in asset.bodies
        }

    initial = snapshot(data)
    steps = round(cfg.duration / cfg.timestep)
    if kind == "gear":
        data.ctrl[:] = 1.0
    if cfg.backend == "mujoco_cpu":
        if on_step is not None:
            on_step(model, data, 0)
        start = time.perf_counter()
        for step in range(steps):
            if kind == "thread":
                dof = model.joint("bolt_joint").dofadr[0]
                data.qvel[dof : dof + 6] = [
                    0,
                    0,
                    data.qvel[dof + 2],
                    0,
                    0,
                    -cfg.spin_rate,
                ]
            mujoco.mj_step(model, data)
            if on_step is not None:
                on_step(model, data, step + 1)
        elapsed = time.perf_counter() - start
        mujoco.mj_kinematics(model, data)
        final = snapshot(data)
        warning_count = sum(int(w.number) for w in data.warning)
        metrics = _score(asset, initial, final, cfg)
        metrics.update(overflow_count=warning_count)
        return {
            "status": (
                "passed"
                if metrics["finite"]
                and not warning_count
                and metrics["success_rate"] == 1
                else "failed"
            ),
            "metrics": metrics,
            "step_time_ms": elapsed * 1000 / steps,
            "timing_scope": "CPU one world; includes prescribed screw speed writes",
            "mujoco_version": mujoco.__version__,
            "effective_solver_options": effective_solver_options,
            "constant_initialization": constant_initialization,
            "initial_state": {name: value.tolist() for name, value in initial.items()},
            "final_state": {name: value.tolist() for name, value in final.items()},
        }
    import mujoco_warp as mjw
    import warp as wp

    if not wp.is_cuda_available():
        raise RuntimeError("Native MuJoCo Warp probe requires CUDA.")
    wp.config.quiet = True
    with wp.ScopedDevice("cuda:0"):
        mw = mjw.put_model(model)
        effective_solver_options = {
            "solver": int(mw.opt.solver),
            "tolerance": float(mw.opt.tolerance.numpy().reshape(-1)[0]),
            "disableflags": int(mw.opt.disableflags),
            "multiccd": not bool(
                mw.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_MULTICCD
            ),
        }
        dw = mjw.put_data(
            model,
            data,
            nworld=cfg.num_envs,
            nconmax=cfg.contact_capacity,
            njmax=6 * cfg.contact_capacity,
        )
        body_weights_before = mw.body_invweight0.numpy().copy()
        if cfg.warp_reference_initialization == "warp_recomputed":
            # Newton's model-property updates use the same public computation.
            # Its degenerate-body fallback can change contact regularization;
            # retain this as an explicit protocol, not a silent baseline change.
            mjw.set_const_0(mw, dw)
        constant_initialization = {
            "protocol": cfg.warp_reference_initialization,
            "body_invweight0_before": body_weights_before.tolist(),
            "body_invweight0_after": mw.body_invweight0.numpy().tolist(),
            "scope": (
                "Imported CPU-compiled constants"
                if cfg.warp_reference_initialization == "cpu_compiled"
                else "Public MuJoCo Warp set_const_0, including its degenerate-body inverse-weight fallback"
            ),
        }
        # First forward loads kernels, without advancing the starting state.
        mjw.forward(mw, dw)
        wp.synchronize()
        start = time.perf_counter()
        for _ in range(steps):
            if kind == "thread":
                from ._warp import constrain_screw_velocity

                wp.launch(
                    constrain_screw_velocity,
                    cfg.num_envs,
                    inputs=[
                        dw.qvel,
                        int(model.joint("bolt_joint").dofadr[0]),
                        cfg.spin_rate,
                    ],
                    device="cuda:0",
                )
            mjw.step(mw, dw)
        wp.synchronize()
        elapsed = time.perf_counter() - start
        overflow = (
            int(np.count_nonzero(dw.overflow.numpy()))
            if hasattr(dw, "overflow")
            else None
        )
        if overflow is None:
            raise RuntimeError(
                "This MuJoCo Warp version does not expose overflow accounting."
            )
        qpos = dw.qpos.numpy()
        qvel = dw.qvel.numpy()
        invalid = int(np.count_nonzero(~np.isfinite(qpos))) + int(
            np.count_nonzero(~np.isfinite(qvel))
        )
        if invalid:
            return {
                "status": "failed",
                "reason": "Native MuJoCo Warp produced nonfinite generalized state.",
                "metrics": {
                    "finite": False,
                    "success_rate": 0.0,
                    "nonfinite_state_count": invalid,
                    "overflow_count": overflow,
                },
                "step_time_ms": elapsed * 1000 / steps / cfg.num_envs,
                "mujoco_version": mujoco.__version__,
                "effective_solver_options": effective_solver_options,
                "constant_initialization": constant_initialization,
            }
        final_rows: dict[str, list[np.ndarray]] = {name: [] for name in ids}
        for world in range(cfg.num_envs):
            mjw.get_data_into(data, model, dw, world_id=world)
            mujoco.mj_kinematics(model, data)
            for name, value in snapshot(data).items():
                final_rows[name].append(value)
        final = {name: np.asarray(rows) for name, rows in final_rows.items()}
        metrics = _score(asset, initial, final, cfg)
        metrics["overflow_count"] = overflow
    return {
        "status": (
            "passed"
            if metrics["finite"] and not overflow and metrics["success_rate"] == 1
            else "failed"
        ),
        "metrics": metrics,
        "step_time_ms": elapsed * 1000 / steps / cfg.num_envs,
        "timing_scope": "GPU batched wall time per world-step; first step compilation may be included",
        "mujoco_version": mujoco.__version__,
        "effective_solver_options": effective_solver_options,
        "constant_initialization": constant_initialization,
        "initial_state": {name: value.tolist() for name, value in initial.items()},
        "final_state": {name: value.tolist() for name, value in final.items()},
    }


def _run_embodichain(
    asset: AssemblyAsset,
    cfg: PrecisionRunCfg,
    *,
    visualization: Any = None,
    on_step: Callable[[Any, dict[str, Any], int], None] | None = None,
    before_step: Callable[[Any, dict[str, Any], int], None] | None = None,
    native_window: bool = False,
    renderer: str = "hybrid",
) -> dict[str, Any]:
    _validate_solver_overrides(cfg)
    import torch
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import (
        DefaultPhysicsCfg,
        DenoisingCfg,
        NewtonPhysicsCfg,
        RenderCfg,
        UsdSourceCfg,
    )

    ground_enabled, ground_attrs = _source_floor_attributes(asset, cfg)
    physics = (
        NewtonPhysicsCfg(
            gravity=asset.source_options.get("gravity", [0, 0, -9.81]),
            num_substeps=1,
            use_cuda_graph=False,
            solver_cfg={
                "solver_type": "mujoco_warp",
                "iterations": cfg.solver_iterations,
                "nconmax": cfg.contact_capacity,
                "njmax": 6 * cfg.contact_capacity,
                "solver": asset.source_options.get("solver", "newton"),
                "integrator": asset.source_options.get("integrator"),
                "use_mujoco_cpu": cfg.use_mujoco_cpu,
                "separate_worlds": False if cfg.use_mujoco_cpu else None,
            },
        )
        if cfg.backend == "newton"
        else DefaultPhysicsCfg(
            gravity=asset.source_options.get("gravity", [0, 0, -9.81])
        )
    )
    if cfg.backend == "newton":
        if cfg.solver is not None:
            physics.solver_cfg["solver"] = cfg.solver
        if cfg.solver_tolerance is not None:
            physics.solver_cfg["tolerance"] = cfg.solver_tolerance
        if cfg.enable_multiccd is not None:
            physics.solver_cfg["enable_multiccd"] = cfg.enable_multiccd
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=not native_window,
            width=1280 if native_window else 1920,
            height=800 if native_window else 1080,
            startup_summary="off",
            enable_entity_gizmo=False,
            num_envs=cfg.num_envs,
            physics_dt=cfg.timestep,
            physics_cfg=physics,
            render_cfg=RenderCfg(
                renderer=renderer if native_window else "no-render",
                spp=16 if native_window else 1,
                denoising=DenoisingCfg(
                    window="optix" if native_window else "off",
                    offscreen="optix" if native_window else "off",
                ),
            ),
            enable_ground_plane=ground_enabled,
            ground_plane_attrs=ground_attrs,
            visualization=visualization,
        )
    )
    objects: dict[str, Any] = {}
    try:
        if cfg.route == "usd":
            path = asset.write_usd(Path(cfg.work_dir) / f"{asset.content_hash}.usda")
            if any(b.joint.kind == "hinge" for b in asset.bodies):
                if cfg.source_contact:
                    raise NotImplementedError(
                        "USD hinge probes cannot yet overlay raw source contact parameters; "
                        "select source_contact=False for an explicit backend-contact comparison."
                    )
                objects = {
                    key.rsplit("/", 1)[-1]: value
                    for key, value in sim.add_usd("assembly", str(path)).items()
                }
            else:
                for body in asset.bodies:
                    body_cfg = body.to_rigid_config(
                        source_contact=cfg.source_contact, backend=cfg.backend
                    )
                    body_cfg.shape = None
                    body_cfg.source = UsdSourceCfg(
                        file_path=str(path), source_prim_path=f"/Assembly/{body.name}"
                    )
                    body_cfg.asset_physics_mode = "overlay"
                    body_cfg.attrs.collision_props.contact_offset = cfg.contact_offset
                    body_cfg.attrs.collision_props.rest_offset = cfg.rest_offset
                    objects[body.name] = sim.add_rigid_object(body_cfg)
        else:
            for body in asset.bodies:
                if body.joint.kind != "hinge":
                    body_cfg = body.to_rigid_config(
                        source_contact=cfg.source_contact,
                        backend=cfg.backend,
                    )
                    if native_window:
                        from ._visual import _VISUAL_FORMAT_VERSION, write_visual_mesh
                        from embodichain.lab.sim.shapes import MeshCfg, ShapePartCfg

                        visual_path = write_visual_mesh(
                            body,
                            Path(cfg.work_dir)
                            / asset.content_hash
                            / f"visual_v{_VISUAL_FORMAT_VERSION}"
                            / f"{body.name}.obj",
                        )
                        body_cfg.shape.visual_parts = [
                            ShapePartCfg(
                                shape=MeshCfg(fpath=str(visual_path.resolve()))
                            )
                        ]
                    body_cfg.attrs.collision_props.contact_offset = cfg.contact_offset
                    body_cfg.attrs.collision_props.rest_offset = cfg.rest_offset
                    objects[body.name] = sim.add_rigid_object(body_cfg)
            for name, art_cfg in asset.to_articulation_configs(
                Path(cfg.work_dir) / asset.content_hash,
                source_contact=cfg.source_contact,
                backend=cfg.backend,
            ).items():
                for group in art_cfg.link_attrs.values():
                    group.attrs.collision_props.contact_offset = cfg.contact_offset
                    group.attrs.collision_props.rest_offset = cfg.rest_offset
                objects[name] = sim.add_articulation(art_cfg)
        sim.prepare()
        collision_counts = {}
        for body in asset.bodies:
            entity = objects[body.name]._entities[0]
            binding = entity._physics_binding
            shape_ids = (
                next(
                    (
                        link.shape_ids
                        for link in getattr(binding, "links", ())
                        if link.name == body.name
                    ),
                    None,
                )
                if body.joint.kind == "hinge"
                else getattr(binding, "shape_ids", None)
            )
            expected = sum(geom.collision for geom in body.geometries)
            actual = None if shape_ids is None else len(shape_ids)
            collision_counts[body.name] = {"expected": expected, "actual": actual}
            if actual is not None and actual != expected:
                raise RuntimeError(
                    f"{body.name}: backend lost collision pieces ({actual} != {expected})."
                )

        def snapshot() -> dict[str, np.ndarray]:
            return {
                name: (
                    obj.get_qpos().detach().cpu().numpy().copy()
                    if asset.config["kind"] == "gear"
                    else obj.get_local_pose()[:, :3].detach().cpu().numpy().copy()
                )
                for name, obj in objects.items()
            }

        initial = snapshot()
        if on_step is not None:
            on_step(sim, objects, 0)
        steps = round(cfg.duration / cfg.timestep)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for step in range(steps):
            if asset.config["kind"] == "thread":
                velocity = objects["bolt"].body_data.lin_vel.clone()
                velocity[:, :2] = 0.0
                angular = torch.zeros_like(velocity)
                angular[:, 2] = -cfg.spin_rate
                objects["bolt"].set_velocity(lin_vel=velocity, ang_vel=angular)
            if asset.config["kind"] == "gear":
                driver = next(b for b in asset.bodies if b.name == "gear_a")
                objects["gear_a"].set_qf(
                    torch.full(
                        (cfg.num_envs, 1), driver.joint.motor_gear, device=sim.device
                    )
                )
            if before_step is not None:
                before_step(sim, objects, step)
            sim.update(step=1)
            if on_step is not None:
                on_step(sim, objects, step + 1)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        final = snapshot()
        metrics = _score(asset, initial, final, cfg)
        # Native capacity health is not exposed by the current Scene facade.
        # Preserve the actual behavior metrics, but do not certify precision.
        metrics["overflow_count"] = None
        return {
            "status": (
                "unverified"
                if metrics["finite"] and metrics["success_rate"] == 1
                else "failed"
            ),
            "reason": (
                "Native contact-overflow accounting is not exposed by the Scene facade; source contact parameters require the documented SDK patch."
                if metrics["finite"] and metrics["success_rate"] == 1
                else "Physical probe failed; native contact-overflow accounting is unavailable."
            ),
            "metrics": metrics,
            "runtime": (
                "Newton/MuJoCo CPU"
                if cfg.backend == "newton" and cfg.use_mujoco_cpu
                else "Newton/MuJoCo Warp" if cfg.backend == "newton" else "Default"
            ),
            "source_solver": asset.source_options.get("solver"),
            "active_solver": (
                cfg.solver or asset.source_options.get("solver")
                if cfg.backend == "newton"
                else None
            ),
            "collision_counts": collision_counts,
            "initial_state": {name: value.tolist() for name, value in initial.items()},
            "final_state": {name: value.tolist() for name, value in final.items()},
            "step_time_ms": elapsed * 1000 / steps / cfg.num_envs,
            "timing_scope": "SimulationManager end-to-end wall time per world-step; includes publication and control",
        }
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


def _source_floor_attributes(
    asset: AssemblyAsset, cfg: PrecisionRunCfg
) -> tuple[bool, Any]:
    """Let the probe host own a ground matching the independent source scene.

    Floors remain outside AssemblyAsset.bodies. Parse their physical metadata
    once during probe preparation; no generator or XML work occurs in a step.
    The current manager ground is an infinite Z=0 plane, matching mjfit's floor.
    """
    import mujoco
    from scipy.spatial.transform import Rotation
    from embodichain.lab.sim.cfg import (
        CollisionPropertiesCfg,
        NewtonCollisionPropertiesCfg,
        NewtonRigidBodyMaterialCfg,
        RigidBodyMaterialCfg,
        RigidBodyPhysicsCfg,
    )

    model = mujoco.MjModel.from_xml_string(asset.source_mjcf)
    world_colliders = np.flatnonzero(
        (model.geom_bodyid == 0)
        & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))
    )
    if len(world_colliders) == 0:
        return False, None
    if len(world_colliders) != 1:
        raise NotImplementedError(
            "Matched assembly probes support one source world floor only."
        )
    index = int(world_colliders[0])
    rotation = Rotation.from_quat(model.geom_quat[index][[1, 2, 3, 0]]).as_matrix()
    if (
        int(model.geom_type[index]) != int(mujoco.mjtGeom.mjGEOM_PLANE)
        or not np.isclose(model.geom_pos[index, 2], 0, atol=1e-12)
        or not np.allclose(rotation[:, 2], [0, 0, 1], atol=1e-12)
        or model.geom_contype[index] != 1
        or model.geom_conaffinity[index] != 1
    ):
        raise NotImplementedError(
            "Source ground must be a standard colliding +Z plane at Z=0."
        )
    if cfg.source_contact and cfg.backend == "default":
        raise NotImplementedError(
            "Default cannot preserve raw source floor solref/solimp; select backend contacts explicitly."
        )
    friction = model.geom_friction[index]
    contact = (
        NewtonCollisionPropertiesCfg(
            contact_offset=cfg.contact_offset,
            rest_offset=cfg.rest_offset,
            condim=int(model.geom_condim[index]),
            priority=int(model.geom_priority[index]),
            solref=model.geom_solref[index].tolist(),
            solimp=model.geom_solimp[index].tolist(),
            solmix=float(model.geom_solmix[index]),
        )
        if cfg.source_contact
        else CollisionPropertiesCfg(
            contact_offset=cfg.contact_offset,
            rest_offset=cfg.rest_offset,
        )
    )
    material = (
        NewtonRigidBodyMaterialCfg(
            dynamic_friction=float(friction[0]),
            torsional_friction=float(friction[1]),
            rolling_friction=float(friction[2]),
        )
        if cfg.backend == "newton"
        else RigidBodyMaterialCfg(
            static_friction=float(friction[0]), dynamic_friction=float(friction[0])
        )
    )
    return True, RigidBodyPhysicsCfg(collision_props=contact, material_props=material)
