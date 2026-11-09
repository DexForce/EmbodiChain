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

"""Run isolated, reproducible native acceptance of USD import recipes.

A JSON manifest contains a list of cases with ``id`` and ``file_path``. Optional
fields are ``category``, ``render``, ``num_envs`` and ``steps``. Results retain
source hashes, runtime versions, per-stage outcomes and logs. Every case runs
in its own process so native resource lifetimes cannot mask another case.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _accept_case(case: dict[str, Any], output: Path) -> dict[str, Any]:
    import importlib.metadata
    import numpy as np
    import torch
    from pxr import Usd, UsdGeom
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import (
        DefaultPhysicsCfg,
        NewtonPhysicsCfg,
        RenderCfg,
        DenoisingCfg,
        UsdImportCfg,
        UsdSourceCfg,
        LightCfg,
        VolumeDeformableObjectCfg,
        SurfaceDeformableObjectCfg,
        RigidObjectCfg,
        VisualObjectCfg,
        ArticulationCfg,
    )
    from embodichain.lab.sim.sensors import CameraCfg
    from embodichain.lab.sim.spawn.usd import _parse_usd_source
    from embodichain.lab.sim.spawn import usd_import
    import dexsim

    path = Path(case["file_path"]).expanduser().resolve()
    record: dict[str, Any] = {
        "id": case["id"],
        "category": case.get("category"),
        "file_path": str(path),
        "source_url": case.get("source_url"),
        "sha256": _sha256(path),
        "stages": {},
        "python": sys.executable,
        "dexsim": importlib.metadata.version("dexsim_engine"),
        "dexsim_module": dexsim.__file__,
        "acceptance_script_sha256": _sha256(Path(__file__)),
    }
    record["loader_sources_sha256"] = {
        module_name: _sha256(Path(sys.modules[module_name].__file__))
        for module_name in (
            "embodichain.lab.sim.sim_manager",
            "embodichain.lab.sim.cfg.usd",
            "embodichain.lab.sim.cfg.source",
            "embodichain.lab.sim.cfg.asset",
            "embodichain.lab.sim.cfg.visual_object",
            "embodichain.lab.sim.cfg.deformable",
            "embodichain.lab.sim.cfg.physics",
            "embodichain.lab.sim.spawn.usd",
            "embodichain.lab.sim.spawn.usd_import",
            "embodichain.lab.sim.spawn.usd_result",
            "embodichain.lab.sim.spawn.scene",
        )
    }
    try:
        record["embodichain_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        record["embodichain_commit"] = "unknown"
    sim = None
    imported = camera = asset = None
    stage_name = "parse"
    started = time.monotonic()
    try:
        source_cfg = UsdSourceCfg(
            file_path=str(path), root_prim_path=case.get("root_prim_path")
        )
        raw = _parse_usd_source(source_cfg)
        entities = list(raw.iter_entities())
        types: dict[str, int] = {}
        for item in entities:
            types[type(item).__name__] = types.get(type(item).__name__, 0) + 1
        record["descriptors"] = types
        if raw.gaussian_splats:
            raise ValueError("Manifest includes excluded 3DGS content.")
        if not entities:
            raise ValueError("Source contains no importable scene entities.")
        record["stages"][stage_name] = "passed"
        stage_name = "initialize"
        newton = (
            bool(raw.particle_sets or raw.cables) or case.get("physics") == "newton"
        )
        device = "cuda:0" if newton else "cpu"
        physics = (
            NewtonPhysicsCfg(device=device, use_cuda_graph=False, collision_cfg=None)
            if newton
            else DefaultPhysicsCfg(device=device)
        )
        render = case.get("render", False)
        if (
            render
            and sys.platform == "linux"
            and not any(
                os.environ.get(key)
                for key in ("DISPLAY", "WAYLAND_DISPLAY", "SDL_VIDEODRIVER")
            )
        ):
            raise RuntimeError(
                "Native offscreen rendering requires a configured display or SDL video driver."
            )
        sim = SimulationManager(
            SimulationManagerCfg(
                headless=True,
                device=device,
                physics_cfg=physics,
                num_envs=case.get("num_envs", 1),
                scene_node_capacity=max(512, len(entities) * 24),
                render_cfg=RenderCfg(
                    renderer="hybrid" if render else "no-render",
                    denoising=DenoisingCfg(window="off", offscreen="off"),
                ),
            )
        )
        record["physics"] = "newton" if newton else "default"
        record["device"] = device
        record["num_envs"] = sim.num_envs
        record["stages"][stage_name] = "passed"
        stage_name = "declare"
        entry_point = case.get("entry_point", "scene")
        if entry_point == "scene":
            imported = sim.add_usd(UsdImportCfg(name=case["id"], source=source_cfg))
            if len(imported.resource_uids) != len(entities):
                raise AssertionError(
                    "An imported entity was lost or a source path is duplicated."
                )
            expected_assets = (
                len(raw.mesh_objects) + len(raw.articulations) + len(raw.particle_sets)
            )
            record["resources"] = dict(imported.resource_uids)
            record["constraints"] = dict(imported.constraint_uids)
        else:
            config_type, add, collection = {
                "volume": (
                    VolumeDeformableObjectCfg,
                    sim.add_deformable_object,
                    raw.particle_sets,
                ),
                "surface": (
                    SurfaceDeformableObjectCfg,
                    sim.add_deformable_object,
                    raw.particle_sets,
                ),
                "rigid": (RigidObjectCfg, sim.add_rigid_object, raw.mesh_objects),
                "visual": (VisualObjectCfg, sim.add_visual_object, raw.mesh_objects),
                "articulation": (
                    ArticulationCfg,
                    sim.add_articulation,
                    raw.articulations,
                ),
            }[entry_point]
            if len(collection) != 1:
                raise ValueError("Typed acceptance requires a singleton source.")
            selected = collection[0].usd.prim_path
            source_cfg.source_prim_path = selected
            asset_cfg = config_type(uid="typed_asset", source=source_cfg)
            if entry_point == "articulation":
                asset_cfg.build_pk_chain = False
            asset = add(asset_cfg)
            imported = {selected: asset}
            expected_assets = 1
            record["resources"] = {selected: asset.uid}
            record["constraints"] = {}
        record["entry_point"] = entry_point
        if len(imported) != expected_assets:
            raise AssertionError(
                f"Expected {expected_assets} EmbodiChain assets, got {len(imported)}."
            )
        record["assets"] = {
            key: {"uid": value.uid, "type": type(value).__name__}
            for key, value in imported.items()
        }
        if render:
            stage = Usd.Stage.Open(str(path))
            box = (
                UsdGeom.BBoxCache(
                    Usd.TimeCode.Default(),
                    [
                        UsdGeom.Tokens.default_,
                        UsdGeom.Tokens.render,
                        UsdGeom.Tokens.proxy,
                        UsdGeom.Tokens.guide,
                    ],
                    False,
                    True,
                )
                .ComputeWorldBound(stage.GetPseudoRoot())
                .ComputeAlignedRange()
            )
            lo, hi = np.array(box.GetMin()), np.array(box.GetMax())
            lo, hi = lo * UsdGeom.GetStageMetersPerUnit(
                stage
            ), hi * UsdGeom.GetStageMetersPerUnit(stage)
            center = (lo + hi) * 0.5
            if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y:
                center = np.array((center[0], -center[2], center[1]))
            radius = float(np.linalg.norm(hi - lo) * 0.5)
            if box.IsEmpty() or not np.isfinite(radius) or radius < 0.01:
                center, radius = np.zeros(3), 1.0
            eye = center + np.array((1.25, -1.65, 1.05)) * radius
            camera = sim.add_sensor(
                CameraCfg(
                    uid="acceptance_camera",
                    width=320,
                    height=240,
                    near=min(0.005, max(0.0001, radius * 0.001)),
                    far=max(100.0, radius * 30),
                    intrinsics=(280.0, 280.0, 160.0, 120.0),
                    extrinsics=CameraCfg.ExtrinsicsCfg(
                        eye=tuple(eye), target=tuple(center), up=(0, 0, 1)
                    ),
                )
            )
            sim.add_light(
                LightCfg(uid="acceptance_light", light_type="direction", intensity=8.0)
            )
            record["camera"] = {"eye": eye.tolist(), "target": center.tolist()}
        record["stages"][stage_name] = "passed"
        stage_name = "prepare"
        sim.prepare()
        for key, asset in imported.items():
            if type(asset).__name__ != "VisualObject" or render:
                if not asset.is_spawn_bound:
                    raise AssertionError(f"Facade {key} did not bind.")
            if asset.num_instances != sim.num_envs:
                raise AssertionError(f"Facade {key} lost replication.")
        record["stages"][stage_name] = "passed"

        def check_state() -> None:
            for asset in imported.values():
                if not asset.is_spawn_bound:
                    continue
                if type(asset).__name__ in (
                    "VolumeDeformableObject",
                    "SurfaceDeformableObject",
                ):
                    value = asset.data.nodal_pos_w
                    if value.numel() == 0:
                        raise AssertionError(
                            f"Deformable {asset.uid} has no simulation nodes."
                        )
                elif type(asset).__name__ in ("Articulation", "Robot"):
                    value = asset.get_qpos()
                else:
                    value = asset.get_local_pose()
                if not torch.isfinite(value).all():
                    raise AssertionError(f"Nonfinite state in {asset.uid}.")
                record.setdefault("state_shapes", {})[asset.uid] = list(value.shape)

        check_state()
        if render:
            stage_name = "render"
            sim.set_ground_plane_visibility(False)
            sim.sync_render_state()
            for _ in range(8):
                camera.update()
            pixels = camera.get_data()["color"][0, :, :, :3].cpu().numpy()
            from PIL import Image

            Image.fromarray(pixels).save(output / "overview.png")
            record["image"] = {
                "path": "overview.png",
                "std": float(pixels.std()),
                "sha256": _sha256(output / "overview.png"),
            }
            if pixels.std() < 0.5:
                raise AssertionError("Offscreen image is uniform.")
            record["stages"][stage_name] = "passed"
        stage_name = "step"
        start_time = sim.simulation_time
        for _ in range(case.get("steps", 3)):
            sim.update()
        check_state()
        if sim.simulation_time <= start_time:
            raise AssertionError("Physics time did not advance.")
        record["simulation_time"] = sim.simulation_time
        record["stages"][stage_name] = "passed"
        stage_name = "reset"
        sim.reset_objects_state()
        check_state()
        record["stages"][stage_name] = "passed"
        record["status"] = "passed"
    except Exception as error:
        record["status"] = "failed"
        record["stages"][stage_name] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        record["traceback"] = traceback.format_exc()
    finally:
        if sim is not None:
            try:
                sim.destroy(exit_process=False)
                imported = camera = asset = None
                SimulationManager.flush_cleanup_queue()
            except Exception as error:
                record["teardown_error"] = f"{type(error).__name__}: {error}"
                record["status"] = "failed"
        record["entry_unchanged"] = _sha256(path) == record["sha256"]
        record["seconds"] = time.monotonic() - started
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--case", type=Path)
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.case:
        record = _accept_case(json.loads(args.case.read_text()), args.output)
        (args.output / "result.json").write_text(json.dumps(record, indent=2))
        print(
            json.dumps(
                {
                    "id": record["id"],
                    "status": record["status"],
                    "error": record.get("error"),
                }
            )
        )
        return
    cases = json.loads(args.manifest.read_text())
    results = []
    for case in cases:
        directory = args.output / case["id"]
        directory.mkdir(exist_ok=True)
        job = directory / "case.json"
        job.write_text(json.dumps(case, indent=2))
        (directory / "result.json").unlink(missing_ok=True)
        try:
            with (directory / "runtime.log").open("w") as log:
                process = subprocess.run(
                    [
                        sys.executable,
                        __file__,
                        "--case",
                        str(job),
                        "--output",
                        str(directory),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                )
            if (directory / "result.json").exists():
                result = json.loads((directory / "result.json").read_text())
                if process.returncode:
                    result["status"] = "failed"
                    result["process_exit"] = process.returncode
            else:
                result = {
                    **case,
                    "status": "failed",
                    "process_exit": process.returncode,
                    "error": "Native process exited without a result; see runtime.log.",
                }
        except subprocess.TimeoutExpired:
            result = {
                **case,
                "status": "failed",
                "error": "Native acceptance timed out.",
            }
        results.append(result)
        (args.output / "results.json").write_text(json.dumps(results, indent=2))
        print(case["id"], result["status"], result.get("error", ""), flush=True)
    print(
        f"{sum(x['status']=='passed' for x in results)}/{len(results)} passed",
        flush=True,
    )
    if any(item["status"] != "passed" for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
