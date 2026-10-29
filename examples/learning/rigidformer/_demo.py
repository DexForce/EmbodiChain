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

"""Display Newton sphere dynamics beside a free RigidFormer rollout.

Run: embodichain run-example learning/rigidformer demo --help
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from ._prepare import sha256

if TYPE_CHECKING:
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.sim.objects import RigidObject, VisualObject
    from embodichain.lab.sim.sensors import Camera
    from embodichain_models.dynamics.rigidformer import RigidFormerModel

__all__ = ["run_demo"]

_PHYSICS_HZ = 240
_WARMUP = 4
_WIDTH, _HEIGHT = 960, 540
_OFFSET = (5.0, 0.0, 0.0)
_EYE, _TARGET = (0.15, -10.0, 5.2), (0.15, 0.0, 1.6)
_COLORS = ((0.87, 0.16, 0.09, 1.0), (0.04, 0.50, 0.85, 1.0), (0.96, 0.65, 0.05, 1.0))


@dataclass
class _Geometry:
    scene: str
    object_ids: list[int]
    local_vertices: torch.Tensor
    valid_vertices: torch.Tensor
    anchors: torch.Tensor
    vertex_rank: torch.Tensor
    parameters: torch.Tensor
    radii: torch.Tensor


def _load_geometry(args: argparse.Namespace, config: dict[str, Any]) -> _Geometry:
    """Read committed sphere templates, anchors and physical properties."""
    from embodichain_models.dynamics.rigidformer._native.data.dataset import (
        MultiObjectDynamics,
    )

    data = config["data"]
    dataset = MultiObjectDynamics(
        str(args.data_root),
        sequence_length=_WARMUP,
        stride=999999,
        split="test",
        split_file=str(args.data_root / "splits.json"),
        object_scale_file=str(args.data_root / "object_scales.json"),
        anchor_k=data["anchor_k"],
        anchor_fps_start=data.get("anchor_fps_start", "index0"),
        fps_order=data.get("fps_order", False),
        if_phys_param=True,
        random_step_size={"enable": False},
        augmentation={},
        require_prepared_cache=True,
    )
    selected = next(
        (
            index
            for index, item in enumerate(dataset.all_sequences)
            if (args.scene is None or item["dataset"] == args.scene)
            and len(dataset._mesh_nv_by_dataset[item["dataset"]]) >= 2
        ),
        None,
    )
    if selected is None:
        raise ValueError("The selected test scene must contain at least two objects")
    sample = dataset[selected]
    scene = dataset.all_sequences[selected]["dataset"]
    count = int(sample["valid_obj_mask"].sum())
    ids = args.object_ids
    if ids is None:
        ids = (np.arange(min(3, count)) * count // min(3, count)).tolist()
    if (
        not 2 <= len(ids) <= 3
        or len(set(ids)) != len(ids)
        or any(index < 0 or index >= count for index in ids)
    ):
        raise ValueError("--object-ids must select two or three distinct scene objects")
    origins = torch.from_numpy(
        np.load(args.data_root / scene / "vis_traj.npy")[0, 0, ids, :3].copy()
    ).float()
    mask = sample["valid_vtx_mask"][0, ids]
    vertices = sample["ref_obj_positions"][ids] - origins[:, None]
    vertices = torch.where(mask[..., None], vertices, 0.0)
    radii = vertices.norm(dim=-1).amax(dim=-1)
    return _Geometry(
        scene,
        list(ids),
        vertices,
        mask,
        sample["anchor_idx"][ids].long(),
        sample["vtx_rank"][ids].long(),
        sample["phys_params"][ids],
        radii,
    )


def _read_poses(bodies: list[RigidObject]) -> torch.Tensor:
    """Snapshot public xyz/xyzw poses without retaining engine buffers."""
    return torch.cat([body.get_local_pose() for body in bodies]).detach().clone()


def _vertices(poses: torch.Tensor, geometry: _Geometry) -> torch.Tensor:
    """Move the template vertices with the observed sphere poses."""
    from embodichain.utils.math import matrix_from_quat

    local = geometry.local_vertices.to(poses.device)
    mask = geometry.valid_vertices.to(poses.device)
    moved = torch.einsum("nij,npj->npi", matrix_from_quat(poses[:, 3:]), local)
    return torch.where(mask[..., None], moved + poses[:, None, :3], 0.0)


def _predict(
    model: RigidFormerModel,
    warmup: torch.Tensor,
    geometry: _Geometry,
    args: argparse.Namespace,
) -> torch.Tensor:
    """Infer the entire free rollout from the four engine observations once."""
    import roma

    device = warmup.device
    mask = geometry.valid_vertices.to(device)
    objects = len(geometry.object_ids)
    batch = {
        "obj_positions": warmup[None],
        "valid_obj_mask": torch.ones(1, objects, dtype=torch.bool, device=device),
        "valid_vtx_mask": mask[None, None].expand(1, _WARMUP, -1, -1),
        "ref_obj_positions": warmup[0][None],
        "anchor_idx": geometry.anchors[None].to(device),
        "vtx_rank": geometry.vertex_rank[None].to(device),
        "phys_params": geometry.parameters[None].to(device),
        "skip_step_size": torch.tensor([float(args.stride)], device=device),
        "dt": torch.ones(1, device=device),
        "time_step_id": torch.zeros(1, dtype=torch.long, device=device),
    }
    with torch.inference_mode():
        predicted = model.rollout(batch, num_frames=args.frames, warmup_frames=_WARMUP)[
            "obj_positions"
        ][0]
        if not torch.isfinite(predicted).all():
            raise RuntimeError("RigidFormer produced a non-finite rollout")
        # Registration recovers body origins even when the vertex centroid is
        # offset. Sphere orientation does not affect its rendered appearance.
        _, centers = roma.rigid_points_registration(
            geometry.local_vertices.to(device)[None].expand_as(predicted),
            predicted,
            weights=mask[None].expand(predicted.shape[:-1]).float(),
        )
    return centers


def _set_prediction(visuals: list[VisualObject], centers: torch.Tensor) -> None:
    """Place render-only spheres beside the dynamic scene."""
    poses = torch.zeros(len(visuals), 7, device=centers.device)
    poses[:, :3] = centers + centers.new_tensor(_OFFSET)
    poses[:, 6] = 1
    for index in range(len(visuals)):
        visuals[index].set_local_pose(poses[index : index + 1])


def _capture(
    sim: SimulationManager, camera: Camera, frame: int, stride: int
) -> np.ndarray:
    """Read a synchronized engine image and label its two sides."""
    import cv2

    sim.sync_render_state()
    camera.update()
    rgb = camera.get_data()["color"][0, ..., :3].cpu().numpy().copy()
    if np.issubdtype(rgb.dtype, np.floating):
        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    cv2.rectangle(rgb, (0, 0), (_WIDTH, 44), (16, 22, 30), -1)
    cv2.putText(
        rgb,
        "NEWTON SIMULATION",
        (90, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        rgb,
        "RIGIDFORMER PREDICTION",
        (555, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        1,
    )
    phase = "observed warm-up" if frame < _WARMUP else "free model rollout"
    cv2.rectangle(rgb, (0, _HEIGHT - 32), (_WIDTH, _HEIGHT), (16, 22, 30), -1)
    cv2.putText(
        rgb,
        f"t={frame * stride / _PHYSICS_HZ:.3f}s | {phase}",
        (16, _HEIGHT - 11),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    return rgb


def run_demo(args: argparse.Namespace) -> None:
    """Run the engine comparison and save a video and aligned trajectories.

    Args:
        args: Parsed ``learning/rigidformer demo`` command-line options.
    """
    import imageio.v2 as imageio
    from embodichain.lab.sim import (
        SimulationManager,
        SimulationManagerCfg,
        VisualMaterialCfg,
    )
    from embodichain.lab.sim.cfg import (
        LightCfg,
        MassPropertiesCfg,
        NewtonPhysicsCfg,
        NewtonRigidBodyMaterialCfg,
        RenderCfg,
        RigidBodyPhysicsCfg,
        RigidObjectCfg,
        VisualObjectCfg,
    )
    from embodichain.lab.sim.sensors import CameraCfg
    from embodichain.lab.sim.shapes import SphereCfg
    from embodichain.lab.visualization import VisualizationCfg
    from embodichain_models.dynamics.rigidformer import load_rigidformer_checkpoint
    from embodichain_models.dynamics.rigidformer.data import _verified_manifest

    if not _WARMUP < args.frames <= 480 or args.stride < 1 or args.fps < 1:
        raise ValueError("frames must be in [5, 480]; stride and fps must be positive")
    if not np.isfinite(args.hold_seconds) or args.hold_seconds < 0:
        raise ValueError("hold-seconds must be finite and nonnegative")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("The engine/RigidFormer demo requires a CUDA device")
    device = torch.device("cuda", device.index if device.index is not None else 0)
    torch.cuda.set_device(device)
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.data_root.is_dir() or not args.checkpoint.is_file():
        raise FileNotFoundError(
            "Prepare MOVi-Sphere data and weights first: embodichain run-example learning/rigidformer "
            "prepare --dataset movi_sphere --assets all --limit-per-split 1; "
            "or pass --data-root and --checkpoint"
        )
    manifest = _verified_manifest(args.data_root)
    if manifest.get("dataset") != "movi_sphere":
        raise ValueError("This sphere demo requires prepared MOVi-Sphere geometry")
    model = load_rigidformer_checkpoint(
        args.checkpoint, device=device, attention_backend=args.attention_backend
    )
    geometry = _load_geometry(args, model.reference_config)
    args.output.mkdir(parents=True)
    sim = None
    writer = None
    camera = None
    window = None
    bodies: list[RigidObject] = []
    visuals: list[VisualObject] = []
    try:
        sim = SimulationManager(
            SimulationManagerCfg(
                headless=True,
                device=str(device),
                gpu_id=device.index,
                width=_WIDTH,
                height=_HEIGHT,
                render_cfg=RenderCfg(renderer="hybrid"),
                visualization=VisualizationCfg(backend="none"),
                physics_cfg=NewtonPhysicsCfg(
                    physics_dt=1 / _PHYSICS_HZ,
                    num_substeps=2,
                    solver_cfg={"solver_type": "xpbd"},
                ),
            )
        )
        sim.add_light(
            LightCfg(uid="demo_light", intensity=100, radius=30, init_pos=(0, -4, 9))
        )
        count = len(geometry.object_ids)
        starts = torch.tensor(
            [[-3.35, -0.2, 2.0], [-1.35, -0.2, 2.0], [-2.35, 0.6, 3.6]]
        )[:count]
        velocities = torch.tensor([[1.4, 0, 0], [-1.4, 0, 0], [0, -0.2, -1.0]])[:count]
        for index in range(count):
            physical = geometry.parameters[index]
            material = VisualMaterialCfg(base_color=list(_COLORS[index]), roughness=0.4)
            shape = SphereCfg(
                radius=float(geometry.radii[index]), visual_material=material
            )
            bodies.append(
                sim.add_rigid_object(
                    RigidObjectCfg(
                        uid=f"simulation_ball_{index}",
                        shape=shape,
                        init_pos=starts[index].tolist(),
                        body_type="dynamic",
                        attrs=RigidBodyPhysicsCfg(
                            mass_props=MassPropertiesCfg(mass=float(physical[0])),
                            material_props=NewtonRigidBodyMaterialCfg(
                                dynamic_friction=float(physical[1]),
                                restitution=float(physical[2]),
                            ),
                        ),
                    )
                )
            )
            visuals.append(
                sim.add_visual_object(
                    VisualObjectCfg(
                        uid=f"prediction_ball_{index}",
                        shape=shape,
                        init_pos=(starts[index] + starts.new_tensor(_OFFSET)).tolist(),
                    )
                )
            )
        camera = sim.add_sensor(
            CameraCfg(
                uid="demo_camera",
                width=_WIDTH,
                height=_HEIGHT,
                intrinsics=(850, 850, _WIDTH / 2, _HEIGHT / 2),
                extrinsics=CameraCfg.ExtrinsicsCfg(eye=_EYE, target=_TARGET),
            )
        )
        sim.prepare()
        for index in range(count):
            bodies[index].set_velocity(lin_vel=velocities[index : index + 1].to(device))
        if not args.headless:
            if not sim.open_window():
                raise RuntimeError(
                    "Unable to open the engine window; use --headless to record"
                )
            window = sim.get_world().get_windows()
            window.set_look_at(
                eye=np.array(_EYE), look_at=np.array(_TARGET), up=np.array([0, 0, 1])
            )
            window.set_window_title("Newton (left) | RigidFormer (right)")
        writer = imageio.get_writer(
            str(args.output / "rollout.mp4"),
            fps=args.fps,
            codec="libx264",
            macro_block_size=None,
        )
        observed, warmup = [], []
        predicted = None
        print("Left: Newton dynamics. Right: RigidFormer after four observed frames.")
        for frame in range(args.frames):
            tick = time.perf_counter()
            if frame:
                # Publish only the sampled state after its matching model pose
                # is written; intermediate physics steps remain unrendered.
                for _ in range(args.stride):
                    sim.update(step=1, render_final_step=False)
            poses = _read_poses(bodies)
            observed.append(poses[:, :3])
            if frame < _WARMUP:
                warmup.append(_vertices(poses, geometry))
                _set_prediction(visuals, poses[:, :3])
            else:
                assert predicted is not None
                _set_prediction(visuals, predicted[frame])
            image = _capture(sim, camera, frame, args.stride)
            writer.append_data(image)
            if frame == min(20, args.frames - 1):
                imageio.imwrite(args.output / "frame.png", image)
            if frame == _WARMUP - 1:
                predicted = _predict(model, torch.stack(warmup), geometry, args)
            if not args.headless:
                time.sleep(max(0.0, 1 / args.fps - (time.perf_counter() - tick)))
        writer.close()
        writer = None
        assert predicted is not None
        simulated = torch.stack(observed).cpu().numpy()
        predicted_centers = predicted.cpu().numpy()
        errors = np.linalg.norm(simulated - predicted_centers, axis=-1)
        np.savez_compressed(
            args.output / "trajectory.npz",
            simulated_centers=simulated,
            predicted_centers=predicted_centers,
            center_error_m=errors,
        )
        summary = {
            "physics_backend": "newton",
            "solver": "xpbd",
            "physics_step_seconds": 1 / _PHYSICS_HZ,
            "scene": geometry.scene,
            "object_ids": geometry.object_ids,
            "frames": args.frames,
            "warmup_frames": _WARMUP,
            "model_stride": args.stride,
            "playback_fps": args.fps,
            "attention_backend": args.attention_backend,
            "checkpoint_sha256": sha256(args.checkpoint),
            "data_manifest_sha256": sha256(args.data_root / "manifest.json"),
            "dataset_scope": manifest["scope"],
            "model_rollout_calls": 1,
            "sphere_radii_m": geometry.radii.tolist(),
            "mass_friction_restitution": geometry.parameters.tolist(),
            "initial_positions_m": starts.tolist(),
            "initial_velocities_m_per_s": velocities.tolist(),
            "elapsed_simulation_seconds": (args.frames - 1) * args.stride / _PHYSICS_HZ,
            "mean_center_error_at_last_frame_m": float(errors[-1].mean()),
            "max_center_error_m": float(errors.max()),
        }
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        print(f"Saved engine video: {args.output / 'rollout.mp4'}")
        if not args.headless:
            time.sleep(args.hold_seconds)
    except Exception as error:
        (args.output / "failure.json").write_text(
            json.dumps({"type": type(error).__name__, "message": str(error)}) + "\n"
        )
        raise
    finally:
        try:
            if writer is not None:
                writer.close()
        finally:
            if sim is not None:
                sim.destroy(exit_process=False)
            bodies.clear()
            visuals.clear()
            camera = None
            window = None
            SimulationManager.flush_cleanup_queue()
