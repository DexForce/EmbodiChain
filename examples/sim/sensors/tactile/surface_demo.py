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

"""Read and visualize completed Newton hydroelastic faces on a tactile pad."""

from __future__ import annotations

from argparse import Namespace

import matplotlib
import numpy as np
import torch
from matplotlib.figure import Figure

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    NewtonCollisionPipelineCfg,
    NewtonPhysicsCfg,
    RigidObjectCfg,
)
from embodichain.lab.sim.sensors import SensorCfg
from embodichain.lab.sim.shapes import MeshCfg, MeshCollisionCfg

from .._scene_camera import SCENE_RENDER, capture_rgb, scene_camera, show_scene

_PAD_SIZE = (0.10, 0.10, 0.10)
_TOOL_SIZE = (0.05, 0.045, 0.06)
_PAD_TOP = 0.50
_REACH = 0.03

Sample = dict[str, np.ndarray]


def collect_samples(args: Namespace) -> list[Sample]:
    """Generate completed physical and geometric samples from the same step."""
    from dexsim.scene import Scene
    from newton.geometry import HydroelasticSDF
    import trimesh

    if args.device.split(":")[0] != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The completed-surface example requires --device cuda:0.")
    if not hasattr(Scene, "get_hydroelastic_contact_capabilities"):
        raise RuntimeError("Install a DexSim SDK with completed hydroelastic surfaces.")

    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            startup_summary="off",
            enable_ground_plane=False,
            render_cfg=SCENE_RENDER,
            enable_entity_gizmo=False,
            width=720,
            height=540,
            physics_cfg=NewtonPhysicsCfg(
                device=args.device,
                gravity=(0.0, 0.0, 0.0),
                num_substeps=3,
                use_cuda_graph=True,
                solver_cfg={
                    "solver_type": "mujoco_warp",
                    "use_mujoco_contacts": False,
                    "nconmax": 256,
                },
                collision_cfg=NewtonCollisionPipelineCfg(
                    update_interval=1,
                    max_triangle_pairs=16384,
                    sdf_hydroelastic_config=HydroelasticSDF.Config(
                        output_contact_surface=True, pre_prune_contacts=True
                    ),
                ),
            ),
        )
    )
    try:
        for uid, body_type, size, z in (
            ("pad_body", "static", _PAD_SIZE, _PAD_TOP - _PAD_SIZE[2] / 2),
            ("tool", "dynamic", _TOOL_SIZE, _PAD_TOP + _TOOL_SIZE[2] / 2 + 0.02),
        ):
            mesh = trimesh.creation.box(extents=size)
            sim.add_rigid_object(
                RigidObjectCfg(
                    uid=uid,
                    body_type=body_type,
                    init_pos=(0.0, 0.0, z),
                    shape=MeshCfg(
                        vertices=mesh.vertices,
                        triangles=mesh.faces,
                        collision=MeshCollisionCfg(
                            approximation="sdf", is_hydroelastic=True, sdf_resolution=32
                        ),
                    ),
                )
            )
        tool = sim.get_rigid_object("tool")
        sensor = sim.add_sensor(
            SensorCfg.from_dict(
                {
                    "sensor_type": "TactileIndentation",
                    "uid": "pad",
                    "target_uids": ["tool"],
                    "mount": {
                        "parent": "pad_body",
                        "pos": [0.0, 0.0, _PAD_SIZE[2] / 2],
                    },
                    "size": _PAD_SIZE[:2],
                    "resolution": (24, 24),
                    "max_depth": _REACH,
                    "surface": {"max_distance": _REACH},
                }
            )
        )
        camera = scene_camera(
            sim,
            (0.145, -0.235, 0.60),
            (0.0, 0.0, 0.49),
            {"pad_body": (0.09, 0.34, 0.43), "tool": (0.96, 0.57, 0.19)},
        )
        poses = torch.zeros((1, 7), device=args.device)
        poses[:, 6] = 1.0
        zero_velocity = torch.zeros((1, 3), device=args.device)
        samples = []
        phases = (0.0, 0.25, 0.50, 0.75, 1.0)
        for phase in np.linspace(0.0, 1.0, args.frames):
            poses[:, 0] = np.interp(
                phase, phases, (-0.018, -0.018, 0.018, -0.018, -0.018)
            )
            separation = np.interp(
                phase, phases, (0.018, -0.013, -0.013, -0.013, 0.018)
            )
            poses[:, 2] = _PAD_TOP + _TOOL_SIZE[2] / 2 + separation
            tool.set_local_pose(poses)
            tool.set_velocity(zero_velocity, zero_velocity)
            sim.update(step=1, render_final_step=False)
            data = sensor.get_data()
            metadata = sensor.get_sample_metadata()
            if metadata is None or not metadata["valid"][0].item():
                raise RuntimeError("No completed tactile sample.")
            sample = {
                name: value[0].detach().cpu().numpy().copy()
                for source in (data, metadata)
                for name, value in source.items()
            }
            sample["camera_rgb"] = capture_rgb(sim, camera)
            samples.append(sample)
        return samples
    finally:
        sim.destroy(exit_process=False)


def draw_sample(
    figure: Figure, samples: list[Sample], index: int, args: Namespace
) -> None:
    """Draw scene, depth and the status of the aligned contact snapshot."""
    sample = samples[index]
    figure.clear()
    figure.set_layout_engine("constrained")
    scene, depth, faces, separation = figure.subplots(
        1, 4, gridspec_kw={"width_ratios": (1.7, 1, 1, 1)}
    )
    show_scene(scene, sample["camera_rgb"])

    complete = bool(sample["surface_complete"])
    face_count = sample["surface_face_count"]
    maximum = max(1, *(int(frame["surface_face_count"].max()) for frame in samples))
    empty = (face_count == 0) if complete else np.ones_like(face_count, dtype=bool)
    panels = (
        (
            depth,
            "Geometric depth",
            "Depth (m)",
            np.ma.masked_where(~sample["hit"], sample["depth"]),
            "viridis",
            0,
            _REACH,
        ),
        (
            faces,
            "Face count",
            "Faces / cell",
            np.ma.masked_where(empty, face_count),
            "viridis",
            0,
            maximum,
        ),
        (
            separation,
            "Signed separation",
            "Separation (m)",
            np.ma.masked_where(empty, sample["surface_separation"]),
            "coolwarm",
            -_REACH,
            _REACH,
        ),
    )
    for axis, title, label, values, palette, low, high in panels:
        cmap = matplotlib.colormaps[palette].with_extremes(bad="#e6e8eb")
        image = axis.imshow(
            values,
            origin="lower",
            extent=(-0.05, 0.05, -0.05, 0.05),
            cmap=cmap,
            vmin=low,
            vmax=high,
            interpolation="nearest",
        )
        axis.set(title=title, xlabel="Pad X (m)", ylabel="Pad Y (m)")
        figure.colorbar(image, ax=axis, label=label, shrink=0.65)

    status = (
        ("Complete · faces present" if face_count.any() else "Complete · no faces")
        if complete
        else "Missing or truncated"
    )
    figure.supxlabel(
        f"Sample {sample['sample_id']}  ·  {sample['time']:.3f} s  ·  "
        f"Generation {sample['surface_generation']}  ·  Substep {sample['surface_substep']}  ·  {status}",
        fontsize=9,
    )
