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
"""Build accepted reference C02 through EmbodiChain's public scene API.

Run with the recorded DexSim/Newton PYTHONPATH; default is a full 64 s video.
"""

from __future__ import annotations
import argparse
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any
import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[7]))
import dexsim
import newton
import warp as wp
from dexsim.engine.newton_physics import RealtimeKinematicJointControl
from dexsim.engine.newton_physics.runtime_control.base import GraphRuntimeControl
from dexsim.engine.newton_physics.backend_registry import get_newton_backend
from embodichain.utils import configclass
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    ArticulationCfg,
    ArticulationRootPropertiesCfg,
    LinkPhysicsOverrideCfg,
    MassPropertiesCfg,
    NewtonCollisionPropertiesCfg,
    NewtonRigidBodyMaterialCfg,
    NewtonPhysicsCfg,
    RenderCfg,
    RigidBodyPhysicsCfg,
    RigidObjectCfg,
    SurfaceDeformableObjectCfg,
    SurfaceDeformablePhysicsCfg,
    UsdSourceCfg,
    VisualObjectCfg,
    LightCfg,
)
from embodichain.lab.sim.material import VisualMaterialCfg
from embodichain.lab.sim.sensors import CameraCfg
from embodichain.lab.sim.shapes import CubeCfg, MeshCfg
from scripts.benchmark.tech_reports.physics.experiments.dexuni.c02_controller import (
    ReferenceController,
)
from scripts.benchmark.tech_reports.physics.experiments.dexuni.c02_replay import (
    OUTPUT,
    load,
    freeze,
    record,
    table_pieces,
    seed_paper,
    audit_model,
)

__all__ = ["main"]


@configclass
class ReferenceHandCollisionCfg(NewtonCollisionPropertiesCfg):
    """Benchmark-only forwarding of the original per-link SDF resolution.

    The public typed collision extension passes these native descriptor fields
    through unchanged; no SDK or production configuration class is modified.
    """

    sdf_target_voxel_size: float = 0.001
    sdf_padding: float = 0.05


class PaperInitialState(GraphRuntimeControl):
    """Seed authored heterogeneous rest data once, with no substep action."""

    def __init__(self, inputs: dict[str, Any]) -> None:
        self.inputs = inputs

    def initialize_model(self, model: Any, result: Any) -> None:
        seed_paper(model, self.inputs)

    def __call__(
        self, context: Any, substep_index: int, substep_count: int, substep_dt: float
    ) -> None:
        pass


def _can_usd(path: Path) -> None:
    """Author an analytic cylinder for EC's public USD-source primitive path."""
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Gf, Sdf

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    prim = UsdGeom.Cylinder.Define(stage, "/can")
    prim.CreateRadiusAttr(0.0325)
    prim.CreateHeightAttr(0.118)
    prim.CreateAxisAttr("Z")
    UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
    material = UsdShade.Material.Define(stage, "/Looks/can")
    s = UsdShade.Shader.Define(stage, "/Looks/can/shader")
    s.CreateIdAttr("UsdPreviewSurface")
    s.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.85, 0.34, 0.1)
    )
    s.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    material.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()


def main() -> None:
    """Declare the exact scene, seed original rest data, and run a real replay."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, default=OUTPUT)
    p.add_argument("--out", type=Path)
    p.add_argument("--until", type=float)
    p.add_argument("--no-video", action="store_true")
    args = p.parse_args()
    inputs = load(args.reference)
    out = args.out or OUTPUT / "platforms/embodichain"
    freeze(inputs, out, Path(__file__))
    joints = inputs["joints"]
    mesh = inputs["bag"]
    mat = inputs["materials"]
    ref = out / "source/reference"
    sys.path.insert(0, str(ref))
    spec = importlib.util.spec_from_file_location(
        "c02_original_bag_demo", ref / "demo_dexuni_w1_bag_packing.py"
    )
    demo = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = demo
    spec.loader.exec_module(demo)
    urdf = out / "inputs/w1_v030/robot.urdf"
    hands = demo.hand_links(urdf)
    original = demo.solver_config(
        SimpleNamespace(device="cuda:0", disable_cuda_graph=False, w1_urdf=urdf)
    ).solver_cfg
    solver = dict(
        solver_type="dexuni",
        mujoco_articulation_paths=("arena_0/w1",),
        joint_mode="kinematic",
        contact_mode="full",
        coupling="one_way",
        full_surface_contact_targets={
            "arena_0/w1": hands,
            **{f"arena_0/{n}": None for n in ("table", "can", "carton")},
        },
        vbd_options=original.vbd_options,
        collision_options=original.collision_options,
    )
    sim = None
    controller = ReferenceController(out.resolve())
    try:
        sim = SimulationManager(
            SimulationManagerCfg(
                num_envs=1,
                headless=True,
                device="cuda:0",
                width=1280,
                height=720,
                enable_entity_gizmo=False,
                robot_ik_gizmo=None,
                render_cfg=RenderCfg(
                    renderer="no-render" if args.no_video else "rt", spp=1
                ),
                physics_cfg=NewtonPhysicsCfg(
                    device="cuda:0",
                    physics_dt=1 / 60,
                    num_substeps=6,
                    use_cuda_graph=True,
                    collision_cfg=None,
                    solver_cfg=solver,
                ),
            )
        )

        def attrs(
            mu: float = 0.65, margin: float = 0.0, density: float | None = None
        ) -> RigidBodyPhysicsCfg:
            return RigidBodyPhysicsCfg(
                collision_props=(
                    ReferenceHandCollisionCfg
                    if margin
                    else NewtonCollisionPropertiesCfg
                )(margin=margin, gap=0.002),
                material_props=NewtonRigidBodyMaterialCfg(
                    ke=4e4, kd=80, dynamic_friction=mu
                ),
                mass_props=MassPropertiesCfg(density=density),
            )

        robot = sim.add_articulation(
            ArticulationCfg(
                uid="w1",
                fpath=str(urdf),
                init_pos=(-0.03, 0, 0),
                asset_physics_mode="overlay",
                build_pk_chain=False,
                attrs=attrs(1.2),
                root_props=ArticulationRootPropertiesCfg(
                    fixed_base=True,
                    newton_kinematic=True,
                    newton_mimic_stabilizer=False,
                ),
                link_attrs={
                    "hands": LinkPhysicsOverrideCfg(
                        link_names_expr=hands, attrs=attrs(1.2, 0.0015)
                    )
                },
            )
        )
        control = RealtimeKinematicJointControl(
            "arena_0/w1",
            controller.initial_joint_positions,
            joint_names=controller.joint_names,
            initialize_pose=True,
            initialize_model_pose=True,
            interpolation_mode="raw",
        )
        sim.add_runtime_control(control)
        for name, center, size in table_pieces():
            sim.add_rigid_object(
                RigidObjectCfg(
                    uid=name,
                    shape=CubeCfg(
                        size=size,
                        visual_material=VisualMaterialCfg(
                            uid=name,
                            base_color=(
                                [0.52, 0.55, 0.56, 1]
                                if name == "table"
                                else [0.22, 0.24, 0.25, 1]
                            ),
                        ),
                    ),
                    body_type="static",
                    init_pos=center,
                    attrs=attrs(),
                )
            )
        canfile = out / "inputs/can.usda"
        _can_usd(canfile)
        goods = []
        goods.append(
            sim.add_rigid_object(
                RigidObjectCfg(
                    uid="can",
                    source=UsdSourceCfg(
                        file_path=str(canfile), source_prim_path="/can"
                    ),
                    asset_physics_mode="overlay",
                    body_type="dynamic",
                    init_pos=(0.43, -0.36, 0.98),
                    attrs=attrs(1.2, density=280),
                )
            )
        )
        goods.append(
            sim.add_rigid_object(
                RigidObjectCfg(
                    uid="carton",
                    shape=CubeCfg(
                        size=(0.048, 0.060, 0.118),
                        visual_material=VisualMaterialCfg(
                            uid="carton", base_color=[0.20, 0.48, 0.28, 1]
                        ),
                    ),
                    body_type="dynamic",
                    init_pos=(0.43, -0.25, 0.98),
                    attrs=attrs(1.2, density=280),
                )
            )
        )
        rest = mesh["vertices"]
        uv = np.column_stack(
            (
                np.arctan2(rest[:, 1] / 0.32, rest[:, 0] / 0.13) / (2 * np.pi) + 0.5,
                rest[:, 2] / 0.25,
            )
        )
        # Scalar declaration is followed by one initialize_model seed of the
        # full original rest tensors/masses before solver allocation.
        bag = sim.add_deformable_object(
            SurfaceDeformableObjectCfg(
                uid="bag",
                shape=MeshCfg(
                    vertices=mat["particle_positions"],
                    triangles=mesh["faces"],
                    uv_coords=uv,
                    visual_material=VisualMaterialCfg(
                        uid="kraft",
                        base_color=[1, 1, 1, 1],
                        base_color_texture=str(out / "inputs/kraft.png"),
                        roughness=0.95,
                    ),
                ),
                particle_radius=0.0012,
                attrs=SurfaceDeformablePhysicsCfg(
                    density=0.18,
                    contact_material={
                        "stiffness": 2e5,
                        "damping": 10.0,
                        "dynamic_friction": 0.6,
                    },
                ),
            )
        )
        sim.add_runtime_control(PaperInitialState(inputs))
        camera = None
        focal = 640 / np.tan(np.radians(20))
        if not args.no_video:
            sim.add_visual_object(
                VisualObjectCfg(
                    uid="floor_visual",
                    shape=CubeCfg(
                        size=(10, 10, 0.01),
                        visual_material=VisualMaterialCfg(
                            uid="floor", base_color=[0.30, 0.32, 0.34, 1]
                        ),
                    ),
                    init_pos=(0, 0, -0.005),
                )
            )
            sim.add_light(
                LightCfg(
                    uid="key",
                    intensity=50.0,
                    color=(1, 1, 1),
                    init_pos=(1.5, -1.0, 3.0),
                    light_type="point",
                    enable_shadow=False,
                )
            )
            camera = sim.add_sensor(
                CameraCfg(
                    uid="preview",
                    width=1280,
                    height=720,
                    near=0.01,
                    far=10.0,
                    intrinsics=(focal, focal, 640, 360),
                    enable_depth=False,
                    enable_mask=False,
                    extrinsics=CameraCfg.ExtrinsicsCfg(
                        eye=(2.75, -0.45, 2.20),
                        target=(0.44, -0.03, 1.08),
                        up=(0, 0, 1),
                    ),
                )
            )
        sim.prepare()
        sim.set_ground_plane_visibility(False)
        initial_ids = [robot.joint_names.index(n) for n in joints["joint_names"]]
        initial_q = torch.tensor(
            joints["targets"][:1], dtype=torch.float32, device=sim.device
        )
        robot.set_qpos(initial_q, joint_ids=initial_ids, target=False)
        robot.set_qpos(initial_q, joint_ids=initial_ids, target=True)
        robot.clear_dynamics()
        for material in bag.get_visual_material_inst():
            if material is not None:
                material.mat.set_double_sided(True)
        if camera is not None:
            camera.set_intrinsics(
                torch.tensor(
                    [[focal, 0, 640], [0, focal, 360], [0, 0, 1]],
                    dtype=torch.float32,
                    device=sim.device,
                )
            )
        backend = get_newton_backend(sim.get_world())
        model = backend.solver.model
        audit = audit_model(model, inputs, out)
        ids = [robot.joint_names.index(n) for n in joints["joint_names"]]
        names = inputs["manifest"]["link_state_order"]
        link_order = [robot.link_names.index(n) for n in names]
        hand_order = [
            names.index(n) for n in ("left_gripper_base", "right_gripper_base")
        ]
        initial_public = robot.body_data.body_link_pose[0, link_order].cpu().numpy()
        expected = np.stack(
            [robot.get_link_pose(n, to_matrix=True)[0].cpu().numpy() for n in names]
        )
        actual = np.broadcast_to(np.eye(4), (len(names), 4, 4)).copy()
        actual[:, :3, :3] = Rotation.from_quat(initial_public[:, 3:]).as_matrix()
        actual[:, :3, 3] = initial_public[:, :3]
        reader_error = float(abs(actual - expected).max())
        assert reader_error < 5e-7, reader_error
        from scripts.benchmark.tech_reports.physics.experiments.dexuni.c02_replay import (
            write_json,
        )

        write_json(
            out / "analysis/state_reader_validation.json",
            dict(
                links=44,
                max_matrix_error=reader_error,
                old_device_to_host_reads_per_frame=44,
                new_device_to_host_reads_per_frame=1,
                performance_timing_measured=False,
            ),
        )

        def readings() -> dict[str, np.ndarray]:
            # One owned public pose batch replaces 44 identical per-link reads.
            public = robot.body_data.body_link_pose[0, link_order].cpu().numpy()
            links = np.column_stack((public[:, 3:], public[:, :3]))
            snack = np.stack([o.get_local_pose()[0].cpu().numpy() for o in goods])
            snack_pose = np.column_stack((snack[:, 3:], snack[:, :3]))
            tcp = (
                Rotation.from_quat(public[hand_order, 3:]).apply(
                    np.array((0, 0, 0.125))
                )
                + public[hand_order, :3]
            )
            return dict(
                time=np.asarray(sim.spawn_result.get_simulation_time()),
                qpos=robot.get_qpos()[0, ids].cpu().numpy(),
                bag=bag.data.nodal_pos_w[0].cpu().numpy(),
                links=links,
                snack_pose=snack_pose,
                tcp=tcp,
                tcp_targets=controller.targets.copy(),
                joint_command=controller.current_command.copy(),
            )

        def step() -> None:
            command = controller.command(
                sim.spawn_result.get_simulation_time(),
                bag.data.nodal_pos_w[0].cpu().numpy(),
            )
            control.set_target(command)
            sim.update(physics_dt=1 / 60, step=1, render_final_step=False)

        def capture() -> np.ndarray:
            with sim.render_frame():
                sim.sync_render_state()
                camera.update()
                return camera.get_data()["color"][0, ..., :3].cpu().numpy().copy()

        def diagnostics() -> dict[str, Any]:
            return dict(
                sim.spawn_result.get_solver_diagnostics(),
                dexsim_version=dexsim.__version__,
                newton_version=newton.__version__,
                differences=[
                    "EC default ground differs; all task particles stay above the tabletop",
                    "Original heterogeneous paper arrays seeded once through public RuntimeControl.initialize_model; no runtime paper force",
                ],
            )

        record(
            "EmbodiChain",
            inputs,
            out,
            step,
            readings,
            None if args.no_video else capture,
            diagnostics,
            audit,
            args.until,
        )
    finally:
        controller.close()
        if sim is not None:
            sim.destroy(exit_process=False)


if __name__ == "__main__":
    main()
