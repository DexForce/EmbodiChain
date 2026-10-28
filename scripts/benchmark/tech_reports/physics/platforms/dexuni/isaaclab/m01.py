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
"""Preview or time the approved W1 dual-arm soft-beam bend (M01) through Isaac Lab.

Run: /home/oem/app/IsaacLab/.venv/bin/python scripts/benchmark/tech_reports/physics/platforms/dexuni/isaaclab/m01.py
Formal timing: add ``--measure --repeat N`` (no camera).
Replays the frozen DexSim joint commands on the same tetrahedral beam.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[7]))

from isaaclab.app import add_launcher_args, launch_simulation
from scripts.benchmark.tech_reports.physics.experiments.dexuni.m01_common import (
    OUTPUT,
    load_inputs,
    record_preview,
    static_boxes,
)

import numpy as np
import torch
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg
from isaaclab.physics import PhysicsManager
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg
from isaaclab.utils import configclass, instantiate
from isaaclab_contrib.coupling import (
    CouplerEntryCfg,
    CouplerProxyCfg,
    CouplerProxyMappingCfg,
)
from isaaclab_newton.physics import (
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonManager,
    NewtonShapeCfg,
    NewtonSoftContactCfg,
    VBDSolverCfg,
)
from isaaclab_newton.sim.schemas import (
    MujocoRigidBodyPropertiesCfg,
    NewtonDeformableBodyPropertiesCfg,
)
from isaaclab_newton.sim.spawners.materials import NewtonDeformableBodyMaterialCfg
from isaaclab_ov.renderers import OVRTXRendererCfg

__all__ = ["main"]


@configclass
class _VBDSolverCfg(VBDSolverCfg):
    """Pass the native reference's contact options through the VBD manager."""

    friction_epsilon: float = 1e-4
    rigid_contact_hard: bool = False
    rigid_contact_history: bool = False


def _write_beam_usd(mesh: dict, cfg: dict, path: Path) -> None:
    """Author the original tetrahedral node order and an orange render material."""
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    root = UsdGeom.Xform.Define(stage, "/Beam")
    stage.SetDefaultPrim(root.GetPrim())
    tet = UsdGeom.TetMesh.Define(stage, "/Beam/tet")
    local = mesh["particle_q"] - np.asarray(cfg["soft_center"])
    tet.GetPointsAttr().Set([Gf.Vec3f(*map(float, p)) for p in local])
    tet.GetTetVertexIndicesAttr().Set(
        [Gf.Vec4i(*map(int, t)) for t in mesh["tet_indices"]]
    )
    tet.GetDisplayColorAttr().Set([Gf.Vec3f(0.9, 0.4, 0.2)])
    material = UsdShade.Material.Define(stage, "/Beam/visual_material")
    shader = UsdShade.Shader.Define(stage, "/Beam/visual_material/shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.9, 0.4, 0.2)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(root.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()


def _matrix(pose: np.ndarray) -> np.ndarray:
    """Convert Isaac Lab's xyz+xyzw pose to a homogeneous matrix."""
    from scipy.spatial.transform import Rotation

    out = np.eye(4)
    out[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    out[:3, 3] = pose[:3]
    return out


def _numpy(value: object) -> np.ndarray:
    """Read a Torch tensor or Isaac Lab Warp wrapper on the host."""
    return (value.torch if hasattr(value, "torch") else value).cpu().numpy()


def main() -> None:
    """Construct a dynamic MuJoCo/VBD proxy scene and record its preview."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    add_launcher_args(parser)
    parser.set_defaults(visualizer=[])
    args = parser.parse_args()
    started = time.perf_counter()
    if args.measure:
        from scripts.benchmark.tech_reports.physics.experiments.dexuni.m01_measurement import (
            measurement_directory,
        )

        output = measurement_directory("isaaclab", args.repeat)
    else:
        output = OUTPUT / "preview/isaaclab"
    inputs = load_inputs()
    cfg, mesh, names = inputs["config"], inputs["mesh"], inputs["names"]
    fingers = [n for n in names if "FINGER" in n]
    other = [n for n in names if n not in fingers]
    urdf = OUTPUT / "inputs" / cfg["robot_urdf"]
    efforts = {
        joint.attrib["name"]: float(joint.find("limit").attrib["effort"])
        for joint in ET.parse(urdf).getroot().findall("joint")
        if joint.find("limit") is not None
    }
    usd = json.loads((OUTPUT / "inputs/w1_usd_path.json").read_text())["usd_path"]
    beam_usd = OUTPUT / "inputs/beam.usda"
    # The previewed authored mesh stays unchanged while timing.
    if not args.measure:
        _write_beam_usd(mesh, cfg, beam_usd)
    pipeline = NewtonCollisionPipelineCfg(
        broad_phase="nxn",
        soft_contact_margin=0.003,
        enable_rigid_soft_full_surface_contact=False,
    )
    physics = NewtonCfg(
        solver_cfg=CouplerProxyCfg(
            entries=[
                CouplerEntryCfg(
                    name="robot",
                    solver_cfg=MJWarpSolverCfg(
                        integrator="implicitfast", ls_iterations=50
                    ),
                    bodies=[r"/World/envs/env_[^/]+/Robot"],
                ),
                CouplerEntryCfg(
                    name="vbd",
                    solver_cfg=_VBDSolverCfg(
                        iterations=cfg["vbd_iterations"],
                        rigid_body_contact_buffer_size=4096,
                    ),
                    all_particles=True,
                    include_static_shapes=True,
                ),
            ],
            proxies=[
                CouplerProxyMappingCfg(
                    source="robot",
                    destination="vbd",
                    bodies=[r"/World/envs/env_[^/]+/Robot/.*"],
                    mode="staggered",
                    mass_scale=cfg["proxy_mass_scale"],
                    proxy_relaxation=cfg["proxy_relaxation"],
                    collide_interval=1,
                    collision_pipeline=pipeline,
                )
            ],
            iterations=1,
        ),
        soft_contact_cfg=NewtonSoftContactCfg(
            soft_contact_ke=cfg["particle_ke"],
            soft_contact_kd=cfg["particle_kd"],
            soft_contact_mu=cfg["particle_mu"],
        ),
        default_shape_cfg=NewtonShapeCfg(
            ke=cfg["shape_ke"],
            kd=cfg["shape_kd"],
            mu=cfg["robot_mu"],
            margin=cfg["shape_margin"],
            gap=cfg["shape_gap"],
        ),
        collision_cfg=pipeline,
        num_substeps=cfg["substeps"],
        use_cuda_graph=cfg["use_cuda_graph"],
    )
    width, height = cfg["camera"]["resolution"]
    aperture = 20.955
    focal = aperture / 2 / np.tan(np.deg2rad(cfg["camera"]["hfov"]) / 2)

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=usd,
                variants={"Physics": "mujoco"},
                rigid_props=MujocoRigidBodyPropertiesCfg(
                    gravcomp=cfg["gravity_compensation"]
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos=dict(zip(names, inputs["targets"][0].tolist()))
            ),
            actuators={
                "body": ImplicitActuatorCfg(
                    joint_names_expr=other,
                    stiffness=cfg["arm_kp"],
                    damping=cfg["arm_kd"],
                    effort_limit_sim={n: efforts[n] for n in other},
                ),
                "fingers": ImplicitActuatorCfg(
                    joint_names_expr=fingers,
                    stiffness=cfg["finger_kp"],
                    damping=cfg["finger_kd"],
                    armature=cfg["finger_armature"],
                    effort_limit_sim=10.0,
                ),
            },
        )
        beam = DeformableObjectCfg(
            prim_path="{ENV_REGEX_NS}/Beam",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(beam_usd),
                deformable_props=NewtonDeformableBodyPropertiesCfg(),
                physics_material=NewtonDeformableBodyMaterialCfg(
                    density=cfg["soft_density"],
                    k_mu=cfg["soft_mu"],
                    k_lambda=cfg["soft_lambda"],
                    k_damp=cfg["soft_damping"],
                    particle_radius=cfg["particle_radius"],
                ),
            ),
            init_state=DeformableObjectCfg.InitialStateCfg(
                pos=tuple(cfg["soft_center"])
            ),
        )
        if not args.measure:
            light = AssetBaseCfg(
                prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=600.0)
            )
            camera = CameraCfg(
                prim_path="{ENV_REGEX_NS}/Camera",
                update_period=0.0,
                height=height,
                width=width,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=float(focal),
                    horizontal_aperture=aperture,
                    clipping_range=(0.01, 10.0),
                ),
                renderer_cfg=OVRTXRendererCfg(),
            )

    # Table, legs and marks use the native static friction (0.8).
    static_material = UsdPhysicsRigidBodyMaterialCfg(
        static_friction=cfg["static_mu"], dynamic_friction=cfg["static_mu"]
    )
    for name, center, size, color in static_boxes(cfg):
        setattr(
            SceneCfg,
            name,
            AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/" + name,
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    physics_material=static_material,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=center),
            ),
        )
    setattr(
        SceneCfg,
        "ground",
        AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Ground",
            spawn=sim_utils.GroundPlaneCfg(size=(8.0, 8.0)),
        ),
    )
    with launch_simulation(physics, args):
        sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(dt=cfg["dt"], device=args.device, physics=physics)
        )
        scene = instantiate(
            SceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=True)
        )
        sim.reset()
        robot = scene["robot"]
        ids = [robot.joint_names.index(n) for n in names]
        finger_ids = [robot.joint_names.index(n) for n in fingers]
        initial = torch.tensor(inputs["targets"][0], device=sim.device).unsqueeze(0)
        robot.write_joint_position_to_sim_index(position=initial, joint_ids=ids)
        robot.set_joint_position_target_index(target=initial, joint_ids=ids)
        robot.write_joint_viscous_friction_coefficient_to_sim_index(
            joint_viscous_friction_coeff=cfg["finger_passive_damping"],
            joint_ids=finger_ids,
        )
        scene.write_data_to_sim()
        scene.update(0.0)
        camera = None
        if not args.measure:
            camera = scene["camera"]
            camera.set_world_poses_from_view(
                eyes=torch.tensor([cfg["camera"]["eye"]], device=sim.device),
                targets=torch.tensor([cfg["camera"]["target"]], device=sim.device),
            )
        model = NewtonManager.backend.model
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output / "model.npz",
            shape_label=np.asarray([str(s) for s in model.shape_label]),
            **{
                name: getattr(model, name).numpy()
                for name in (
                    "particle_mass",
                    "particle_q",
                    "particle_radius",
                    "tet_indices",
                    "tet_materials",
                    "tri_indices",
                    "shape_material_mu",
                    "joint_target_ke",
                    "joint_target_kd",
                    "joint_armature",
                    "joint_effort_limit",
                    "joint_damping",
                    "body_mass",
                    "body_inertia",
                )
                if hasattr(getattr(model, name, None), "numpy")
            },
        )
        body_ids = [
            robot.body_names.index(f"{side}_gripper_base") for side in ["left", "right"]
        ]

        def set_target(frame: int) -> None:
            robot.set_joint_position_target_index(
                target=torch.tensor(
                    inputs["targets"][frame], device=sim.device
                ).unsqueeze(0),
                joint_ids=ids,
            )

        def advance() -> None:
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(cfg["dt"])

        def step(frame: int) -> None:
            set_target(frame)
            advance()

        def readings() -> dict[str, np.ndarray]:
            body = _numpy(robot.data.body_link_pose_w)[0]
            return {
                "qpos": _numpy(robot.data.joint_pos)[0, ids],
                "particles": _numpy(scene["beam"].data.nodal_pos_w)[0],
                "tcp": np.stack(
                    [
                        (_matrix(body[i]) @ np.array([0, 0, 0.125, 1]))[:3]
                        for i in body_ids
                    ]
                ),
            }

        def capture() -> np.ndarray:
            sim.render()
            return _numpy(camera.data.output["rgb"])[0, ..., :3]

        simulation_time_origin = PhysicsManager._sim_time

        def diagnostics() -> dict:
            return {
                "outer_steps_per_command": 1,
                "simulated_elapsed_s": PhysicsManager._sim_time
                - simulation_time_origin,
                "num_substeps": NewtonManager._num_substeps,
                "cuda_graph_status": (
                    "captured" if NewtonManager._graph is not None else "disabled"
                ),
                "vbd_iterations": NewtonManager._solver._entries[
                    "vbd"
                ].solver.iterations,
                "newton_version": __import__("newton").__version__,
            }

        if args.measure:
            import warp as wp

            from scripts.benchmark.tech_reports.physics.experiments.dexuni.m01_measurement import (
                record_episode,
            )

            record_episode(
                "isaaclab",
                output,
                inputs,
                set_target,
                advance,
                readings,
                lambda: wp.synchronize_device(args.device),
                diagnostics,
                time.perf_counter() - started,
            )
        else:
            record_preview("isaaclab", inputs, step, readings, capture, diagnostics)


if __name__ == "__main__":
    main()
