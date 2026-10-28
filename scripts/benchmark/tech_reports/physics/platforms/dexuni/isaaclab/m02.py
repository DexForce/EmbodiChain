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
"""Preview the approved W1 one-way single cloth fold (M02) through Isaac Lab.

Run: /home/oem/app/IsaacLab/.venv/bin/python scripts/benchmark/tech_reports/physics/platforms/dexuni/isaaclab/m02.py
The frozen DexSim joint trajectory is replayed kinematically inside the solver
graph (as in M06); Newton's stock VBD solver integrates the cloth. Isaac Lab has
no surface-fast preset, so the cloth uses plain VBD sweeps.
Formal timing: add ``--measure --repeat N`` (no camera).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, ClassVar

sys.path.insert(0, str(Path(__file__).resolve().parents[7]))

from isaaclab.app import add_launcher_args, launch_simulation

import numpy as np
import torch
import warp as wp
import newton
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg
from isaaclab.physics import PhysicsManager
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg
from isaaclab.utils import configclass, instantiate
from isaaclab_newton.physics import (
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonManager,
    NewtonShapeCfg,
    NewtonSoftContactCfg,
    VBDSolverCfg,
)
from isaaclab_newton.physics.vbd_manager import NewtonVBDManager
from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import (
    NewtonSurfaceDeformableBodyMaterialCfg,
)
from isaaclab_ov.renderers import OVRTXRendererCfg
from scipy.spatial.transform import Rotation

from scripts.benchmark.tech_reports.physics.experiments.dexuni.m02_common import (
    CLOTH_CENTER,
    OUTPUT,
    TABLE_Z,
    load_inputs,
    record_preview,
)
from scripts.benchmark.tech_reports.physics.platforms.dexuni.isaaclab.m06 import (
    _advance_clock,
    _write_kinematic_joints,
)

# Stock VBD sweeps per substep: Newton's default, also DexUni's plain-VBD
# default. The native run used surface-fast (8 accelerated sweeps), which
# Isaac Lab does not provide.
VBD_ITERATIONS = 10
__all__ = ["main"]


class _KinematicReplay:
    """Named joint targets replayed per substep inside the captured graph."""

    def __init__(self, model: newton.Model, config: Any) -> None:
        names = list(config.replay_joint_names)
        self.model = model
        self.substeps = config.replay_substeps
        self.frame_dt = config.replay_dt
        self.joint_count = len(names)
        starts, dofs = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
        coord, velocity = {}, {}
        for i, label in enumerate(model.joint_label):
            name = label.rsplit("/", 1)[-1]
            if starts[i + 1] > starts[i] and name in names:
                if (
                    name in coord
                    or starts[i + 1] - starts[i] != 1
                    or dofs[i + 1] - dofs[i] != 1
                ):
                    raise ValueError(f"Unexpected M02 joint layout: {label}")
                coord[name], velocity[name] = int(starts[i]), int(dofs[i])
        missing = set(names) - set(coord)
        if missing:
            raise ValueError(f"Joints missing from the Newton model: {sorted(missing)}")
        self.q_indices = wp.array(
            [coord[n] for n in names], dtype=wp.int32, device=model.device
        )
        self.qd_indices = wp.array(
            [velocity[n] for n in names], dtype=wp.int32, device=model.device
        )
        self.targets = wp.array(
            config.replay_targets, dtype=wp.float32, device=model.device
        )
        self.counter = wp.zeros(1, dtype=wp.int32, device=model.device)

    def step(self, state_in: newton.State, state_out: newton.State) -> None:
        """Publish robot poses at the beginning and end of this substep."""
        wp.launch(
            _write_kinematic_joints,
            self.joint_count,
            inputs=[
                self.targets,
                self.counter,
                self.q_indices,
                self.qd_indices,
                self.substeps,
                self.frame_dt,
                state_in.joint_q,
                state_in.joint_qd,
                state_out.joint_q,
                state_out.joint_qd,
            ],
            device=self.model.device,
        )
        newton.eval_fk(self.model, state_in.joint_q, state_in.joint_qd, state_in)
        newton.eval_fk(self.model, state_out.joint_q, state_out.joint_qd, state_out)


class NewtonM02VBDManager(NewtonVBDManager):
    """Standard VBD manager dispatch with a kinematic (prescribed) robot."""

    replay: ClassVar[_KinematicReplay | None] = None

    @classmethod
    def _prepare_builder_for_finalize(cls, builder: newton.ModelBuilder) -> None:
        """Mark every robot body kinematic: the cloth does not push back."""
        for i in range(builder.body_count):
            builder.body_flags[i] = int(newton.BodyFlags.KINEMATIC)
        super()._prepare_builder_for_finalize(builder)

    @classmethod
    def _create_solver(cls, model: newton.Model, solver_cfg: VBDSolverCfg) -> Any:
        """Construct upstream VBD and the fixed kinematic replay buffers."""
        cls.replay = _KinematicReplay(model, solver_cfg)
        return super()._create_solver(model, solver_cfg)

    @classmethod
    def _step_solver(
        cls,
        state_0: newton.State,
        state_1: newton.State,
        control: newton.Control,
        contacts: Any,
        substep_dt: float,
    ) -> None:
        """Refresh kinematics and one contact set before upstream VBD."""
        cls.replay.step(state_0, state_1)
        cls._collision_pipeline.collide(state_0, contacts)
        super()._step_solver(state_0, state_1, control, contacts, substep_dt)
        wp.launch(
            _advance_clock,
            1,
            inputs=[cls.replay.counter],
            device=cls.replay.model.device,
        )

    @classmethod
    def _simulate_physics_only(cls) -> None:
        """Preserve the VBD manager's BVH, force, swap and sensor lifecycle."""
        if cls.backend.model.particle_count > 0:
            cls._solver.rebuild_bvh(cls.backend.state_0)
        cls._stage_body_forces()
        cls._run_solver_substeps(cls._contacts)
        for callback in cls._post_step_callbacks:
            callback()
        cls._update_sensors(cls._contacts)

    @classmethod
    def _simulate_full(cls) -> None:
        """The kinematic scene has no actuator integration."""
        cls._simulate_physics_only()


@configclass
class _M02VBDSolverCfg(VBDSolverCfg):
    """Bind the kinematic manager and the native friction smoothing."""

    class_type: type = NewtonM02VBDManager
    friction_epsilon: float = 1e-4
    replay_targets: Any = None
    replay_joint_names: Any = None
    replay_dt: float = 1 / 60
    replay_substeps: int = 10


def _write_cloth_usd(mesh: dict, color, path: Path) -> None:
    """Author the frozen node order (relative to the cloth centre) and colour."""
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    root = UsdGeom.Xform.Define(stage, "/Cloth")
    stage.SetDefaultPrim(root.GetPrim())
    surface = UsdGeom.Mesh.Define(stage, "/Cloth/mesh")
    local = mesh["vertices"] - CLOTH_CENTER
    surface.GetPointsAttr().Set([Gf.Vec3f(*map(float, p)) for p in local])
    surface.GetFaceVertexCountsAttr().Set([3] * len(mesh["triangles"]))
    surface.GetFaceVertexIndicesAttr().Set(mesh["triangles"].reshape(-1).tolist())
    surface.GetDoubleSidedAttr().Set(True)
    surface.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    surface.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
    material = UsdShade.Material.Define(stage, "/Cloth/Looks/Fabric")
    shader = UsdShade.Shader.Define(stage, "/Cloth/Looks/Fabric/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
    material.CreateSurfaceOutput().ConnectToSource(
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    )
    UsdShade.MaterialBindingAPI.Apply(surface.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()


def _numpy(value: object) -> np.ndarray:
    """Copy an Isaac Lab tensor or Warp wrapper to the host."""
    return (value.torch if hasattr(value, "torch") else value).cpu().numpy().copy()


def main() -> None:
    """Construct the kinematic W1 + VBD cloth scene and record its preview."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    add_launcher_args(parser)
    parser.set_defaults(visualizer=[])
    args = parser.parse_args()
    started = time.perf_counter()
    measure_dir = None
    if args.measure:
        from scripts.benchmark.tech_reports.physics.experiments.dexuni.m02_measurement import (
            measurement_directory,
        )

        measure_dir = measurement_directory("isaaclab", args.repeat)
    inputs = load_inputs()
    cfg, mesh, names = inputs["config"], inputs["mesh"], inputs["names"]
    cloth_cfg, camera_cfg = cfg["cloth"], cfg["camera"]
    usd = str(OUTPUT / "inputs/w1_usd/robot/robot.usda")
    cloth_usd = OUTPUT / "inputs/cloth.usda"
    # The previewed authored mesh stays unchanged while timing.
    if not args.measure:
        _write_cloth_usd(mesh, cloth_cfg["color"], cloth_usd)
    contact = cfg["self_contact"]
    physics = NewtonCfg(
        solver_cfg=_M02VBDSolverCfg(
            replay_targets=inputs["targets"],
            replay_joint_names=names,
            replay_dt=cfg["dt"],
            replay_substeps=cfg["substeps"],
            friction_epsilon=cfg["friction_epsilon"],
            iterations=VBD_ITERATIONS,
            integrate_with_external_rigid_solver=True,
            particle_enable_self_contact=contact["particle_enable_self_contact"],
            particle_self_contact_radius=contact["particle_self_contact_radius"],
            particle_self_contact_margin=contact["particle_self_contact_margin"],
            particle_rest_shape_contact_exclusion_radius=contact[
                "particle_rest_shape_contact_exclusion_radius"
            ],
            # Native (DexUni) values: detection once per step, default buffers
            # and topological filter.
            particle_collision_detection_interval=-1,
            particle_vertex_contact_buffer_size=32,
            particle_edge_contact_buffer_size=64,
            particle_topological_contact_filter_threshold=2,
        ),
        soft_contact_cfg=NewtonSoftContactCfg(
            soft_contact_ke=cloth_cfg["contact_ke"],
            soft_contact_kd=cloth_cfg["contact_kd"],
            soft_contact_mu=cloth_cfg["contact_mu"],
        ),
        default_shape_cfg=NewtonShapeCfg(
            ke=cfg["shape_ke"],
            kd=cfg["shape_kd"],
            mu=cfg["robot_mu"],
            margin=cfg["shape_margin"],
            gap=cfg["shape_gap"],
        ),
        collision_cfg=NewtonCollisionPipelineCfg(
            broad_phase="nxn",
            soft_contact_margin=cfg["soft_contact_margin"],
            enable_rigid_soft_full_surface_contact=False,
        ),
        num_substeps=cfg["substeps"],
        collision_decimation=0,
        use_cuda_graph=cfg["use_cuda_graph"],
    )
    width, height = camera_cfg["resolution"]
    aperture = 20.955
    focal = aperture / 2 / np.tan(np.deg2rad(camera_cfg["hfov"]) / 2)

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            actuators={},
            spawn=sim_utils.UsdFileCfg(usd_path=usd, variants={"Physics": "mujoco"}),
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos=dict(zip(names, inputs["targets"][0].tolist()))
            ),
        )
        cloth = DeformableObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cloth",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(cloth_usd),
                deformable_props=NewtonDeformableBodyPropertiesCfg(),
                physics_material=NewtonSurfaceDeformableBodyMaterialCfg(
                    density=cloth_cfg["surface_density"],
                    particle_radius=cloth_cfg["particle_radius"],
                    **{
                        k: cloth_cfg[k]
                        for k in ("tri_ke", "tri_ka", "tri_kd", "edge_ke", "edge_kd")
                    },
                ),
            ),
            init_state=DeformableObjectCfg.InitialStateCfg(
                pos=tuple(float(v) for v in CLOTH_CENTER)
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

    # Table and legs use the native static friction (0.8).
    static_material = UsdPhysicsRigidBodyMaterialCfg(
        static_friction=cfg["static_mu"], dynamic_friction=cfg["static_mu"]
    )
    boxes = [("table", (0.46, 0.0, TABLE_Z - 0.025), (0.6, 1.64, 0.05))] + [
        (f"leg_{i}_{j}", (x, y, (TABLE_Z - 0.05) / 2), (0.05, 0.05, TABLE_Z - 0.05))
        for i, x in enumerate((0.21, 0.71))
        for j, y in enumerate((-0.74, 0.74))
    ]
    for name, center, size in boxes:
        setattr(
            SceneCfg,
            name,
            AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/" + name,
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    physics_material=static_material,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.5, 0.5, 0.5)
                    ),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=center),
            ),
        )
    SceneCfg.ground = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Ground",
        spawn=sim_utils.GroundPlaneCfg(size=(8.0, 8.0)),
    )
    try:
        with launch_simulation(physics, args):
            sim = sim_utils.SimulationContext(
                sim_utils.SimulationCfg(
                    dt=cfg["dt"],
                    device=args.device,
                    gravity=(0, 0, -9.81),
                    physics=physics,
                )
            )
            scene = instantiate(
                SceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=True)
            )
            sim.reset()
            replay = NewtonM02VBDManager.replay
            replay.counter.zero_()
            model = NewtonManager.get_model()
            if np.any(model.particle_inv_mass.numpy() <= 0):
                raise RuntimeError("All M02 cloth nodes must remain dynamic")
            output = measure_dir if args.measure else OUTPUT / "preview/isaaclab"
            output.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output / "model.npz",
                shape_label=np.asarray([str(s) for s in model.shape_label]),
                **{
                    name: getattr(model, name).numpy()
                    for name in (
                        "particle_q",
                        "particle_mass",
                        "particle_radius",
                        "tri_indices",
                        "tri_materials",
                        "edge_rest_angle",
                        "edge_bending_properties",
                        "shape_material_ke",
                        "shape_material_kd",
                        "shape_material_mu",
                        "shape_margin",
                        "shape_gap",
                        "body_flags",
                    )
                    if hasattr(getattr(model, name, None), "numpy")
                },
            )
            robot, cloth = scene["robot"], scene["cloth"]
            ids = [robot.joint_names.index(n) for n in names]
            robot.write_joint_position_to_sim_index(
                position=torch.tensor(
                    inputs["targets"][:1], device=sim.device, dtype=torch.float32
                ),
                joint_ids=ids,
            )
            scene.write_data_to_sim()
            scene.update(0.0)
            camera = None
            if not args.measure:
                camera = scene["camera"]
                camera.set_world_poses_from_view(
                    eyes=torch.tensor([camera_cfg["eye"]], device=sim.device),
                    targets=torch.tensor([camera_cfg["target"]], device=sim.device),
                )
            body_ids = [
                robot.body_names.index(f"{s}_gripper_base") for s in ("left", "right")
            ]
            initial = _numpy(cloth.data.nodal_pos_w)[0]
            if not np.allclose(initial, mesh["vertices"], atol=1e-6):
                raise RuntimeError(
                    "Cloth node order/positions differ from the frozen mesh"
                )

            def readings() -> dict[str, np.ndarray]:
                pose = _numpy(robot.data.body_link_pose_w)[0]
                return {
                    "time": np.asarray(PhysicsManager.get_simulation_time()),
                    "qpos": _numpy(robot.data.joint_pos)[0, ids],
                    "particles": _numpy(cloth.data.nodal_pos_w)[0],
                    "tcp": np.stack(
                        [
                            pose[i, :3]
                            + Rotation.from_quat(pose[i, 3:]).apply(cfg["tcp_offset"])
                            for i in body_ids
                        ]
                    ),
                }

            def step(frame: int) -> None:
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(cfg["dt"])
                if int(replay.counter.numpy()[0]) != frame * cfg["substeps"]:
                    raise RuntimeError(f"Replay counter differs at frame {frame}")
                qpos = _numpy(robot.data.joint_pos)[0, ids]
                if not np.allclose(qpos, inputs["targets"][frame], atol=1e-5):
                    raise RuntimeError(
                        f"Kinematic joint tracking failed at frame {frame}"
                    )

            def capture() -> np.ndarray:
                sim.render()
                return _numpy(camera.data.output["rgb"])[0, ..., :3]

            def diagnostics() -> dict:
                return {
                    "outer_steps_per_command": 1,
                    "simulated_elapsed_s": PhysicsManager.get_simulation_time(),
                    "actual_substeps": int(replay.counter.numpy()[0]),
                    "num_substeps": physics.num_substeps,
                    "solver_dt": NewtonManager.get_solver_dt(),
                    "cuda_graph_status": (
                        "captured"
                        if NewtonManager._graph is not None
                        else "not captured"
                    ),
                    "vbd_iterations": NewtonM02VBDManager._solver.iterations,
                    "vbd_preset": None,
                    "solver": "upstream Newton SolverVBD (Isaac Lab VBD manager)",
                    "joint_mode": "kinematic",
                    "particle_color_groups": len(model.particle_color_groups),
                    "newton_version": newton.__version__,
                }

            if args.measure:
                from scripts.benchmark.tech_reports.physics.experiments.dexuni.m02_measurement import (
                    record_episode,
                )

                def advance() -> None:
                    scene.write_data_to_sim()
                    sim.step(render=False)
                    scene.update(cfg["dt"])

                record_episode(
                    "isaaclab",
                    measure_dir,
                    inputs,
                    advance,
                    readings,
                    lambda: wp.synchronize_device(args.device),
                    diagnostics,
                    time.perf_counter() - started,
                )
            else:
                record_preview("isaaclab", inputs, step, readings, capture, diagnostics)
    finally:
        NewtonM02VBDManager.replay = None


if __name__ == "__main__":
    main()
