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
"""S03 physical cloth-corner lift through the isaaclab platform.

Run with DEXUNI_ALLOW_PILOT=1 and --inputs pointing to a prepared S03 snapshot.
All cloth nodes remain dynamic. Results default to outputs/benchmarks/tech_reports/dexuni/.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

REPO_ROOT = Path(__file__).resolve().parents[7]
sys.path.insert(0, str(REPO_ROOT))
from scripts.benchmark.tech_reports.physics.experiments.dexuni.paths import (
    EXPERIMENTS,
    PLATFORMS,
    input_dir,
    output_dir,
    franka_usd,
    require_pilot,
)

from scripts.benchmark.tech_reports.physics.experiments.dexuni.s03_common import (
    CornerLiftController as PickPlaceController,
    read_inputs,
)

__all__ = ["run"]

TASK_DIR = EXPERIMENTS / "s03_cloth_corner_lift"
INPUTS = input_dir("s03")
BENCH_ROOT = PLATFORMS
if __name__ == "__main__":
    require_pilot()
PLATFORM = "isaaclab"
# Converted input is selected outside the source tree.
FRANKA_USD = franka_usd()

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawTextHelpFormatter
)
parser.add_argument("--state", default="s00")
parser.add_argument("--variant", default="nominal")
parser.add_argument("--kind", choices=("full", "timing"), default="full")
parser.add_argument("--images", choices=("full", "keyframes", "none"), default="full")
parser.add_argument("--repeat", type=int, default=0)
parser.add_argument(
    "--preview-only",
    action="store_true",
    help="Record scene video/state only; no timing or task metrics",
)
parser.add_argument("--inputs", help="alternative inputs directory")
parser.add_argument(
    "--set", action="append", help="debug override a.b=<json>, recorded in config"
)
parser.add_argument("--out-root", help="write raw_data/ under this root")
add_launcher_args(parser)
parser.set_defaults(visualizer=[])
args_cli = parser.parse_args()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import (
    ArticulationCfg,
    AssetBaseCfg,  # noqa: E402
    DeformableObjectCfg,
    RigidObjectCfg,
)
from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.utils import configclass, instantiate  # noqa: E402
from isaaclab_contrib.coupling import (
    CouplerEntryCfg,  # noqa: E402
    CouplerProxyCfg,
    CouplerProxyMappingCfg,
)
from isaaclab_newton.physics import (
    MJWarpSolverCfg,
    NewtonCfg,  # noqa: E402
    NewtonCollisionPipelineCfg,
    NewtonShapeCfg,
    NewtonSoftContactCfg,
    VBDSolverCfg,
)
from isaaclab_ov.renderers import OVRTXRendererCfg  # noqa: E402
from isaaclab_newton.sim.schemas import (  # noqa: E402
    MujocoRigidBodyPropertiesCfg,
    NewtonDeformableBodyPropertiesCfg,
)
from isaaclab_newton.sim.spawners.materials import (
    NewtonSurfaceDeformableBodyMaterialCfg,
)  # noqa: E402

sys.path.insert(0, str(BENCH_ROOT / "_lib"))  # benchlib copy

# Common segmentation label ids (config measurement.images.labels).
LABELS = {"ground": 1, "table": 2, "robot_stand": 3, "franka": 4, "cloth": 5}
# Scene prim name -> common label (semantic class tag on each prim).
PRIM_LABELS = {
    "Ground": "ground",
    "Table": "table",
    "RobotStand": "robot_stand",
    "Robot": "franka",
}


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_path(config: dict, key: str, value) -> None:
    node = config
    parts = key.split(".")
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


def load_inputs(args) -> dict:
    global INPUTS
    if args.inputs:
        INPUTS = Path(args.inputs).resolve()
    return read_inputs(args, INPUTS)


def arm_target(inputs, s: float) -> np.ndarray:
    """Interpolate the shared IK path table (arc length s -> 7 arm joints)."""
    x = inputs["ik_s"]
    if not x[0] - 1e-9 <= s <= x[-1] + 1e-9:
        raise ValueError(f"path parameter {s:.4f} m outside the IK table")
    return np.array([np.interp(s, x, col) for col in inputs["ik_q"].T])


def block_width(particles: np.ndarray, patches: np.ndarray) -> float:
    """Mean height of the fixed material corner nodes, in world metres."""
    return float(particles[patches, 2].mean())


def lame_parameters(youngs_modulus: float, poisson_ratio: float):
    e, nu = float(youngs_modulus), float(poisson_ratio)
    return e / (2.0 * (1.0 + nu)), e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))


# --------------------------------------------------------------------------
# Closed-loop pick-and-place rule (identical class in every platform script)
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Authored USD input (written from the shared files for each run)
# --------------------------------------------------------------------------


def write_block_usd(inputs, path: Path) -> None:
    """Author a USD triangle Mesh for Newton surface-deformable discovery."""
    centre = np.asarray(inputs["state"]["block_center"], dtype=np.float32)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Cloth")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Cloth/mesh")
    mesh.GetPointsAttr().Set(
        [Gf.Vec3f(*map(float, p)) for p in inputs["initial"] - centre]
    )
    mesh.GetFaceVertexCountsAttr().Set([3] * len(inputs["surface"]))
    mesh.GetFaceVertexIndicesAttr().Set(inputs["surface"].reshape(-1).tolist())
    mesh.GetDoubleSidedAttr().Set(True)
    mesh.GetDisplayColorAttr().Set(
        [Gf.Vec3f(*inputs["config"]["cloth"].get("render_color", (0.25, 0.65, 0.9)))]
    )
    if inputs["cloth_texture"]:
        uv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
        )
        uv.Set([Gf.Vec2f(*map(float, p)) for p in inputs["cloth_uv"]])
        material = UsdShade.Material.Define(stage, "/Cloth/Looks/Fabric")
        surface = UsdShade.Shader.Define(stage, "/Cloth/Looks/Fabric/Surface")
        surface.CreateIdAttr("UsdPreviewSurface")
        surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
            inputs["config"]["cloth"]["visual"]["roughness"]
        )
        surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        reader = UsdShade.Shader.Define(stage, "/Cloth/Looks/Fabric/UV")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        texture = UsdShade.Shader.Define(stage, "/Cloth/Looks/Fabric/Texture")
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            inputs["cloth_texture"]
        )
        texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        for axis in ("wrapS", "wrapT"):
            texture.CreateInput(axis, Sdf.ValueTypeNames.Token).Set("repeat")
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        )
        surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        )
        material.CreateSurfaceOutput().ConnectToSource(
            surface.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        )
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()


# --------------------------------------------------------------------------
# Scene and physics configuration
# --------------------------------------------------------------------------


@configclass
class S03VBDSolverCfg(VBDSolverCfg):
    """VBDSolverCfg plus SolverVBD options it does not declare.

    Isaac Lab forwards every cfg field that matches a ``SolverVBD.__init__``
    parameter, so these reach the solver unchanged.
    """

    friction_epsilon: float = 0.01
    rigid_contact_hard: bool = True


def collision_pipeline(config):
    """Per-vertex particle contact with the shared margin (Isaac Lab default
    soft_contact_margin is 0.01 m; DexSim gets the same config value)."""
    return NewtonCollisionPipelineCfg(
        broad_phase="nxn",
        soft_contact_margin=config["contact"]["soft_contact_margin"],
        enable_rigid_soft_full_surface_contact=False,
    )


def physics_config(config):
    solver = config["solver"]
    contact = config["contact"]
    two_way = config["coupling"] == "two_way"
    return NewtonCfg(
        solver_cfg=CouplerProxyCfg(
            entries=[
                CouplerEntryCfg(
                    name="rigid",
                    solver_cfg=MJWarpSolverCfg(
                        cone=solver["mujoco_cone"],
                        ls_iterations=50,
                        integrator="implicitfast",
                    ),
                    bodies=[r"/World/envs/env_[^/]+/Robot"],
                ),
                CouplerEntryCfg(
                    name="soft",
                    solver_cfg=S03VBDSolverCfg(
                        iterations=solver["vbd_iterations"],
                        friction_epsilon=solver["vbd_friction_epsilon"],
                        rigid_contact_hard=solver["vbd_rigid_contact_hard"],
                        rigid_body_particle_contact_buffer_size=1024,
                        **solver.get("vbd_self_contact", {}),
                    ),
                    bodies=[],
                    all_particles=True,
                    include_static_shapes=True,
                ),
            ],
            proxies=[
                CouplerProxyMappingCfg(
                    source="rigid",
                    destination="soft",
                    # DexUni proxies every MuJoCo link (solver.proxied_robot_links
                    # = "all"); "hand" keeps only the hand and fingers.
                    bodies=(
                        [r"/World/envs/env_[^/]+/Robot/.*"]
                        if solver["proxied_robot_links"] == "all"
                        else [
                            r"/World/envs/env_[^/]+/Robot/.*fr3_hand",
                            r"/World/envs/env_[^/]+/Robot/.*fr3_(left|right)finger",
                        ]
                    ),
                    mode=solver["coupling_mode"],
                    mass_scale=solver["proxy_mass_scale"],
                    proxy_relaxation=(solver["proxy_relaxation"] if two_way else 0.0),
                    collide_interval=1,
                    collision_pipeline=collision_pipeline(config),
                )
            ],
            iterations=solver["coupling_iterations"],
        ),
        soft_contact_cfg=NewtonSoftContactCfg(
            soft_contact_ke=contact["particle_contact_ke"],
            soft_contact_kd=contact["particle_contact_kd"],
            soft_contact_mu=config["cloth"]
            .get("contact_material", {})
            .get("dynamic_friction", contact["friction"]),
        ),
        default_shape_cfg=NewtonShapeCfg(
            margin=contact["shape_margin"],
            gap=contact["shape_gap"],
            ke=contact["shape_ke"],
            kd=contact["shape_kd"],
            mu=contact["friction"],
        ),
        collision_cfg=collision_pipeline(config),
        num_substeps=solver["num_substeps"],
        use_cuda_graph=solver["use_cuda_graph"],
    )


def static_box(prim, center, size, color):
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/" + prim,
        init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(center)),
        spawn=sim_utils.CuboidCfg(
            size=tuple(size),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            semantic_tags=[("class", PRIM_LABELS[prim])],
        ),
    )


def robot_config(config, initial_row):
    robot = config["robot"]
    arm, fingers = robot["arm_joints"], robot["finger_joints"]
    joint_pos = {name: float(v) for name, v in zip(arm + fingers, initial_row)}
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(FRANKA_USD),
            variants={"Physics": "mujoco"},
            # Same mechanism as DexSim's newton_gravity_compensation=1.0:
            # MuJoCo body gravcomp (disable_gravity is PhysX-only).
            rigid_props=MujocoRigidBodyPropertiesCfg(gravcomp=1.0),
            semantic_tags=[("class", "franka")],
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(robot["base_position"]), joint_pos=joint_pos
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=arm,
                stiffness=dict(zip(arm, robot["arm_kp"])),
                damping=dict(zip(arm, robot["arm_kd"])),
                effort_limit_sim=dict(zip(arm, robot["arm_effort_limit"])),
                armature=robot["arm_armature"],
            ),
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=fingers,
                stiffness=robot["finger_kp"],
                damping=robot["finger_kd"],
                effort_limit_sim=robot["finger_effort_limit"],
                armature=robot["finger_armature"],
            ),
        },
    )


def intrinsics(config) -> np.ndarray:
    width, height = config["measurement"]["camera_resolution"]
    fx = (
        0.5 * width / np.tan(0.5 * np.deg2rad(config["measurement"]["camera_hfov_deg"]))
    )
    return np.array([[fx, 0, 0.5 * width], [0, fx, 0.5 * height], [0, 0, 1]])


def scene_config(inputs, initial_row, usd_dir: Path, cameras: bool):
    config = inputs["config"]
    scene = config["scene"]
    block_cfg = config["cloth"]
    block_usd = usd_dir / "cloth.usda"
    write_block_usd(inputs, block_usd)
    width, height = config["measurement"]["camera_resolution"]
    aperture = 20.955
    focal = (
        0.5
        * aperture
        / np.tan(0.5 * np.deg2rad(config["measurement"]["camera_hfov_deg"]))
    )

    @configclass
    class S03SceneCfg(InteractiveSceneCfg):
        robot = robot_config(config, initial_row)
        table = static_box(
            "Table",
            scene["table"]["center"],
            scene["table"]["size"],
            tuple(config.get("scene_colors", {}).get("table", (0.57, 0.48, 0.36))),
        )
        robot_stand = static_box(
            "RobotStand",
            scene["robot_stand"]["center"],
            scene["robot_stand"]["size"],
            tuple(
                config.get("scene_colors", {}).get("robot_stand", (0.30, 0.32, 0.35))
            ),
        )
        ground = static_box(
            "Ground", (0.0, 0.0, -0.01), (8.0, 8.0, 0.02), (0.35, 0.36, 0.38)
        )
        block = DeformableObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cloth",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(block_usd),
                deformable_props=NewtonDeformableBodyPropertiesCfg(),
                physics_material=NewtonSurfaceDeformableBodyMaterialCfg(
                    density=block_cfg["density"],
                    **{
                        k: v
                        for k, v in block_cfg["surface_props"].items()
                        if k not in ("tri_drag", "tri_lift")
                    },
                    particle_radius=block_cfg["particle_radius"],
                ),
            ),
            init_state=DeformableObjectCfg.InitialStateCfg(
                pos=tuple(inputs["state"]["block_center"])
            ),
        )
        light = AssetBaseCfg(
            prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=600.0)
        )

    if cameras:
        for name in config["measurement"]["cameras"]:
            setattr(
                S03SceneCfg,
                f"camera_{name}",
                CameraCfg(
                    # OVRTX requires cameras under the environment namespace.
                    prim_path=f"{{ENV_REGEX_NS}}/Camera_{name}",
                    update_period=0.0,
                    height=height,
                    width=width,
                    data_types=[
                        "rgb",
                        "distance_to_image_plane",
                        "instance_segmentation",
                    ],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=float(focal),
                        horizontal_aperture=aperture,
                        clipping_range=(0.01, 10.0),
                    ),
                    renderer_cfg=OVRTXRendererCfg(colorize_instance_segmentation=False),
                ),
            )
    return S03SceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=True)


# --------------------------------------------------------------------------
# Readings and cameras
# --------------------------------------------------------------------------


def pose_matrix(position, quat_xyzw) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    m = np.eye(4)
    m[:3, :3] = Rotation.from_quat(np.asarray(quat_xyzw)).as_matrix()
    m[:3, 3] = position
    return m


def torch_of(value):
    return value.torch if hasattr(value, "torch") else value


class Readings:
    """Asset-view readers."""

    def __init__(self, scene, config, names):
        self.robot = scene["robot"]
        self.block = scene["block"]
        self.joint_ids = [self.robot.joint_names.index(n) for n in names]
        self.body_ids = {
            name: self.robot.body_names.index(name)
            for name in (
                config["robot"]["tcp_body"],
                "fr3_leftfinger",
                "fr3_rightfinger",
            )
        }
        self.tcp_body = config["robot"]["tcp_body"]
        self.tcp_offset = np.asarray(config["robot"]["tcp_offset"])

    def qpos(self):
        return torch_of(self.robot.data.joint_pos)[0, self.joint_ids].cpu().numpy()

    def _body(self, name):
        pose = (
            torch_of(self.robot.data.body_link_pose_w)[0, self.body_ids[name]]
            .cpu()
            .numpy()
        )
        return pose_matrix(pose[:3], pose[3:7])

    def tcp(self):
        m = self._body(self.tcp_body)
        return m[:3, :3] @ self.tcp_offset + m[:3, 3]

    def finger_poses(self):
        return np.stack([self._body("fr3_leftfinger"), self._body("fr3_rightfinger")])

    def particles(self):
        return torch_of(self.block.data.nodal_pos_w)[0].cpu().numpy().copy()


class Cameras:

    def __init__(self, scene, config, device):
        self.scene = scene
        self.names = list(config["measurement"]["cameras"])
        for name, spec in config["measurement"]["cameras"].items():
            scene[f"camera_{name}"].set_world_poses_from_view(
                eyes=torch.tensor([spec["eye"]], device=device),
                targets=torch.tensor([spec["target"]], device=device),
            )
        self.label_of_id = None

    def _camera(self, name):
        return self.scene[f"camera_{name}"]

    def intrinsics(self):
        return {
            n: torch_of(self._camera(n).data.intrinsic_matrices)[0]
            .cpu()
            .numpy()
            .tolist()
            for n in self.names
        }

    def extrinsics(self):
        """Camera-to-world 4x4 in the OpenGL camera convention."""
        result = {}
        for n in self.names:
            data = self._camera(n).data
            pos = torch_of(data.pos_w)[0].cpu().numpy()
            quat = torch_of(data.quat_w_opengl)[0].cpu().numpy()
            result[n] = pose_matrix(pos, quat).tolist()
        return result

    def _label_table(self, info) -> dict[int, int]:
        table = {}
        for key, value in info["idToLabels"].items():
            text = str(value)
            for prim, label in PRIM_LABELS.items():
                if f"/{prim}" in text and (
                    prim != "Robot" or "/RobotStand" not in text
                ):
                    table[int(key)] = LABELS[label]
        return table

    def capture(self, with_depth: bool):
        frames = {}
        for name in self.names:
            camera = self._camera(name)
            out = camera.data.output
            rgb = torch_of(out["rgb"])[0, ..., :3].cpu().numpy()
            frame = {"rgb": rgb.astype(np.uint8)}
            if with_depth:
                frame["depth"] = (
                    torch_of(out["distance_to_image_plane"])[0, ..., 0]
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                native = (
                    torch_of(out["instance_segmentation"])[0, ..., 0]
                    .cpu()
                    .numpy()
                    .astype(np.int64)
                )
                if self.label_of_id is None:
                    info = camera.data.info
                    info = info[0] if isinstance(info, list) else info
                    self.id_to_labels = {
                        int(k): str(v)
                        for k, v in info["instance_segmentation"]["idToLabels"].items()
                    }
                    self.label_of_id = self._label_table(info["instance_segmentation"])
                seg = np.zeros(native.shape, dtype=np.uint8)
                for uid, label in self.label_of_id.items():
                    seg[native == uid] = label
                # Deformable meshes have no shape entry and render as
                # UNLABELLED; every rigid prim carries a class tag, so the
                # UNLABELLED pixels are the soft block (checked by analyze).
                unlabelled = [
                    k for k, v in self.id_to_labels.items() if "UNLABELLED" in v.upper()
                ]
                seg[np.isin(native, unlabelled)] = LABELS["cloth"]
                frame["seg"] = seg
                frame["native_ids"] = np.unique(native)
            frames[name] = frame
        return frames


# --------------------------------------------------------------------------
# Bookkeeping
# --------------------------------------------------------------------------


def utc_stamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def git_state() -> dict:
    def run(*cmd):
        try:
            return subprocess.run(
                ["git", "-C", str(REPO_ROOT), *cmd],
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout.strip()
        except Exception as exc:  # pragma: no cover - diagnostic only
            return f"unavailable: {exc}"

    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty_files": run("status", "--porcelain").splitlines()[:200],
        "diff_sha256": hashlib.sha256(run("diff", "HEAD").encode()).hexdigest(),
    }


def gpu_info() -> list:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except Exception:
        return []
    return [
        dict(zip(("name", "memory", "driver"), [p.strip() for p in line.split(",")]))
        for line in out.strip().splitlines()
    ]


def to_json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (Path, set)):
        return sorted(value) if isinstance(value, set) else str(value)
    raise TypeError(f"Not JSON serializable: {type(value)}")


def write_json(path: Path, data) -> None:
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, default=to_json)
        stream.write("\n")


def downsample(image, factor: int):
    """Every factor-th pixel (rows/columns 0, f, 2f, ...)."""
    return np.ascontiguousarray(image[::factor, ::factor])


def stored_intrinsics(k, factor: int) -> np.ndarray:
    """K of an image built by ``downsample``.

    Stored pixel u is render pixel f*u, whose centre f*u + 0.5 maps to
    stored coordinate u + 0.5/f; hence fx/f and cx/f - 0.5/f + 0.5.
    """
    k = np.asarray(k, dtype=np.float64)
    out = k.copy()
    out[0, 0] = k[0, 0] / factor
    out[1, 1] = k[1, 1] / factor
    out[0, 2] = k[0, 2] / factor - 0.5 / factor + 0.5
    out[1, 2] = k[1, 2] / factor - 0.5 / factor + 0.5
    return out


class VideoWriter:

    def __init__(self, path: Path, fps: int):
        import imageio.v2 as imageio

        path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(
            str(path),
            fps=fps,
            codec="libx264",
            quality=9,
            macro_block_size=16,
            pixelformat="yuv420p",
        )

    def append(self, rgb):
        self.writer.append_data(np.ascontiguousarray(rgb[..., :3]))

    def close(self):
        self.writer.close()


def describe_model():
    """Newton model contents and coupler ownership, for parity checks."""
    from isaaclab_newton.physics import NewtonManager

    model = NewtonManager.backend.model
    solver = NewtonManager._solver
    labels = list(model.body_label)
    shapes = {
        "body": [labels[b] if b >= 0 else "static" for b in model.shape_body.numpy()]
    }
    for name in (
        "shape_material_ke",
        "shape_material_kd",
        "shape_material_mu",
        "shape_margin",
        "shape_gap",
        "shape_flags",
    ):
        if hasattr(model, name):
            shapes[name] = getattr(model, name).numpy().tolist()
    report = {
        "shapes": shapes,
        "body_label": labels,
        "body_mass": model.body_mass.numpy().tolist(),
        "shape_count": int(model.shape_count),
        "particle_count": int(model.particle_count),
        "particle_mass_sum": float(model.particle_mass.numpy().sum()),
        "particle_radius": float(model.particle_radius.numpy().max()),
        "tet_count": int(model.tet_count),
        "tri_count": int(model.tri_count),
        "edge_count": int(model.edge_count),
        "soft_contact_ke": float(model.soft_contact_ke),
        "soft_contact_kd": float(model.soft_contact_kd),
        "soft_contact_mu": float(model.soft_contact_mu),
        "solver": type(solver).__name__,
    }
    return report


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------


def run(args):
    preview_only = getattr(args, "preview_only", False)
    inputs = load_inputs(args)
    config = inputs["config"]
    controller = PickPlaceController(config, inputs["waypoint_s"])
    finger0 = config["robot"]["finger_initial_q"]
    initial_row = np.r_[arm_target(inputs, controller.s), finger0, finger0]
    names = list(config["robot"]["arm_joints"]) + list(config["robot"]["finger_joints"])
    run_id = (
        f"s03_{PLATFORM}_{args.state}_{args.variant}_r{args.repeat:02d}"
        f"_{args.kind}_{utc_stamp()}_{os.urandom(2).hex()}"
    )
    out_root = output_dir("s03", args.out_root)
    out = out_root / "raw_data" / PLATFORM / run_id
    usd_dir = out / "authored_usd"
    usd_dir.mkdir(parents=True, exist_ok=True)
    with_cameras = args.kind == "full" and args.images != "none"
    files = {
        "script": Path(__file__).resolve(),
        "controller_module": EXPERIMENTS / "s03_common.py",
        "franka_usd": FRANKA_USD,
        "franka_usd_source": FRANKA_USD.parents[1] / "source.json",
        **{p.name: p for p in sorted(INPUTS.iterdir()) if p.is_file()},
    }
    import importlib.metadata as md

    manifest = {
        "run_id": run_id,
        "qualification": "native_platform_comparison",
        "control_contract": "native platform control; not matched replay",
        "timing_protocol": {
            "warmup_frames": 30,
            "scope": "synchronized completed outer step; excludes control, readback and capture",
            "window": "episode frames after 30 initial settling frames; no reset",
        },
        "protocol_revision": "dexuni-fixed-five-v1",
        "task": config["task"],
        "subtask": config["subtask"],
        "platform": PLATFORM,
        "scenario": "task",
        "state": inputs["state"],
        "variant": args.variant,
        "variant_settings": config["variants"][args.variant],
        "kind": args.kind,
        "images": args.images if with_cameras else "none",
        "repeat": args.repeat,
        "lift_height": config["path"]["lift_height"],
        "coupling": config["coupling"],
        "joint_mode": config["joint_mode"],
        "requested": {
            "num_substeps": config["solver"]["num_substeps"],
            "vbd_iterations": config["solver"]["vbd_iterations"],
            "use_cuda_graph": config["solver"]["use_cuda_graph"],
        },
        "started_utc": utc_stamp(),
        "files_sha256": {k: sha256_file(v) for k, v in files.items()},
        "robot_urdf_sha256": sha256_file(config["robot"]["urdf"]),
        "python": sys.version,
        "executable": sys.executable,
        "host": platform.node(),
        "gpus": gpu_info(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "repo": git_state(),
        "isaaclab_version": md.version("isaaclab"),
        "isaaclab_newton_version": md.version("isaaclab-newton"),
        "newton_version": md.version("newton"),
        "mujoco_warp_version": md.version("mujoco-warp"),
        "warp_version": wp.__version__,
        "torch_version": torch.__version__,
        "solver_path": config["platform_solvers"][PLATFORM],
        "contact_force": "unavailable (no public per-contact force readout)",
        "joint_torque": "unavailable (not read; drive targets and positions are recorded)",
        "status": "running",
    }
    if preview_only:
        manifest["qualification"] = "scene_preview_only"
        manifest["tests_started"] = False
        manifest.pop("timing_protocol", None)
    log_lines = []

    def log(message):
        print(message, flush=True)
        log_lines.append(message)

    from benchlib.gpu_memory import GpuProcessMemoryTracker

    memory = GpuProcessMemoryTracker(gpu_id=0)
    memory_ok = not preview_only and memory.start()
    columns: dict[str, list] = {}
    image_columns: dict[str, dict[str, list]] = {}
    timing_rows, videos = [], {}
    t_start = time.perf_counter() if not preview_only else 0.0
    try:
        physics = physics_config(config)
        with launch_simulation(physics, args):
            sim = sim_utils.SimulationContext(
                sim_utils.SimulationCfg(
                    dt=config["solver"]["dt"],
                    device=args.device,
                    gravity=(0.0, 0.0, -9.81),
                    physics=physics,
                )
            )
            scene = instantiate(
                scene_config(inputs, initial_row, usd_dir, with_cameras)
            )
            sim.reset()
            readings = Readings(scene, config, names)
            robot = scene["robot"]
            manifest["newton_model"] = describe_model()
            # Width-patch node ids assume the simulated node order equals
            # the mesh order.
            order_error = float(np.abs(readings.particles() - inputs["initial"]).max())
            if order_error > 1e-5:
                raise RuntimeError(
                    f"node order/positions differ from the "
                    f"initial mesh by {order_error:.2e} m"
                )
            q0 = torch.tensor(
                initial_row, device=sim.device, dtype=torch.float32
            ).unsqueeze(0)
            robot.write_joint_position_to_sim_index(
                position=q0, joint_ids=readings.joint_ids
            )
            robot.set_joint_position_target_index(
                target=q0, joint_ids=readings.joint_ids
            )
            scene.write_data_to_sim()
            scene.update(0.0)
            start_error = float(np.abs(readings.particles() - inputs["initial"]).max())
            if start_error > 1e-5:
                raise RuntimeError(
                    f"initial node positions changed by the "
                    f"joint reset ({start_error:.2e} m)"
                )
            cameras = None
            if with_cameras:
                cameras = Cameras(scene, config, sim.device)
                manifest["camera_intrinsics"] = cameras.intrinsics()
                manifest["camera_extrinsics_world_pose"] = cameras.extrinsics()
                manifest["camera_extrinsics_convention"] = (
                    "camera-to-world 4x4 from Camera.data.pos_w and "
                    "quat_w_opengl (xyzw): camera looks along its local -Z, "
                    "+Y up (OpenGL)"
                )
                fps = config["measurement"]["camera_rate_hz"]
                for name in cameras.names:
                    if args.images == "full":
                        # Raw source for the single comparison video.
                        videos[name] = VideoWriter(out / f"rgb_{name}.mp4", fps)
                    image_columns[name] = {
                        "frame": [],
                        "time": [],
                        "rgb": [],
                        "depth": [],
                        "seg": [],
                    }
            init_seconds = (time.perf_counter() if not preview_only else 0.0) - t_start
            memory.sample("init")
            log(
                f"[{run_id}] initialized, particles=" f"{readings.particles().shape[0]}"
            )
            rate = config["trajectory"]["frame_rate"]
            stride = rate // config["measurement"]["camera_rate_hz"]
            frame_dt = 1.0 / rate
            dt = config["solver"]["dt"]
            patches = inputs["patches"]
            start_particles = readings.particles()
            record_states = args.kind == "full"
            factor = int(config["measurement"]["image_downsample"])

            def record(now, target, s_cmd, finger, phase, particles):
                row = {
                    "time": now,
                    "phase": controller.PHASES.index(phase),
                    "particles": particles,
                    "target_q": target,
                    "path_s_cmd": s_cmd,
                    "finger_target": finger,
                    "qpos": readings.qpos(),
                    "tcp": readings.tcp(),
                    "finger_poses": readings.finger_poses(),
                    "corner_height": block_width(particles, patches),
                }
                for key, value in row.items():
                    columns.setdefault(key, []).append(np.array(value, copy=True))

            def capture(frame, now, keyframe):
                depth = not preview_only and (args.images == "full" or keyframe)
                for name, data in cameras.capture(depth).items():
                    if name in videos and frame % stride == 0:
                        videos[name].append(data["rgb"])
                    if depth:
                        col = image_columns[name]
                        col["frame"].append(frame)
                        col["time"].append(now)
                        if name not in videos:  # keyframe runs keep RGB here
                            col["rgb"].append(downsample(data["rgb"], factor))
                        col["depth"].append(downsample(data["depth"], factor))
                        col["seg"].append(downsample(data["seg"], factor))
                        manifest.setdefault(
                            "segmentation_native_ids_seen", set()
                        ).update(int(v) for v in data["native_ids"])

            if record_states:
                record(
                    0.0,
                    initial_row,
                    controller.s,
                    controller.finger,
                    controller.phase,
                    start_particles,
                )
                if cameras:
                    scene.update(0.0)
                    capture(0, 0.0, True)
                    manifest["segmentation_native_to_label"] = cameras.label_of_id
                    if not preview_only:
                        manifest["segmentation_id_to_labels"] = cameras.id_to_labels
            sim_seconds = 0.0
            width = block_width(start_particles, patches)
            frame = 0
            last_frames = controller.max_frames()
            phase = controller.phase
            from isaaclab_newton.physics import NewtonManager
            from isaaclab.physics import PhysicsManager

            simulation_time_origin = PhysicsManager._sim_time

            while frame < last_frames and not controller.finished:
                frame += 1
                now = frame * frame_dt
                previous = phase
                s_cmd, finger, phase = controller.step(now, width)
                target_row = np.r_[arm_target(inputs, s_cmd), finger, finger]
                target = torch.tensor(
                    target_row, device=sim.device, dtype=torch.float32
                ).unsqueeze(0)
                robot.set_joint_position_target_index(
                    target=target, joint_ids=readings.joint_ids
                )
                torch.cuda.synchronize()
                started = time.perf_counter() if not preview_only else 0.0
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(dt)
                torch.cuda.synchronize()
                step = (time.perf_counter() if not preview_only else 0.0) - started
                sim_seconds += step
                readback_started = time.perf_counter() if not preview_only else 0.0
                capture_seconds = 0.0
                particles = readings.particles()
                if record_states:
                    started = time.perf_counter() if not preview_only else 0.0
                    record(now, target_row, s_cmd, finger, phase, particles)
                    last = frame == last_frames or controller.finished
                    if cameras and frame % stride == 0:
                        capture(frame, now, phase != previous or last)
                    elif cameras and (phase != previous or last):
                        capture(frame, now, True)
                    capture_seconds = (
                        time.perf_counter() if not preview_only else 0.0
                    ) - started
                # The closed-loop rule needs the measured width every frame.
                width = block_width(particles, patches)
                timing_rows.append(
                    (
                        frame,
                        now,
                        step,
                        capture_seconds,
                        phase,
                        (time.perf_counter() if not preview_only else 0.0)
                        - readback_started,
                    )
                )
                if frame == 2:
                    manifest["cuda_graph_status_after_first_steps"] = (
                        "captured"
                        if NewtonManager._graph is not None
                        else (
                            "disabled"
                            if not config["solver"]["use_cuda_graph"]
                            else "not captured"
                        )
                    )
                if frame % 30 == 0:
                    memory.sample(f"frame_{frame}")
                    comp = controller.compression()
                    comp_text = "   n/a" if comp is None else f"{100 * comp:5.1f}%"
                    tcp = readings.tcp()
                    log(
                        f"t={now:5.2f} {phase:14s} corner_z={width * 1e3:6.2f}mm "
                        f"comp={comp_text} s={s_cmd * 1e3:6.1f}mm "
                        f"tcp=({tcp[0]:.3f},{tcp[1]:.3f},{tcp[2]:.3f}) "
                        f"zmin={particles[:, 2].min():.4f} "
                        f"finger={readings.qpos()[7] * 1e3:.2f}mm "
                        + (
                            "step=not measured (preview)"
                            if preview_only
                            else f"step={step * 1e3:.2f}ms"
                        )
                    )
                    if not np.isfinite(particles).all():
                        raise FloatingPointError("Non-finite particle state")
            manifest["effective"] = {
                "outer_steps_per_command": 1,
                "simulated_elapsed_s": PhysicsManager._sim_time
                - simulation_time_origin,
                "vbd_iterations": next(
                    entry.solver.iterations
                    for entry in NewtonManager._solver._entries.values()
                    if type(entry.solver).__name__ == "SolverVBD"
                ),
                "num_substeps": int(NewtonManager._num_substeps),
                "solver_dt": float(NewtonManager._solver_dt),
                "cuda_graph_status": (
                    "captured"
                    if NewtonManager._graph is not None
                    else (
                        "disabled"
                        if not config["solver"]["use_cuda_graph"]
                        else "not captured"
                    )
                ),
                "source": "NewtonManager._num_substeps/_solver_dt/_graph "
                "(private attributes; no public accessor)",
            }
            manifest["controller"] = {
                "reference_height": controller.reference_height,
                "grip": "physical contact; all cloth particles dynamic",
                "finger_closed": controller.finger_closed,
                "events": controller.events,
                "finished": controller.finished,
            }
            manifest.update(
                status="completed",
                sim_seconds=sim_seconds,
                init_seconds=init_seconds,
                frames=frame,
            )
            if record_states:
                arrays = {k: np.stack(v) for k, v in columns.items()}
                np.savez_compressed(
                    out / "states.npz",
                    **arrays,
                    joint_names=np.asarray(names),
                    initial_particles=inputs["initial"],
                    corner_ids=patches,
                    triangles=inputs["surface"],
                    tets=inputs["tets"],
                )
            for name, col in image_columns.items():
                if col["frame"]:
                    np.savez_compressed(
                        out / f"images_{name}.npz",
                        frame=np.asarray(col["frame"]),
                        time=np.asarray(col["time"]),
                        rgb=(
                            np.stack(col["rgb"])
                            if col["rgb"]
                            else np.zeros((0, 0, 0, 3), np.uint8)
                        ),
                        depth=np.stack(col["depth"]),
                        seg=np.stack(col["seg"]),
                        K=stored_intrinsics(
                            manifest["camera_intrinsics"][name], factor
                        ),
                        K_render=np.asarray(manifest["camera_intrinsics"][name]),
                        camera_to_world=np.asarray(
                            manifest["camera_extrinsics_world_pose"][name]
                        ),
                    )
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = repr(exc)
        log(traceback.format_exc())
        raise
    finally:
        for writer in videos.values():
            writer.close()
        if "segmentation_native_ids_seen" in manifest:
            manifest["segmentation_native_ids_seen"] = sorted(
                manifest["segmentation_native_ids_seen"]
            )
        manifest["gpu_memory"] = (
            memory.metrics() if memory_ok else {"status": "unavailable"}
        )
        memory.close()
        if preview_only:
            manifest.pop("sim_seconds", None)
            manifest.pop("init_seconds", None)
            manifest["gpu_memory"] = {"status": "not measured (scene preview only)"}
        manifest["finished_utc"] = utc_stamp()
        with open(
            out / ("preview_no_timing.txt" if preview_only else "timing.csv"), "w"
        ) as stream:
            if preview_only:
                stream.write(
                    "Scene preview only: no step timing, task metrics or performance samples were collected.\n"
                )
            else:
                stream.write(
                    "frame,sim_time,step_seconds,capture_seconds,phase,readback_record_capture_seconds\n"
                )
                for row in timing_rows:
                    stream.write(",".join(str(v) for v in row) + "\n")
        write_json(out / "config.json", config)
        write_json(out / "manifest.json", manifest)
        (out / "run.log").write_text("\n".join(log_lines) + "\n")
        print(f"run directory: {out}", flush=True)


if __name__ == "__main__":
    run(args_cli)
