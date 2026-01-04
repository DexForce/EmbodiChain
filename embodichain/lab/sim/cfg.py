# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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

from __future__ import annotations
import os
import numpy as np
import torch

from typing import Sequence, Union, Dict, Literal, List, Any
from dataclasses import field, MISSING

from dexsim.types import (
    PhysicalAttr,
    ActorType,
    AxisArrowType,
    AxisCornerType,
    VoxelConfig,
    SoftBodyAttr,
    SoftBodyMaterialModel,
)
from embodichain.utils import configclass, is_configclass
from embodichain.data.constants import EMBODICHAIN_DEFAULT_DATA_ROOT
from embodichain.data import get_data_path
from embodichain.utils import logger
from embodichain.utils.utility import key_in_nested_dict

from .shapes import ShapeCfg, MeshCfg


@configclass
class PhysicsCfg:
    gravity: np.ndarray = field(default_factory=lambda: np.array([0, 0, -9.81]))
    """Gravity vector for the simulation environment."""

    bounce_threshold: float = 2.0
    """The speed threshold below which collisions will not produce bounce effects."""

    enable_pcm: bool = True
    """Enable persistent contact manifold (PCM) for improved collision handling."""

    enable_tgs: bool = True
    """Enable temporal gauss-seidel (TGS) solver for better stability."""

    enable_ccd: bool = False
    """Enable continuous collision detection (CCD) for fast-moving objects."""

    enable_enhanced_determinism: bool = False
    """Enable enhanced determinism for consistent simulation results."""

    enable_friction_every_iteration: bool = True
    """Enable friction calculations at every solver iteration."""

    length_tolerance: float = 0.05
    """The length tolerance for the simulation.
    
    Note: the larger the tolerance, the faster the simulation will be. 
    """
    speed_tolerance: float = 0.25
    """The speed tolerance for the simulation.
    
    Note: the larger the tolerance, the faster the simulation will be.
    """

    def to_dexsim_args(self) -> Dict[str, Any]:
        """Convert to dexsim physics args dictionary."""
        args = {
            "gravity": self.gravity.tolist(),
            "bounce_threshold": self.bounce_threshold,
            "enable_pcm": self.enable_pcm,
            "enable_tgs": self.enable_tgs,
            "enable_ccd": self.enable_ccd,
            "enable_enhanced_determinism": self.enable_enhanced_determinism,
            "enable_friction_every_iteration": self.enable_friction_every_iteration,
        }
        return args


@configclass
class MarkerCfg:
    """Configuration for visual markers in the simulation.

    This class defines properties for creating visual markers such as coordinate frames,
    lines, and points that can be used for debugging, visualization, or reference purposes
    in the simulation environment.
    """

    name: str = "empty-mesh"
    """Name of the marker for identification purposes."""

    marker_type: Literal["axis", "line", "point"] = "axis"
    """Type of marker to display. Can be 'axis' (3D coordinate frame), 'line', or 'point'. (only axis supported now)"""

    axis_xpos: List[np.ndarray] = None
    """List of 4x4 transformation matrices defining the position and orientation of each axis marker."""

    axis_size: float = 0.002
    """Thickness/size of the axis lines in meters."""

    axis_len: float = 0.005
    """Length of each axis arm in meters."""

    line_color: List[float] = [1, 1, 0, 1.0]
    """RGBA color values for the marker lines. Values should be between 0.0 and 1.0."""

    arrow_type: AxisArrowType = AxisArrowType.CONE
    """Type of arrow head for axis markers (e.g., CONE, ARROW, etc.)."""

    corner_type: AxisCornerType = AxisCornerType.SPHERE
    """Type of corner/joint visualization for axis markers (e.g., SPHERE, CUBE, etc.)."""

    arena_index: int = -1
    """Index of the arena where the marker should be placed. -1 means all arenas."""


@configclass
class GPUMemoryCfg:
    """A gpu memory configuration dataclass that neatly holds all parameters that configure physics GPU memory for simulation"""

    temp_buffer_capacity: int = 2**24
    """Increase this if you get 'PxgPinnedHostLinearMemoryAllocator: overflowing initial allocation size, increase capacity to at least %.' """

    max_rigid_contact_count: int = 2**19
    """Increase this if you get 'Contact buffer overflow detected'"""

    max_rigid_patch_count: int = (
        2**18
    )  # 81920 is DexSim default but most tasks work with 2**18
    """Increase this if you get 'Patch buffer overflow detected'"""

    heap_capacity: int = 2**26

    found_lost_pairs_capacity: int = (
        2**25
    )  # 262144 is DexSim default but most tasks work with 2**25
    found_lost_aggregate_pairs_capacity: int = 2**10
    total_aggregate_pairs_capacity: int = 2**10


@configclass
class RigidBodyAttributesCfg:
    """Physical attributes for rigid bodies.

    There are three parts of attributes that can be set:
    1. The dynamic properties, such as mass, damping, etc.
    2. The collision properties.
    3. The physics material properties.
    """

    mass: float = 1.0
    """Mass of the rigid body in kilograms. 
    
    Set to 0 will use density to calculate mass.
    """

    density: float = 1000.0
    """Density of the rigid body in kg/m^3."""

    angular_damping: float = 0.7
    """Angular damping coefficient."""

    linear_damping: float = 0.7
    """Linear damping coefficient."""

    max_depenetration_velocity: float = 10.0
    """Maximum depenetration velocity."""

    sleep_threshold: float = 0.001
    """Threshold below which the body can go to sleep."""

    min_position_iters: int = 4
    """Minimum position iterations."""

    min_velocity_iters: int = 1
    """Minimum velocity iterations."""

    max_linear_velocity: float = 1e2
    """Maximum linear velocity."""

    max_angular_velocity: float = 1e2
    """Maximum angular velocity."""

    # collision properties.
    enable_ccd: bool = False
    """Enable continuous collision detection (CCD)."""

    contact_offset: float = 0.002
    """Contact offset for collision detection."""

    rest_offset: float = 0.001
    """Rest offset for collision detection."""

    enable_collision: bool = True
    """Enable collision for the rigid body."""

    # physics material properties.
    restitution: float = 0.0
    """Restitution (bounciness) coefficient."""

    dynamic_friction: float = 0.5
    """Dynamic friction coefficient."""

    static_friction: float = 0.5
    """Static friction coefficient."""

    def attr(self) -> PhysicalAttr:
        """Convert to dexsim PhysicalAttr"""
        attr = PhysicalAttr()
        attr.mass = self.mass
        attr.contact_offset = self.contact_offset
        attr.rest_offset = self.rest_offset
        attr.dynamic_friction = self.dynamic_friction
        attr.static_friction = self.static_friction
        attr.angular_damping = self.angular_damping
        attr.linear_damping = self.linear_damping
        attr.sleep_threshold = self.sleep_threshold
        attr.restitution = self.restitution
        attr.enable_ccd = self.enable_ccd
        attr.max_depenetration_velocity = self.max_depenetration_velocity
        attr.min_position_iters = self.min_position_iters
        attr.min_velocity_iters = self.min_velocity_iters
        return attr

    @classmethod
    def from_dict(
        cls, init_dict: Dict[str, Union[str, float, int]]
    ) -> RigidBodyAttributesCfg:
        """Initialize the configuration from a dictionary."""
        cfg = cls()
        for key, value in init_dict.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
            else:
                logger.log_warning(
                    f"Key '{key}' not found in {cfg.__class__.__name__}."
                )
        return cfg


@configclass
class SoftbodyVoxelAttributesCfg:
    # voxel config
    triangle_remesh_resolution: int = 8
    """Resolution to remesh the softbody mesh before building physx collision mesh."""

    triangle_simplify_target: int = 0
    """Simplify mesh faces to target value. Do nothing if this value is zero."""

    # TODO: this value will be automatically computed with simulation_mesh_resolution and mesh scale.
    maximal_edge_length: float = 0
    # """To shorten edges that are too long, additional points get inserted at their center leading to a subdivision of the input mesh. Do nothing if this value is zero."""

    simulation_mesh_resolution: int = 8
    """Resolution to build simulation voxelize textra mesh. This value must be greater than 0."""

    simulation_mesh_output_obj: bool = False
    """Whether to output the simulation mesh as an obj file for debugging."""

    def attr(self) -> VoxelConfig:
        """Convert to dexsim VoxelConfig"""
        attr = VoxelConfig()
        attr.triangle_remesh_resolution = self.triangle_remesh_resolution
        attr.maximal_edge_length = self.maximal_edge_length
        attr.simulation_mesh_resolution = self.simulation_mesh_resolution
        attr.triangle_simplify_target = self.triangle_simplify_target
        return attr


@configclass
class SoftbodyPhysicalAttributesCfg:
    # material properties
    youngs: float = 1e6
    """Young's modulus (higher = stiffer)."""

    poissons: float = 0.45
    """Poisson's ratio (higher = closer to incompressible)."""

    dynamic_friction: float = 0.0
    """Dynamic friction coefficient."""

    elasticity_damping: float = 0.0
    """Elasticity damping factor."""

    # soft body properties
    material_model: SoftBodyMaterialModel = SoftBodyMaterialModel.CO_ROTATIONAL
    """Material constitutive model."""

    # --- Mode / collision switches ---
    enable_kinematic: bool = False
    """If True, (partially) kinematic behavior is enabled."""

    enable_ccd: bool = False
    """Enable continuous collision detection (CCD)."""

    enable_self_collision: bool = False
    """Enable self-collision handling."""

    has_gravity: bool = True
    """Whether the soft body is affected by gravity."""

    # --- Self-collision & simplification parameters ---
    self_collision_stress_tolerance: float = 0.9
    """Stress tolerance threshold for self-collision constraints."""

    collision_mesh_simplification: bool = True
    """Whether to simplify the collision mesh for self-collision."""

    self_collision_filter_distance: float = 0.1
    """Distance threshold below which vertex pairs may be filtered from self-collision checks."""

    # --- Damping, sleep & settling ---
    vertex_velocity_damping: float = 0.005
    """Per-vertex velocity damping."""

    linear_damping: float = 0.0
    """Global linear damping applied to the soft body."""

    sleep_threshold: float = 0.05
    """Velocity/energy threshold below which the soft body can go to sleep."""

    settling_threshold: float = 0.1
    """Threshold used to decide convergence/settling state."""

    settling_damping: float = 10.0
    """Additional damping applied during settling phase."""

    # --- Mass / density & velocity limits ---
    mass: float = -1.0
    """Total mass of the soft body. If set to a negative value, density will be used to compute mass."""

    density: float = 1000.0
    """Material density in kg/m^3."""

    max_depenetration_velocity: float = 1e6
    """Maximum velocity used to resolve penetrations. Must be larger than zero."""

    max_velocity: float = 100
    """Clamp for linear (or vertex) velocity. If set to zero, the limit is ignored."""

    # --- Solver iteration counts ---
    min_position_iters: int = 4
    """Minimum solver iterations for position correction."""

    min_velocity_iters: int = 1
    """Minimum solver iterations for velocity updates."""

    def attr(self) -> SoftBodyAttr:
        attr = SoftBodyAttr()
        attr.youngs = self.youngs
        attr.poissons = self.poissons
        attr.dynamic_friction = self.dynamic_friction
        attr.elasticity_damping = self.elasticity_damping
        attr.material_model = self.material_model
        attr.enable_kinematic = self.enable_kinematic
        attr.enable_ccd = self.enable_ccd
        attr.enable_self_collision = self.enable_self_collision
        attr.has_gravity = self.has_gravity
        attr.self_collision_stress_tolerance = self.self_collision_stress_tolerance
        attr.collision_mesh_simplification = self.collision_mesh_simplification
        attr.vertex_velocity_damping = self.vertex_velocity_damping
        attr.mass = self.mass
        attr.density = self.density
        attr.max_depenetration_velocity = self.max_depenetration_velocity
        attr.max_velocity = self.max_velocity
        attr.self_collision_filter_distance = self.self_collision_filter_distance
        attr.linear_damping = self.linear_damping
        attr.sleep_threshold = self.sleep_threshold
        attr.settling_threshold = self.settling_threshold
        attr.settling_damping = self.settling_damping
        attr.min_position_iters = self.min_position_iters
        attr.min_velocity_iters = self.min_velocity_iters
        return attr


@configclass
class JointDrivePropertiesCfg:
    """Properties to define the drive mechanism of a joint."""

    drive_type: Literal["force", "acceleration"] = "force"
    """Joint drive type to apply.

    If the drive type is "force", then the joint is driven by a force and the acceleration is computed based on the force applied.
    If the drive type is "acceleration", then the joint is driven by an acceleration and the force is computed based on the acceleration applied.
    """

    stiffness: Union[Dict[str, float], float] = 1e4
    """Stiffness of the joint drive.

    The unit depends on the joint model:

    * For linear joints, the unit is kg-m/s^2 (N/m).
    * For angular joints, the unit is kg-m^2/s^2/rad (N-m/rad).
    """

    damping: Union[Dict[str, float], float] = 1e3
    """Damping of the joint drive.

    The unit depends on the joint model:

    * For linear joints, the unit is kg-m/s (N-s/m).
    * For angular joints, the unit is kg-m^2/s/rad (N-m-s/rad).
    """

    max_effort: Union[Dict[str, float], float] = 1e10
    """Maximum effort that can be applied to the joint (in kg-m^2/s^2)."""

    max_velocity: Union[Dict[str, float], float] = 1e10
    """Maximum velocity that the joint can reach (in rad/s or m/s).

    For linear joints, this is the maximum linear velocity with unit m/s.
    For angular joints, this is the maximum angular velocity with unit rad/s.
    """

    friction: Union[Dict[str, float], float] = 0.0
    """Friction coefficient of the joint"""

    @classmethod
    def from_dict(
        cls, init_dict: Dict[str, Union[str, float, int]]
    ) -> JointDrivePropertiesCfg:
        """Initialize the configuration from a dictionary."""
        cfg = cls()
        for key, value in init_dict.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
            else:
                logger.log_warning(
                    f"Key '{key}' not found in {cfg.__class__.__name__}."
                )
        return cfg


@configclass
class ObjectBaseCfg:
    """Base configuration for an asset in the simulation.

    This class defines the basic properties of an asset, such as its type, initial state, and collision group.
    It is used as a base class for specific asset configurations.
    """

    uid: Union[str, None] = None

    init_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Position of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""

    init_rot: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Euler angles (in degree) of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""

    init_local_pose: np.ndarray | None = None
    """4x4 transformation matrix of the root in local frame. If specified, it will override init_pos and init_rot."""

    @classmethod
    def from_dict(cls, init_dict: Dict[str, Union[str, float, tuple]]) -> ObjectBaseCfg:
        """Initialize the configuration from a dictionary."""
        cfg = cls()  # Create a new instance of the class (cls)
        for key, value in init_dict.items():
            if hasattr(cfg, key):
                attr = getattr(cfg, key)
                if is_configclass(attr):
                    setattr(
                        cfg, key, attr.from_dict(value)
                    )  # Call from_dict on the attribute
                else:
                    setattr(cfg, key, value)
            else:
                logger.log_warning(
                    f"Key '{key}' not found in {cfg.__class__.__name__}."
                )

        # Automatically infer init_local_pose if not provided
        if cfg.init_local_pose is None:
            # If only init_pos or init_rot are provided, generate the 4x4 pose matrix
            from scipy.spatial.transform import Rotation as R

            T = np.eye(4)
            T[:3, 3] = np.array(cfg.init_pos)
            T[:3, :3] = R.from_euler("xyz", np.deg2rad(cfg.init_rot)).as_matrix()
            cfg.init_local_pose = T
        else:
            # If only init_local_pose is provided, extract init_pos and init_rot
            from scipy.spatial.transform import Rotation as R

            T = np.array(cfg.init_local_pose)
            cfg.init_pos = tuple(T[:3, 3])
            cfg.init_rot = tuple(R.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True))

        return cfg


@configclass
class LightCfg(ObjectBaseCfg):
    """Configuration for a light asset in the simulation.

    This class extends the base asset configuration to include specific properties for lights,
    """

    # TODO: to be added more light type, such as spot, sun, etc.
    light_type: Literal["point"] = "point"

    color: tuple[float, float, float] = (1.0, 1.0, 1.0)

    intensity: float = 50.0
    """Intensity of the light source with unit of watts/m^2."""

    radius: float = 1e2
    """Falloff of the light, only used for point light."""


@configclass
class RigidObjectCfg(ObjectBaseCfg):
    """Configuration for a rigid body asset in the simulation.

    This class extends the base asset configuration to include specific properties for rigid bodies,
    such as physical attributes and collision group.
    """

    shape: ShapeCfg = ShapeCfg()
    """Shape configuration for the rigid body. """

    # TODO: supoort basic primitive shapes, such as box, sphere, etc cfg and spawn method.

    attrs: RigidBodyAttributesCfg = RigidBodyAttributesCfg()

    body_type: Literal["dynamic", "kinematic", "static"] = "dynamic"

    max_convex_hull_num: int = 1
    """The maximum number of convex hulls that will be created for the rigid body.

    If `max_convex_hull_num` is set to larger than 1, the rigid body will be decomposed into multiple convex hulls using coacd alogorithm.
    Reference: https://github.com/SarahWeiii/CoACD
    """

    body_scale: Union[tuple, list] = (1.0, 1.0, 1.0)
    """Scale of the rigid body in the simulation world frame."""

    def to_dexsim_body_type(self) -> ActorType:
        """Convert the body type to dexsim ActorType."""
        if self.body_type == "dynamic":
            return ActorType.DYNAMIC
        elif self.body_type == "kinematic":
            return ActorType.KINEMATIC
        elif self.body_type == "static":
            return ActorType.STATIC
        else:
            logger.log_error(
                f"Invalid body type '{self.body_type}' specified. Must be one of 'dynamic', 'kinematic', or 'static'."
            )


@configclass
class SoftObjectCfg(ObjectBaseCfg):
    """Configuration for a soft body asset in the simulation.

    This class extends the base asset configuration to include specific properties for soft bodies,
    such as physical attributes and collision group.
    """

    voxel_attr: SoftbodyVoxelAttributesCfg = SoftbodyVoxelAttributesCfg()
    """Tetra mesh voxelization attributes for the soft body."""

    physical_attr: SoftbodyPhysicalAttributesCfg = SoftbodyPhysicalAttributesCfg()
    """Physical attributes for the soft body."""

    shape: MeshCfg = MeshCfg()
    """Mesh configuration for the soft body."""


@configclass
class RigidObjectGroupCfg:
    """Configuration for a rigid object group asset in the simulation.

    Rigid object groups can be initialized from multiple rigid object configurations specified in a folder.
    If `folder_path` is specified, user should provide a RigidObjectCfg in `rigid_objects` as a template configuration for
    all objects in the group.

    For example:
    ```python
    rigid_object_group: RigidObjectGroupCfg(
        folder_path="path/to/folder",
        max_num=5,
        rigid_objects={
            "template_obj": RigidObjectCfg(
                shape=MeshCfg(
                    fpath="",  # fpath will be ignored when folder_path is specified
                ),
                body_type="dynamic",
            )
        }
    )
    """

    uid: Union[str, None] = None

    rigid_objects: Dict[str, RigidObjectCfg] = MISSING
    """Configuration for the rigid objects in the group."""

    body_type: Literal["dynamic", "kinematic"] = "dynamic"
    """Body type for all rigid objects in the group. """

    folder_path: str | None = None
    """Path to the folder containing the rigid object assets.
    
    This is used to initialize multiple rigid object configurations from a folder.
    """

    max_num: int = 1
    """Maximum number of rigid objects to initialize from the folder.
    
    This is only used when `folder_path` is specified.
    """

    ext: str = ".obj"
    """File extension for the rigid object assets.
    
    This is only used when `folder_path` is specified.
    """

    @classmethod
    def from_dict(cls, init_dict: Dict[str, Any]) -> RigidObjectGroupCfg:
        """Initialize the configuration from a dictionary."""
        cfg = cls()
        for key, value in init_dict.items():
            if hasattr(cfg, key):
                attr = getattr(cfg, key)
                if is_configclass(attr):
                    setattr(
                        cfg, key, attr.from_dict(value)
                    )  # Call from_dict on the attribute
                elif key == "rigid_objects" and "folder_path" not in init_dict:
                    rigid_objects_cfg = {}
                    for obj_name, obj_cfg in value.items():
                        rigid_objects_cfg[obj_name] = RigidObjectCfg.from_dict(obj_cfg)
                    setattr(cfg, key, rigid_objects_cfg)
                elif key == "rigid_objects" and "folder_path" in init_dict:
                    folder_path = init_dict["folder_path"]
                    max_num = init_dict.get("max_num", 1)
                    rigid_objects_cfg = {}
                    if os.path.exists(folder_path) and os.path.isdir(folder_path):
                        files = os.listdir(folder_path)
                        files = [f for f in files if f.endswith(cfg.ext)]
                        # select files up to max_num
                        n_file = len(files)
                        select_files = []
                        for i in range(max_num):
                            select_files.append(files[i % n_file])

                        for i, file_name in enumerate(select_files):
                            file_path = os.path.join(folder_path, file_name)
                            rigid_obj_cfg: RigidObjectCfg = RigidObjectCfg.from_dict(
                                list(init_dict["rigid_objects"].values())[0]
                            )
                            rigid_obj_cfg.uid = f"{cfg.uid}_obj_{i}"
                            rigid_obj_cfg.shape.fpath = file_path
                            rigid_objects_cfg[rigid_obj_cfg.uid] = rigid_obj_cfg
                        setattr(cfg, "rigid_objects", rigid_objects_cfg)
                    else:
                        logger.log_error(
                            f"Folder '{folder_path}' does not exist or is not a directory."
                        )
                else:
                    setattr(cfg, key, value)
            else:
                logger.log_warning(
                    f"Key '{key}' not found in {cfg.__class__.__name__}."
                )
        return cfg


@configclass
class URDFCfg:
    """Standalone configuration class for URDF assembly."""

    components: Dict[str, Dict[str, Union[str, Dict, np.ndarray]]] = field(
        default_factory=dict
    )
    """Dictionary of robot components to be assembled."""

    sensors: Dict[str, Dict[str, Union[str, np.ndarray]]] = field(default_factory=dict)
    """Dictionary of sensors to be attached to the robot."""

    use_signature_check: bool = True
    """Whether to use signature check when merging URDFs."""

    base_link_name: str = "base_link"
    """Name of the base link in the assembled robot."""

    fpath: str | None = None
    """Full output file path for the assembled URDF. If specified, overrides fname and fpath_prefix."""

    fname: str | None = None
    """Name used for output file and directory. If not specified, auto-generated from component names."""

    fpath_prefix: str = EMBODICHAIN_DEFAULT_DATA_ROOT + "/assembled"
    """Output directory prefix for the assembled URDF file."""

    def __init__(
        self,
        components: list[dict[str, str | np.ndarray]] | None = None,
        sensors: dict[str, dict[str, str | np.ndarray]] | None = None,
        fpath: str | None = None,
        fname: str | None = None,
        fpath_prefix: str = EMBODICHAIN_DEFAULT_DATA_ROOT + "/assembled",
        use_signature_check: bool = True,
        base_link_name: str = "base_link",
    ):
        """
        Initialize URDFCfg with optional list of components and output path settings.

        Args:
            components (list[dict[str, str | np.ndarray]] | None): List of component configurations. Each dict should contain:
                - 'component_type' (str): The type/name of the component (e.g., 'chassis', 'arm', 'hand').
                - 'urdf_path' (str): Path to the component's URDF file.
                - 'transform' (np.ndarray | None): 4x4 transformation matrix (optional).
                - Additional params can be included as extra keys.
            sensors (dict[str, dict[str, str | np.ndarray]] | None): Sensor configurations for the robot.
            fpath (str | None): Full output file path for the assembled URDF. If specified, overrides fname and fpath_prefix.
            fname (str | None): Name used for output file and directory. If not specified, auto-generated from component names.
            fpath_prefix (str): Output directory prefix for the assembled URDF file.
            use_signature_check (bool): Whether to use signature check when merging URDFs.
            base_link_name (str): Name of the base link in the assembled robot.
        """
        self.components = {}
        self.sensors = sensors or {}
        self.fpath = fpath
        self.use_signature_check = use_signature_check
        self.base_link_name = base_link_name
        self.fname = fname
        self.fpath_prefix = fpath_prefix

        # Auto-add components if provided
        if components:
            for comp_config in components:
                if not isinstance(comp_config, dict):
                    logger.log_error(
                        f"Component configuration must be a dict, got {type(comp_config)}"
                    )
                    continue

                # Extract required fields
                component_type = comp_config.get("component_type")
                urdf_path = comp_config.get("urdf_path")

                if not component_type or not urdf_path:
                    logger.log_error(
                        f"Component configuration must contain 'component_type' and 'urdf_path', got {comp_config}"
                    )
                    continue

                # Extract optional fields
                transform = comp_config.get("transform", np.eye(4))

                # Extract additional params (exclude known keys)
                params = {
                    k: v
                    for k, v in comp_config.items()
                    if k not in ["component_type", "urdf_path", "transform"]
                }

                # Add the component
                self.add_component(component_type, urdf_path, transform, **params)

        if sensors is not None:
            if not isinstance(sensors, list):
                logger.log_error(
                    f"sensors must be a list of dicts, got {type(sensors)}"
                )
                self.sensors = []
            else:
                # Optionally check each sensor dict
                valid_sensors = []
                for sensor_config in sensors:
                    if not isinstance(sensor_config, dict):
                        logger.log_error(
                            f"Sensor configuration must be a dict, got {type(sensor_config)}"
                        )
                        continue
                    sensor_name = sensor_config.get("sensor_name")
                    if not sensor_name:
                        logger.log_error(
                            f"Sensor configuration must contain 'sensor_name', got {sensor_config}"
                        )
                        continue
                    valid_sensors.append(sensor_config)
                self.sensors = valid_sensors

    def set_urdf(self, urdf_path: str) -> "URDFCfg":
        """Directly specify a single URDF file for the robot, compatible with the single-URDF robot case.

        Args:
            urdf_path (str): Path to the robot's URDF file.

        Returns:
            URDFCfg: Returns self to allow method chaining.
        """
        self.components.clear()
        urdf_file = os.path.splitext(os.path.basename(urdf_path))[0]
        self.components[urdf_file] = {
            "urdf_path": urdf_path,
            "transform": None,
            "params": {},
        }
        self.fpath = urdf_path
        return self

    def add_component(
        self,
        component_type: str,
        urdf_path: str,
        transform: np.ndarray | None = None,
        **params,
    ) -> URDFCfg:
        """Add a robot component to the assembly configuration.

        Args:
            component_type (str): The type/name of the component. Should be one of SUPPORTED_COMPONENTS
                (e.g., 'chassis', 'torso', 'head', 'left_arm', 'right_hand', 'arm', 'hand', etc.).
            urdf_path (str): Path to the component's URDF file.
            transform (np.ndarray | None): 4x4 transformation matrix for the component in the robot frame (default: None).
            **params: Additional keyword parameters for the component (e.g., color, material, etc.).

        Returns:
            URDFCfg: Returns self to allow method chaining.
        """
        if urdf_path:
            if not os.path.exists(urdf_path):
                urdf_path_candidate = get_data_path(urdf_path)
                if os.path.exists(urdf_path_candidate):
                    urdf_path = urdf_path_candidate
                else:
                    logger.log_error(f"URDF path '{urdf_path}' does not exist.")
                    raise FileNotFoundError(f"URDF path '{urdf_path}' does not exist.")

        if transform is None:
            transform = np.eye(4)

        self.components[component_type] = {
            "urdf_path": urdf_path,
            "transform": np.array(transform),
            "params": params,
        }

        if self.fname:
            self.fpath = f"{self.fpath_prefix}/{self.fname}/{self.fname}.urdf"
        else:
            # Update output_path to use all component urdf file names joined by underscores as directory
            if len(self.components) == 1:
                # Only one component, use its urdf file name
                urdf_file = os.path.splitext(os.path.basename(urdf_path))[0]
                name = urdf_file
            else:
                # Multiple components, join all urdf file names
                urdf_files = [
                    os.path.splitext(os.path.basename(v["urdf_path"]))[0]
                    for v in self.components.values()
                ]
                name = "_".join(urdf_files)
            self.fpath = f"{self.fpath_prefix}/{name}/{name}.urdf"

        return self

    def add_sensor(self, sensor_name: str, **sensor_config) -> URDFCfg:
        """Add a sensor to the robot configuration.

        Args:
            sensor_name (str): The name of the sensor.
            **sensor_config: Additional configuration parameters for the sensor.

        Returns:
            URDFCfg: Returns self to allow method chaining.
        """
        self.sensors.append({"sensor_name": sensor_name, **sensor_config})
        return self

    def assemble_urdf(self) -> str:
        """Assemble URDF files for the robot based on the configuration.

        Returns:
            str: The path to the resulting (possibly merged) URDF file.
        """
        components = list(self.components.items())
        # If there is only one component, return its URDF path directly.
        if len(components) == 1:
            _, comp_config = components[0]
            return comp_config["urdf_path"]

        from embodichain.toolkits.urdf_assembly import URDFAssemblyManager

        # If there are multiple components, merge them into a single URDF file.
        manager = URDFAssemblyManager()
        manager.base_link_name = self.base_link_name
        for comp_type, comp_config in components:
            params = comp_config.get("params", {})
            success = manager.add_component(
                comp_type,
                comp_config["urdf_path"],
                comp_config.get("transform"),
                **params,
            )
            if not success:
                logger.log_error(
                    f"Failed to add component '{comp_type}' with config: {comp_config}"
                )

        for sensor in self.sensors:
            manager.attach_sensor(
                sensor_name=sensor.get("sensor_name"),
                sensor_source=sensor.get("sensor_source"),
                parent_component=sensor.get("parent_component"),
                parent_link=sensor.get("parent_link"),
                sensor_type=sensor.get("sensor_type"),
                **{
                    k: v
                    for k, v in sensor.items()
                    if k
                    not in [
                        "sensor_name",
                        "sensor_source",
                        "parent_component",
                        "parent_link",
                        "sensor_type",
                    ]
                },
            )

        try:
            # Merge all added components into a single URDF file at the specified output path.
            merged_urdf_xml = manager.merge_urdfs(self.fpath, self.use_signature_check)
        except Exception as e:
            logger.log_error(f"URDF merge failed: {e}")

        return self.fpath

    @classmethod
    def from_dict(cls, init_dict: Dict) -> "URDFCfg":
        if isinstance(init_dict, cls):
            return init_dict
        components = init_dict.get("components", None)
        if isinstance(components, dict):
            components = [{"component_type": k, **v} for k, v in components.items()]
        sensors = init_dict.get("sensors", None)
        fpath = init_dict.get("fpath", None)
        use_signature_check = init_dict.get("use_signature_check", True)
        base_link_name = init_dict.get("base_link_name", "base_link")
        return cls(
            components=components,
            sensors=sensors,
            fpath=fpath,
            use_signature_check=use_signature_check,
            base_link_name=base_link_name,
        )


@configclass
class ArticulationCfg(ObjectBaseCfg):
    """Configuration for an articulation asset in the simulation.

    This class extends the base asset configuration to include specific properties for articulations,
    such as joint drive properties, physical attributes.
    """

    fpath: str = None
    """Path to the articulation asset file."""

    drive_pros: JointDrivePropertiesCfg = JointDrivePropertiesCfg()
    """Properties to define the drive mechanism of a joint."""

    body_scale: Union[tuple, list] = (1.0, 1.0, 1.0)
    """Scale of the articulation in the simulation world frame."""

    attrs: RigidBodyAttributesCfg = RigidBodyAttributesCfg()
    """Physical attributes for all links . """

    fix_base: bool = True
    """Whether to fix the base of the articulation.

    Set to True for articulations that should not move, such as a fixed base robot arm or a door.
    Set to False for articulations that should move freely, such as a mobile robot or a humanoid robot.
    """

    disable_self_collision: bool = True
    """Whether to enable or disable self-collisions."""

    init_qpos: Union[torch.Tensor, np.ndarray, Sequence[float]] = None
    """Initial joint positions of the articulation. 
    
    If None, the joint positions will be set to zero.
    If provided, it should be a array of shape (num_joints,).
    """

    sleep_threshold: float = 0.005
    """Energy below which the articulation may go to sleep. Range: [0, max_float32]"""

    min_position_iters: int = 4
    """Number of position iterations the solver should perform for this articulation. Range: [1,255]."""

    min_velocity_iters: int = 1
    """Number of velocity iterations the solver should perform for this articulation. Range: [0,255]."""

    build_pk_chain: bool = True
    """Whether to build pytorch-kinematics chain for forward kinematics and jacobian computation."""

    compute_uv: bool = False
    """Whether to compute the UV mapping for the articulation link.
    
    Currently, the uv mapping is computed for each link with projection uv mapping method.
    """


@configclass
class RobotCfg(ArticulationCfg):
    from embodichain.lab.sim.solvers import SolverCfg

    """Configuration for a robot asset in the simulation.
    """

    control_parts: Dict[str, List[str]] | None = None
    """Control parts is the mapping from part name to joint names.

    For example, {'left_arm': ['joint1', 'joint2'], 'right_arm': ['joint3', 'joint4']}
    If no control part is specified, the robot will use all joints as a single control part.

    Note: 
        - if `control_parts` is specified, `solver_cfg` must be a dict with part names as
            keys corresponding to the control parts name.
        - The joint names in the control parts support regular expressions, e.g., 'joint[1-6]'.
            After initialization of robot, the names will be expanded to a list of full joint names.
        - `Robot` is a derived class of `Articulation`, with control parts support. So the `drive_pros`
            in `ArticulationCfg` can use control part as key to specify the corresponding joint drive properties, 
            which will be overridden if these joint names are already specified.
    """

    urdf_cfg: URDFCfg | None = None
    """URDF assembly configuration which allows for assembling a robot from multiple URDF components.
    """

    # TODO: how to support one solver for multiple parts?
    solver_cfg: Union[SolverCfg, Dict[str, SolverCfg], None] = None
    """Solver is used to compute forward and inverse kinematics for the robot.
    """

    @classmethod
    def from_dict(cls, init_dict: Dict[str, Union[str, float, tuple]]) -> RobotCfg:
        """Initialize the configuration from a dictionary."""
        if isinstance(init_dict, cls):
            return init_dict

        import importlib

        solver_module = importlib.import_module("embodichain.lab.sim.solvers")

        cfg = cls()  # Create a new instance of the class (cls)
        for key, value in init_dict.items():
            if hasattr(cfg, key):
                attr = getattr(cfg, key)
                if key == "urdf_cfg":
                    from embodichain.lab.sim.cfg import URDFCfg

                    setattr(cfg, key, URDFCfg.from_dict(value))
                elif key == "fpath":
                    setattr(cfg, key, get_data_path(value))
                elif is_configclass(attr):
                    setattr(
                        cfg, key, attr.from_dict(value)
                    )  # Call from_dict on the attribute
                elif isinstance(value, dict) and "class_type" in value:
                    setattr(
                        cfg,
                        key,
                        getattr(solver_module, f"{value['class_type']}Cfg").from_dict(
                            value
                        ),
                    )
                elif isinstance(value, dict) and key_in_nested_dict(
                    value, "class_type"
                ):
                    setattr(
                        cfg,
                        key,
                        {
                            k: getattr(
                                solver_module, f"{v['class_type']}Cfg"
                            ).from_dict(v)
                            for k, v in value.items()
                        },
                    )

                else:
                    setattr(cfg, key, value)
            else:
                logger.log_warning(
                    f"Key '{key}' not found in {cfg.__class__.__name__}."
                )
        return cfg

    def build_pk_serial_chain(
        self, device: torch.device = torch.device("cpu"), **kwargs
    ) -> Dict[str, "pk.SerialChain"]:
        """Build the serial chain from the URDF file.

        Note:
            This method is usually used in imitation dataset saving (compute eef pose from qpos using FK)
            and model training (provide a differentiable FK layer or loss computation).

        Args:
            device (torch.device): The device to which the chain will be moved. Defaults to CPU.
            **kwargs: Additional arguments for building the serial chain.

        Returns:
            Dict[str, pk.SerialChain]: The serial chain of the robot for specified control part.
        """
        return {}
