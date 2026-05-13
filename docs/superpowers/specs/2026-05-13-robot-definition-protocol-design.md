# Robot Definition Protocol Design

**Date**: 2026-05-13
**Status**: Approved
**Scope**: `embodichain/lab/sim/robots/`

## Problem

Adding a new robot to EmbodiChain requires significant boilerplate and an inconsistent process:

- Each robot reimplements `from_dict`, `_build_default_cfgs`, `_build_default_physics_cfgs`, `build_pk_serial_chain`, and serialization methods.
- CobotMagic (~200 LOC, single file) and DexforceW1 (~750 LOC, 4-file package) follow different structural patterns with no shared contract.
- External users must understand the full class hierarchy and import paths to define a robot.
- No central registry to discover or look up available robots.

## Goal

1. Define a **standard protocol** that every specific robot must implement.
2. **Reduce complexity** — minimize boilerplate, make the common path declarative.
3. Maintain **full backward compatibility** with existing `SimulationManager.add_robot(cfg: RobotCfg)`.

## Design

### RobotDef Protocol

A `RobotDef` protocol defines the standard contract for robot definitions. It specifies the data every robot must provide:

```python
# embodichain/lab/sim/robots/protocol.py

@runtime_checkable
class RobotDef(Protocol):
    """Standard protocol that every robot definition must satisfy.

    Simple robots declare data as class-level fields.
    Complex robots with variants use @property methods that compute
    values based on constructor parameters.
    """

    @property
    def name(self) -> str: ...

    @property
    def urdf_cfg(self) -> URDFCfg: ...

    @property
    def control_parts(self) -> Dict[str, List[str]]: ...

    @property
    def solver_cfg(self) -> Dict[str, SolverCfg]: ...

    @property
    def drive_pros(self) -> JointDrivePropertiesCfg: ...

    @property
    def attrs(self) -> RigidBodyAttributesCfg: ...

    def build_pk_serial_chain(
        self, device: torch.device
    ) -> Dict[str, "pk.SerialChain"]: ...

    def build_cfg(self, **overrides) -> RobotCfg:
        """Convert this definition into a RobotCfg for the spawner.

        Default implementation:
        1. Create RobotCfg(), populate with protocol properties.
        2. Delegate to merge_robot_cfg for user overrides.
        """
        ...
```

### Bridge to RobotCfg

The `build_cfg(**overrides) -> RobotCfg` method is the bridge between `RobotDef` and the existing simulation infrastructure. Its generic default implementation:

1. Creates a `RobotCfg()` instance.
2. Populates it with the protocol's properties (`urdf_cfg`, `control_parts`, `solver_cfg`, `drive_pros`, `attrs`).
3. Copies any extra protocol fields (e.g., `min_position_iters`, `min_velocity_iters`).
4. Calls the existing `merge_robot_cfg(cfg, overrides)` for user-provided overrides (`uid`, `init_pos`, custom drive properties, etc.).

`SimulationManager.add_robot(cfg: RobotCfg)` is completely unchanged.

### Simple Robot Example — CobotMagic

Simple robots with no variants use flat class-level declarations:

```python
@register_robot("CobotMagic")
class CobotMagicDef:
    """CobotMagic dual-arm robot definition."""

    name: str = "CobotMagic"

    urdf_cfg: URDFCfg = URDFCfg(components=[
        {"component_type": "left_arm",
         "urdf_path": get_data_path("CobotMagicArm/CobotMagicWithGripperV100.urdf"),
         "transform": np.array([
             [1,0,0,0.233], [0,1,0,0.300],
             [0,0,1,0],     [0,0,0,1]])},
        {"component_type": "right_arm",
         "urdf_path": get_data_path("CobotMagicArm/CobotMagicWithGripperV100.urdf"),
         "transform": np.array([
             [1,0,0,0.233], [0,1,0,-0.300],
             [0,0,1,0],     [0,0,0,1]])},
    ])

    control_parts: Dict[str, List[str]] = {
        "left_arm": ["LEFT_JOINT[1-6]"],
        "left_eef": ["LEFT_JOINT[7-8]"],
        "right_arm": ["RIGHT_JOINT[1-6]"],
        "right_eef": ["RIGHT_JOINT[7-8]"],
    }

    solver_cfg: Dict[str, SolverCfg] = {
        "left_arm": OPWSolverCfg(
            end_link_name="left_link6",
            root_link_name="left_arm_base",
            tcp=np.array([[-1,0,0,0],[0,-1,0,0],[0,0,1,0.143],[0,0,0,1]])),
        "right_arm": OPWSolverCfg(
            end_link_name="right_link6",
            root_link_name="right_arm_base",
            tcp=np.array([[-1,0,0,0],[0,-1,0,0],[0,0,1,0.143],[0,0,0,1]])),
    }

    drive_pros: JointDrivePropertiesCfg = JointDrivePropertiesCfg(
        stiffness={"LEFT_JOINT[1-6]": 7e4, "RIGHT_JOINT[1-6]": 7e4,
                   "LEFT_JOINT[7-8]": 3e2, "RIGHT_JOINT[7-8]": 3e2},
        damping={"LEFT_JOINT[1-6]": 1e3, "RIGHT_JOINT[1-6]": 1e3,
                 "LEFT_JOINT[7-8]": 3e1, "RIGHT_JOINT[7-8]": 3e1},
        max_effort={"LEFT_JOINT[1-6]": 3e6, "RIGHT_JOINT[1-6]": 3e6,
                    "LEFT_JOINT[7-8]": 3e3, "RIGHT_JOINT[7-8]": 3e3},
    )

    attrs: RigidBodyAttributesCfg = RigidBodyAttributesCfg(
        mass=0.1, static_friction=0.95, dynamic_friction=0.9,
        linear_damping=0.7, angular_damping=0.7,
        contact_offset=0.001, rest_offset=0.001,
        restitution=0.01, max_depenetration_velocity=1e1,
    )

    min_position_iters: int = 8
    min_velocity_iters: int = 2

    def build_pk_serial_chain(self, device) -> Dict[str, "pk.SerialChain"]:
        # Robot-specific PK chain construction
        ...

    def build_cfg(self, **overrides) -> RobotCfg:
        # Generic implementation from base
        ...
```

**Reduction**: ~200 LOC (with custom from_dict, _build_default_cfgs, manual attribute setting) -> ~50 LOC of pure declarations.

### Complex Robot Example — DexforceW1

Complex robots with variants (arm_kind x hand_brand x version) use `@property` methods that compute values based on constructor parameters:

```python
@register_robot("DexforceW1")
class DexforceW1Def:
    """DexforceW1 humanoid robot definition with variant support."""

    name: str = "DexforceW1"

    # Variant parameters (set at construction)
    version: DexforceW1Version = DexforceW1Version.V021
    arm_kind: DexforceW1ArmKind = DexforceW1ArmKind.ANTHROPOMORPHIC
    hand_types: Dict[DexforceW1ArmSide, DexforceW1HandBrand] = None
    include_chassis: bool = True
    include_torso: bool = True
    include_head: bool = True
    include_hand: bool = True

    def __post_init__(self):
        if self.hand_types is None:
            default = (DexforceW1HandBrand.DH_PGC_GRIPPER_M
                       if self.arm_kind == DexforceW1ArmKind.INDUSTRIAL
                       else DexforceW1HandBrand.BRAINCO_HAND)
            self.hand_types = {
                DexforceW1ArmSide.LEFT: default,
                DexforceW1ArmSide.RIGHT: default,
            }

    @property
    def urdf_cfg(self) -> URDFCfg:
        return _build_urdf_cfg(
            self.arm_kind, self.hand_types, self.version,
            self.include_chassis, self.include_torso,
            self.include_head, self.include_hand)

    @property
    def control_parts(self) -> Dict[str, List[str]]:
        return _build_control_parts(
            self.arm_kind, self.include_torso,
            self.include_head, self.include_hand)

    @property
    def solver_cfg(self) -> Dict[str, SolverCfg]:
        return _build_solver_cfg(self.arm_kind, self.version)

    @property
    def drive_pros(self) -> JointDrivePropertiesCfg:
        return _build_drive_pros(self.arm_kind, self.include_hand)

    @property
    def attrs(self) -> RigidBodyAttributesCfg:
        return RigidBodyAttributesCfg(
            mass=1.0, static_friction=0.95, dynamic_friction=0.9,
            linear_damping=0.7, angular_damping=0.7,
            contact_offset=0.005, rest_offset=0.001,
            restitution=0.05, max_depenetration_velocity=10.0)

    min_position_iters: int = 32
    min_velocity_iters: int = 8

    def build_pk_serial_chain(self, device) -> Dict[str, "pk.SerialChain"]:
        ...

    def build_cfg(self, **overrides) -> RobotCfg:
        ...
```

Helper functions (`_build_urdf_cfg`, `_build_control_parts`, `_build_solver_cfg`, `_build_drive_pros`) are extracted from the current `_build_default_*` static methods and live alongside the def class (in the same module or a shared utils module within the robot package).

**Unchanged files**: `types.py` (enums) and `params.py` (kinematics data) remain as-is — they are pure data with no refactoring needed.

**Reduction**: ~390 LOC cfg.py -> ~80 LOC def.py + extracted helper functions. The `from_dict`, `to_dict`, `save_to_file`, and `_build_default_solver_cfg` methods are all handled by the generic `build_cfg` and base serialization.

### Robot Registry

A central registry allows robot lookup by name:

```python
# embodichain/lab/sim/robots/registry.py

_ROBOT_REGISTRY: Dict[str, Type[RobotDef]] = {}

def register_robot(name: str, def_cls: Type[RobotDef]):
    """Decorator to register a robot definition class."""
    _ROBOT_REGISTRY[name] = def_cls
    return def_cls

def get_robot_def(name: str, **variant_kwargs) -> RobotDef:
    """Look up a robot definition by name and instantiate with variant kwargs."""
    if name not in _ROBOT_REGISTRY:
        raise ValueError(
            f"Unknown robot: {name}. "
            f"Available: {list(_ROBOT_REGISTRY.keys())}")
    return _ROBOT_REGISTRY[name](**variant_kwargs)

def build_robot_cfg(name: str, **kwargs) -> RobotCfg:
    """Convenience: look up robot, build cfg, return RobotCfg for the spawner."""
    overrides = kwargs.pop("overrides", {})
    robot_def = get_robot_def(name, **kwargs)
    return robot_def.build_cfg(**overrides)
```

**Usage**:

```python
# Simple robot — no variants
cfg = build_robot_cfg("CobotMagic",
    overrides={"uid": "my_robot", "init_pos": [0, 0, 1.0]})
robot = sim.add_robot(cfg=cfg)

# Complex robot — with variants
cfg = build_robot_cfg("DexforceW1",
    arm_kind="anthropomorphic",
    overrides={"uid": "w1", "init_pos": [0, 0, 0.5]})
robot = sim.add_robot(cfg=cfg)

# Direct instantiation (still works)
cfg = CobotMagicDef().build_cfg(uid="my_robot", init_pos=[0, 0, 1.0])
```

### File Structure

```
embodichain/lab/sim/robots/
├── __init__.py              # Updated: import registry, all robot defs
├── protocol.py              # NEW: RobotDef protocol + generic build_cfg
├── registry.py              # NEW: register_robot decorator + lookup functions
├── cobotmagic.py            # REFACTORED: CobotMagicDef replaces CobotMagicCfg
├── dexforce_w1/
│   ├── __init__.py          # Updated: export DexforceW1Def
│   ├── types.py             # UNCHANGED
│   ├── params.py            # UNCHANGED
│   ├── utils.py             # REFACTORED: extract helpers, remove managers now in def
│   └── def.py               # NEW: DexforceW1Def class
```

### Backward Compatibility

- Old `CobotMagicCfg` and `DexforceW1Cfg` are kept as thin wrappers that delegate to the new `Def` classes. They will be deprecated after 1-2 release cycles.
- `SimulationManager.add_robot(cfg: RobotCfg)` is completely unchanged.
- Existing task environments importing `CobotMagicCfg` or `DexforceW1Cfg` continue to work without modification.
- `merge_robot_cfg` in `cfg_utils.py` is unchanged.

### Complexity Reduction Summary

| Component | Before | After |
|---|---|---|
| CobotMagic | ~200 LOC, custom from_dict + _build_default_cfgs | ~50 LOC flat declarations |
| DexforceW1 cfg.py | ~390 LOC, custom from_dict + 3 builders + serialization | ~80 LOC def + extracted helpers |
| Per-robot from_dict | Custom per robot | Generic build_cfg() from protocol |
| Per-robot to_dict/save_to_file | Custom per robot | Generic serialization from base |
| Robot discovery | Must know import path | Registry lookup by name |
