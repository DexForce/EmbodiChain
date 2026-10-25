---
name: add-robot
description: Use when adding a new robot to EmbodiChain — scaffolds a RobotCfg subclass (single-file or package layout) with the _build_defaults hook, build_pk_serial_chain, registration, docs page, and test stub.
---

# Add Robot

## When to Use

- Adding a new robot to EmbodiChain.
- Adding a variant to an existing robot (a new version / arm kind / hand brand).
- Scaffolding a `RobotCfg` subclass.

## The RobotCfg Protocol

Every robot config subclasses `RobotCfg` and overrides two hooks:

- `_build_defaults(self, init_dict=None)` — read variant fields from `init_dict`,
  set them on `self`, then populate `urdf_cfg` / `control_parts` / `solver_cfg` /
  `joint_drive_props` / `attrs`.
- `build_pk_serial_chain(self, device=...)` — return `{control_part: pk.SerialChain}`,
  reading the PK URDF from a single `_pk_urdf_path` source.

`from_dict` is a 3-line template (do not reimplement):

```python
cfg = cls()
cfg._build_defaults(init_dict)
return merge_robot_cfg(cfg, init_dict)
```

`to_dict` / `to_string` / `save_to_file` are inherited from `RobotCfg` and round-trip.

## Two Layouts

Concrete configs live in the `extensions/embodichain_robots/` import package, bundled
with the core distribution/version. Generic robot runtime, configuration,
solvers, and composition stay in `embodichain/lab/sim/`.

| Layout | When | Files |
|--------|------|-------|
| Single-file | Model using a vendor-shared archive | `<vendor>/<model>.py` |
| Package | Model-owned assets or versions/calibrations | `<vendor>/<model>/cfg.py`, `assets.py`, optional specifications, lazy `__init__.py` |

## The Contract (read first)

A cfg's `_build_defaults` must populate:

- `uid` (str)
- `urdf_cfg` (URDFCfg) or `fpath`
- `control_parts` (Dict[str, List[str]]; joint names support regex)
- `solver_cfg` (Dict[str, SolverCfg]; keys match `control_parts`)
- `joint_drive_props` (JointDrivePropertiesCfg)
- `attrs` (RigidBodyPhysicsCfg)

Use `arm` / `eef` for default single-arm control groups and `left_arm`,
`right_arm`, `left_eef`, `right_eef` for dual-arm defaults. Add an EEF group
only when the robot has controllable end-effector joints. These names are a
convention: user-defined names remain valid, without automatic aliases or a
reserved-name validator. Keep asset component, link and joint identifiers intact.

`build_pk_serial_chain` must read from `_pk_urdf_path` (a property for
constant-path robots, a method for variant-dependent paths). The PK chain's DOF
must match the matching `control_parts` entry (the test stub asserts this).

## Steps

1. **Pick a vendor/model owner** and a layout from the table above. Keep vendor
   and model initializers lazy. Asset-only imports must not load configurations.
2. **Create the cfg file(s).** Subclass `RobotCfg`. Declare variant fields (enums)
   if using the package layout.
3. **Implement `_build_defaults(self, init_dict=None)`.** Set variant fields from
   `init_dict`, then populate the Contract fields. Single-file template:

   ```python
   def _build_defaults(self, init_dict=None):
       init_dict = init_dict or {}
       self.uid = "MyRobot"
       self.urdf_cfg = URDFCfg(components=[...])
       self.control_parts = {"arm": ["JOINT[1-6]"]}
       self.solver_cfg = {"arm": OPWSolverCfg(end_link_name="link6", root_link_name="base_link")}
       self.joint_drive_props = JointDrivePropertiesCfg(stiffness={"JOINT[1-6]": 1e4})
   ```

   Variant-aware template (reads version / arm_kind):

   ```python
   def _build_defaults(self, init_dict=None):
       init_dict = init_dict or {}
       self.version = MyRobotVersion(init_dict.get("version", "v1"))
       self.arm_kind = MyRobotArmKind(init_dict.get("arm_kind", "default"))
       ...  # then urdf_cfg / control_parts / solver_cfg / joint_drive_props / attrs
   ```

4. **Implement `build_pk_serial_chain`** reading from `_pk_urdf_path`:

   ```python
   @property
   def _pk_urdf_path(self) -> str:
       return get_data_path("MyRobot/arm.urdf")

   def build_pk_serial_chain(self, device=torch.device("cpu"), **kwargs):
       chain = create_pk_serial_chain(
           urdf_path=self._pk_urdf_path, device=device,
           end_link_name="link6", root_link_name="base_link",
       )
       return {"arm": chain}
   ```

5. **Keep `from_dict` as the 3-line template** — do not reimplement it.
6. **Add `__all__` and register** the class in
   `extensions/embodichain_robots/_catalog.py`, then add its lazy export to
   `extensions/embodichain_robots/__init__.py` (`_EXPORTS` and `__all__`):

   ```python
   # Catalog entry; importing the catalog must not import the robot module.
   "MyRobot": {"class_path": "embodichain_robots.my_vendor.my_robot:MyRobotCfg"}
   ```

   The main wheel already advertises this catalog through `embodichain.robots`.
   Gym, workspace CLI, and dual-arm composition use its shared resolver.
   Variant defaults belong in the catalog. Concrete robots use only the
   canonical package; no core compatibility module or CLI class map is needed.
   Supply optional control metadata when known: `dof` counts unique independent
   default control joints, excluding mimic followers; `control_parts` records
   raw per-part counts. `mimic_joints` maps followers to leaders, and
   `control_part_mimics` records follower membership by group. Overlapping groups
   must not inflate DOF. Declare `supports_dual_arm` only for a compatible default
   single-arm URDF/control-part/solver contract. Families may declare `variants`
   with control profiles and a `default_variant`; flat preset IDs use `variant_of`.
   Add explicit `vendor`, `morphology` and `arm_count` metadata when known;
   unknown fields stay `None`. Variant classification inherits parent values
   unless explicitly overridden. Keep the catalog free of runtime imports.
   Put model-owned asset classes in `assets.py` (shared archives at vendor level)
   and register their unchanged asset IDs, class paths and categories in
   `_asset_catalog.py`. The `embodichain.assets` entry point connects them to
   the core resolver and data CLI; listing must not construct datasets.

7. **Add documentation:** create `docs/source/resources/robot/<name>.md` and add
   it to `docs/source/resources/robot/index.rst`.
8. **Add a test stub** with a `__main__` smoke test + the DOF drift guard. Use
   `/add-test` for full test scaffolding; the guard snippet is:

   ```python
   chains = cfg.build_pk_serial_chain()
   for part, chain in chains.items():
       assert len(chain.get_joint_parameter_names()) == len(cfg.control_parts[part])
   ```

9. **Verify:** `preview-asset` CLI + `RobotCfg.from_dict(cfg.to_dict())` round-trip.

## Reusable embodiment handoff

`RobotCfg` owns simulation construction, control parts, solvers, and physical
properties, plus optional mounted camera configurations in `sensors`.
Use robot-local parent link names and stable sensor UIDs. Standalone simulation
creates these cameras only with `enable_sensor=True`; Gym overrides that
switch with its environment acquisition setting. An explicit sensor list
replaces defaults and `sensors: []` clears them. Keep independent cameras,
task declarations and Task Program skill profiles outside the robot class.

If the user also wants the robot selectable from task deployments, invoke
`$add-embodiment-component` after the robot config passes its own tests. That
component selects the robot, declares additional independent sensors, and maps stable
logical resources/endpoints/capabilities/commands into `skill_profile`.
Do not copy those semantic declarations into the robot class.

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| `all` instead of `__all__` | Use `__all__` — lowercase `all` breaks `import *`. |
| `solver_cfg` set twice | Set it once in `_build_defaults` only. |
| PK URDF drifts from sim URDF | Route PK through `_pk_urdf_path`; keep the DOF guard. |
| Reimplementing `from_dict` | Keep the 3-line template; put logic in `_build_defaults`. |
| `root_link_name` as a tuple | It must be a `str`. |
| Calling a nonexistent `validate` | Don't call methods that don't exist. |

## Quick Reference

| Parameter | Type | Description |
|-----------|------|-------------|
| `uid` | str | Unique robot identifier |
| `urdf_cfg` | URDFCfg | URDF file and components |
| `control_parts` | Dict[str, List[str]] | Joint groups for control |
| `solver_cfg` | Dict[str, SolverCfg] | IK solver configurations |
| `joint_drive_props` | JointDrivePropertiesCfg | Joint drive, limits, friction, and armature |
| `attrs` | RigidBodyPhysicsCfg | Rigid-body physics attributes |
| variant fields | enum / str / bool | Optional subclass fields |
| `_pk_urdf_path` | property or method → str | URDF for the FK/IK serial chain |

**File locations:**

- Config: `extensions/embodichain_robots/<vendor>/<model>.py` or `<vendor>/<model>/`
- Catalog / lazy exports: `extensions/embodichain_robots/_catalog.py`, `extensions/embodichain_robots/__init__.py`
- Shared resolver: `embodichain/robot_registry.py`
- Docs: `docs/source/resources/robot/<name>.md`
- Tests: `tests/sim/objects/test_robot_cfg.py`
- Base class: `embodichain/lab/sim/cfg/robot.py` (`RobotCfg`)
- Guide: `docs/source/guides/add_robot.rst` · Tutorial: `docs/source/tutorial/add_robot.rst`
