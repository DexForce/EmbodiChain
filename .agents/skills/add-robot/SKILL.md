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
  `drive_pros` / `attrs`.
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

| Layout | When | Files |
|--------|------|-------|
| Single-file | Variant-less robot | `my_robot.py` |
| Package | Robot with variants (versions / arm kinds / hand brands) | `types.py`, `cfg.py`, optional `params.py`/`utils.py`, `__init__.py` |

## The Contract (read first)

A cfg's `_build_defaults` must populate:

- `uid` (str)
- `urdf_cfg` (URDFCfg) or `fpath`
- `control_parts` (Dict[str, List[str]]; joint names support regex)
- `solver_cfg` (Dict[str, SolverCfg]; keys match `control_parts`)
- `drive_pros` (JointDrivePropertiesCfg)
- `attrs` (RigidBodyAttributesCfg)

`build_pk_serial_chain` must read from `_pk_urdf_path` (a property for
constant-path robots, a method for variant-dependent paths). The PK chain's DOF
must match the matching `control_parts` entry (the test stub asserts this).

## Steps

1. **Pick a layout** using the table above. Single-file for variant-less robots;
   package for robots with variants.
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
       self.drive_pros = JointDrivePropertiesCfg(stiffness={"JOINT[1-6]": 1e4})
   ```

   Variant-aware template (reads version / arm_kind):

   ```python
   def _build_defaults(self, init_dict=None):
       init_dict = init_dict or {}
       self.version = MyRobotVersion(init_dict.get("version", "v1"))
       self.arm_kind = MyRobotArmKind(init_dict.get("arm_kind", "default"))
       ...  # then urdf_cfg / control_parts / solver_cfg / drive_pros / attrs
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
6. **Add `__all__` and register** in `embodichain/lab/sim/robots/__init__.py`:

   ```python
   from .my_robot import MyRobotCfg
   __all__ = ["MyRobotCfg"]
   ```

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
| `drive_pros` | JointDrivePropertiesCfg | Joint stiffness, damping, force |
| `attrs` | RigidBodyAttributesCfg | Rigid-body physics attributes |
| variant fields | enum / str / bool | Optional subclass fields |
| `_pk_urdf_path` | property or method → str | URDF for the FK/IK serial chain |

**File locations:**

- Config: `embodichain/lab/sim/robots/<name>.py` or `embodichain/lab/sim/robots/<name>/`
- Registry: `embodichain/lab/sim/robots/__init__.py`
- Docs: `docs/source/resources/robot/<name>.md`
- Tests: `tests/sim/objects/test_robot_cfg.py`
- Base class: `embodichain/lab/sim/cfg.py` (`RobotCfg`)
- Guide: `docs/source/guides/add_robot.rst` · Tutorial: `docs/source/tutorial/add_robot.rst`
