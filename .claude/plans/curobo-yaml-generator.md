# Plan: Auto-generate cuRobo robot YAML from the robot's URDF (cached)

## Goal
1. Promote `generate_curobo_robot_yaml` from the example into the package as a reusable tool.
2. On every cuRobo planner init, auto-generate the robot YAML from the robot's URDF
   (reading TCP/end-link/root-link from `robot._solvers[control_part]`), with a content-hashed
   disk cache so repeated inits skip regeneration.
3. Remove the hardcoded `_franka_profile` from the example (TCP no longer hardcoded).
4. Keep full backward compatibility: explicit `robot_config_path` + `sim_to_curobo_joint_names`
   profiles (and the existing tests) still work unchanged.

## Files

### A. NEW `embodichain/lab/sim/planners/curobo_yaml.py`
Move `generate_curobo_robot_yaml` here (out of the example), adapted:
- Apache header, `from __future__ import annotations`, `__all__ = ["generate_curobo_robot_yaml"]`.
- Inline the `get_robot_urdf_link_vert_face` logic (fetch `robot.cfg.fpath` + per-link
  `robot.get_link_vert_face`); do not depend on the example helper.
- Switch `print("[WARN]...")` -> `logger.log_warning(...)` (package convention).
- Same signature as today: `(robot, control_part, output_path, *, tool_frame=None,
  fit_type="morphit", num_spheres=None, sphere_density=1.0, surface_radius=0.005,
  iterations=200, collision_sphere_buffer=0.0, max_acceleration=15.0, max_jerk=500.0,
  device="cuda:0") -> str`.
- Keep all the verified logic: UrdfRobotParser(load_meshes=False) for base_link + parent map
  (self_collision_ignore) + mimic joints; MORPHIT/VOXEL/SURFACE sphere fitting; mimic-excluded
  cspace + lock_joints.

### B. `embodichain/lab/sim/planners/curobo_planner.py` (the adapter)
1. **`CuroboRobotProfileCfg`**: make `robot_config_path: str | None = None` and
   `sim_to_curobo_joint_names: dict[str, str] | None = None` (both were `MISSING`). When `None`,
   the adapter auto-derives them. Update docstrings. All other fields unchanged.
2. **New `CuroboAutoGenCfg`** configclass:
   `cache_dir: str | None = None` (default `~/.cache/embodichain_curobo` / `XDG_CACHE_HOME`),
   `fit_type: str = "voxel"`, `num_spheres: int | None = None`, `sphere_density: float = 1.0`,
   `surface_radius: float = 0.005`, `iterations: int = 200`, `collision_sphere_buffer: float = 0.0`,
   `force: bool = False` (bypass cache).
3. **`CuroboPlannerCfg.auto_gen: CuroboAutoGenCfg = CuroboAutoGenCfg()`**.
4. **New helpers on `CuroboPlanner`**:
   - `_materialize_profile(profile, control_part) -> CuroboRobotProfileCfg`:
     returns a `dataclasses.replace` copy with auto-derived fields filled from the solver:
     - `solver = self.robot._solvers.get(control_part)` (guard: `_solvers` may be empty).
     - `tool_frame_name` <- `solver.end_link_name` (fallback: control part's last link).
     - `tool_frame_to_tcp` <- `solver.tcp_xpos.tolist()` (fallback None). For Franka this is
       identity -> the adapter's `_convert_tcp_to_tool_frame` treats it as a no-op.
     - `base_link_name` <- `solver.root_link_name` (fallback `robot.cfg.base_link_name`).
     - `sim_base_link_name` <- `solver.root_link_name`.
     - `sim_to_curobo_joint_names` <- identity `{j: j for j in control-part joints}` (the
       generated YAML uses the same URDF joint names, so the mapping is identity).
     - `robot_config_path` <- if already set, keep; else `_auto_generate_robot_yaml(...)`.
   - `_auto_generate_robot_yaml(control_part, tool_frame) -> str`:
     resolves `urdf_path` (`solver.urdf_path` or `robot.cfg.fpath`), computes cache key, returns
     cached path or calls `generate_curobo_robot_yaml` to create it.
   - `_robot_yaml_cache_key(urdf_path, control_part, tool_frame, auto_gen) -> str`:
     `md5(urdf_path + urdf_file_bytes + control_part + tool_frame + fit params)`. Includes the
     path (so moving the URDF regenerates, since the YAML embeds the path) and content (so editing
     regenerates).
5. **`_get_backend`**: first line becomes `profile = self._materialize_profile(profile,
   control_part)`. The rest (MotionPlannerCfg.create, validations, backend) is unchanged and now
   sees a fully concrete profile. Add `import dataclasses` / `import hashlib` / `import os` as
   needed (local or top-level).

### C. `embodichain/lab/sim/planners/__init__.py`
Add `from .curobo_yaml import *` so `generate_curobo_robot_yaml` is exported.

### D. `examples/sim/planners/curobo_planner.py` (the example)
- Remove `def _franka_profile` (line 176) and its call site in `main()` (line ~592).
- Remove the now-unused `def get_robot_urdf_link_vert_face` and the explicit
  `generate_curobo_robot_yaml(...)` call in `main()` (its logic moved to the package; the adapter
  auto-generates internally).
- Remove the local `def generate_curobo_robot_yaml` definition (moved to package).
- Update the demo's `CuroboPlannerCfg` to use `robot_profiles={CONTROL_PART: CuroboRobotProfileCfg()}`
  (all defaults -> auto-derive + auto-generate). Keep `world=CuroboWorldCfg(...)` etc.
- Net: the example no longer hardcodes TCP or `franka.yml`; it relies on auto-derivation from the
  robot + solver.

### E. Docs
- `docs/source/overview/sim/planners/curobo_planner.md`: add a short "Auto-generated robot YAML"
  section describing the auto-gen + cache behavior and that `robot_config_path`/TCP are now
  optional. **Keep all strings asserted by `tests/docs/test_curobo_planner_docs.py`**:
  `CuroboPlannerCfg`, `planner_type="curobo"`, `cuRobo V2`, `attached-object`, `tool_frame_to_tcp`,
  `sim_base_to_curobo_base`, `multi_env=True`, `lock_joints`, `--record-save-path`,
  `examples/sim/planners/curobo_planner.py`.

## Backward compatibility
- Explicit profiles (`robot_config_path="franka.yml"` + `sim_to_curobo_joint_names={...}`) still
  work: `_materialize_profile` only fills `None` fields, so explicit values pass through.
- `tests/sim/planners/test_curobo_planner.py` constructs profiles with explicit fields -> still
  pass (optional defaults don't affect explicit construction).
- `tests/sim/planners/test_curobo_integration.py` and
  `tests/sim/atomic_actions/test_curobo_motion_source_e2e.py` have their own `_franka_profile`
  using `franka.yml` -> untouched, still valid (explicit path).

## Verification
1. `black --target-version py310` on all touched files.
2. `python -m py_compile` + import check.
3. Run existing unit tests: `pytest tests/sim/planners/test_curobo_planner.py` (no CUDA needed).
4. Run the example headless: `python examples/sim/planners/curobo_planner.py --headless
   --disable-record` -> confirm it auto-generates a cached YAML on first run, hits cache on second
   run, and plans successfully.
5. Re-validate the generated cached YAML loads via `curobo.motion_planner.MotionPlanner` (as
   before): `planner.joint_names` == the 7 arm joints, `base_link == "base"`,
   `tool_frames == ["fr3_hand_tcp"]`.

## Decisions / defaults
- `auto_gen.fit_type` defaults to `"voxel"` (fast first-gen at runtime; cached after). Users wanting
  best quality set `auto_gen.fit_type="morphit"`. (The standalone `generate_curobo_robot_yaml`
  still defaults to `"morphit"`.)
- Cache location respects `XDG_CACHE_HOME`, else `~/.cache/embodichain_curobo/`.
- `solver.tcp_xpos.tolist()` is set as `tool_frame_to_tcp` even when identity (harmless no-op in
  the adapter); avoids fragile float-equality checks.

## Out of scope
- Touching the test files' own `_franka_profile` copies (they exercise the explicit path).
- Changing the world/scene YAML handling.
- attached-object support (still unsupported).
