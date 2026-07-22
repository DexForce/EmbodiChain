# Plan: cuRobo planner - self-generated YAML only, drop robot_profiles, fewer spheres

## Goal
Make the cuRobo planner generate **all** of its YAML internally from the `Robot`
(URDF) and `RigidObject` meshes - no external `franka.yml` / `collision_franka_demo.yml`.
Remove the now-redundant `CuroboPlannerCfg.robot_profiles` (and the public
`CuroboRobotProfileCfg`); profiles are auto-derived from the robot's solver. Cut
the auto-generated robot collision-sphere count so planning is fast.

## End-state API
```python
CuroboPlannerCfg(
    robot_uid=ROBOT_UID,
    world=CuroboWorldCfg(rigid_objects=[demo_block]),  # or empty world if None
    auto_gen=CuroboAutoGenCfg(),                       # reduced spheres by default
    sim_base_to_curobo_base=None,                      # only override (auto-derives otherwise)
    warmup=..., max_attempts=..., use_cuda_graph=...,
)
```
- No `robot_profiles`, no public `CuroboRobotProfileCfg`, no `world_config_path`,
  no external `robot_config_path`.
- Control part resolved at plan time from `CuroboPlanOptions.control_part`,
  validated against `robot.control_parts`.
- Robot YAML + world YAML both auto-generated and cached.

## Files

### A. `embodichain/lab/sim/planners/curobo_planner.py` (the adapter)
1. **`CuroboRobotProfileCfg` -> private `_CuroboProfile`** (internal dataclass, not
   exported). Drop user-facing fields that are always auto-derived:
   `robot_config_path` (always auto-gen), `sim_to_curobo_joint_names` (identity),
   `active_joint_names`. Keep the internal fields the backend needs:
   `robot_config_path`, `sim_to_curobo_joint_names`, `tool_frame_name`,
   `tool_frame_to_tcp`, `base_link_name`, `sim_base_link_name`,
   `sim_base_to_curobo_base`. Remove `"CuroboRobotProfileCfg"` from `__all__`.
2. **`CuroboWorldCfg`**: remove `world_config_path` and its "both set" validation.
   Keep `rigid_objects`, `obstacle_representation`, `collision_cache`,
   `dynamic_obstacle_names`, `multi_env`, `_RigidObjectRefList.__post_init__`.
3. **`CuroboPlannerCfg`**: remove `robot_profiles`. Add
   `sim_base_to_curobo_base: list[list[float]] | None = None` (the one frame
   convention not auto-derivable from the solver). Keep `auto_gen`, `world`,
   planning params.
4. **`CuroboPlanner`**:
   - `_resolve_profile(options)` -> `_resolve_control_part(options) -> str`:
     validate `options.control_part` against `robot.control_parts` (not
     `cfg.robot_profiles`).
   - `_materialize_profile(profile, control_part)` ->
     `_materialize_profile(control_part) -> _CuroboProfile`: always auto-derive
     `tool_frame`/`base_link`/`sim_base_link`/`tool_frame_to_tcp` from
     `robot._solvers[control_part]`, identity `sim_to_curobo`, call
     `_auto_generate_robot_yaml`, and pull `sim_base_to_curobo_base` from
     `self.cfg.sim_base_to_curobo_base`.
   - `_get_backend(profile, control_part, batch_size)` ->
     `_get_backend(control_part, batch_size)`: calls `_materialize_profile`
     internally. World resolution: `world_config_path = self._auto_generate_world_yaml(world_cfg) if world_cfg.rigid_objects else None`.
   - `plan()`: `_get_backend(control_part, start.shape[0])`.
   - `__init__`: drop the `world_config_path`+`rigid_objects` mutual-exclusion
     check (only `rigid_objects` remains); keep `obstacle_representation` check.
   - Keep `_validate_profile_joint_names` (now validates the auto-derived identity
     mapping against the loaded planner joints), `_resolve_tool_frame`,
     `_validate_base_link_name`, `_to_curobo_joint_state`, `_map_curobo_to_sim`,
     `_tcp_to_tool_pose`, `_materialize_multi_env_scene_model` (still clones the
     auto-gen world path per row). Update `_CuroboBackend.profile` type to
     `_CuroboProfile`.
5. **`CuroboAutoGenCfg`**: reduce default sphere count. `sphere_density: 1.0` ->
   `0.1` (scales the per-link auto-estimate down ~10x, ~668 -> ~67 for Franka,
   preserving proportional allocation). Keep `fit_type="voxel"`, `num_spheres=None`.
   (Empirically confirm ~50-100 total + planning succeeds during implementation;
   adjust to 0.15 if coverage is too sparse.)

### B. `embodichain/lab/sim/planners/curobo_yaml.py`
- Docstrings: drop `franka.yml` / `CuroboRobotProfileCfg.robot_config_path` /
  `world_config_path` references; reframe as "the cuRobo planner auto-generates
  this from the Robot/RigidObject". No logic change.

### C. `examples/sim/planners/curobo_planner.py`
- `CuroboPlannerCfg(robot_uid=..., world=CuroboWorldCfg(rigid_objects=[demo_block]), warmup=..., max_attempts=..., use_cuda_graph=...)` - drop `robot_profiles=`. Drop the
  `CuroboRobotProfileCfg` import.

### D. Remove `embodichain/data/assets/curobo/collision_franka_demo.yml`
No longer used (no external world YAML). Delete the file.

### E. Tests
- **`tests/sim/planners/test_curobo_planner.py`** (dependency-free):
  - Remove explicit-profile / joint-mapping-validation tests
    (`test_curobo_robot_profile_cfg_requires_joint_map`, the
    `_default_profile()`/`franka.yml`-based validation tests). Keep pure-helper
    tests (`_reorder_by_names`, `_matrix_to_position_quaternion`,
    `_validate_dynamic_obstacles`) and config-default/export/lazy-import tests
    (adapted to removed fields).
  - Backend tests: `_FakeCuroboBindings(full_joint_names=["sim_a","sim_b"])`
    (identity); `_FakeRobot` gets `_solvers={"arm": SimpleNamespace(end_link_name="tool", root_link_name="sim_base", urdf_path="/fake.urdf", tcp_xpos=None)}` + `cfg=SimpleNamespace(fpath="/fake.urdf")` + `joint_names`; mock
    `generate_curobo_robot_yaml` (monkeypatch) to return a fake path. `_make_planner`
    drops `profiles=`, adds `auto_gen=CuroboAutoGenCfg(cache_dir=tmp_path)`. Call
    sites `planner._get_backend(profile,"arm",n)` -> `planner._get_backend("arm",n)`;
    drop `planner.cfg.robot_profiles["arm"]`.
  - `test_multi_env_materializes_one_scene_mapping_per_batch_row`: replace
    `world_config_path=str(scene_path)` with `rigid_objects=[_FakeRigidObject(...)]`
    (reuse the fake from `test_curobo_world_yaml.py`) so the auto-gen world YAML is
    cloned per row.
- **`tests/sim/planners/test_curobo_integration.py`** + **`tests/sim/atomic_actions/test_curobo_motion_source_e2e.py`**:
  remove `_franka_profile` / `_demo_world_path` / `robot_profiles=`; use
  `CuroboPlannerCfg(robot_uid=..., world=CuroboWorldCfg(rigid_objects=[block]))`.
  Expect slower first run (robot YAML gen) but fast planning (reduced spheres).
- **`tests/sim/planners/test_curobo_world_yaml.py`**: drop the
  `collision_franka_demo.yml` comment and the `cfg.world_config_path is None`
  assertion (field removed); keep the conversion/round-trip tests.

### F. Docs - `docs/source/overview/sim/planners/curobo_planner.md`
- Rewrite the profile section: profiles are auto-derived from the robot's URDF +
  solver (no `robot_config_path`/`sim_to_curobo_joint_names`/`franka.yml`).
- Rewrite the world section: world is auto-generated from `RigidObject`s (no
  `world_config_path`); mention `obstacle_representation`.
- **Keep every string asserted by `tests/docs/test_curobo_planner_docs.py`**:
  `CuroboPlannerCfg`, `planner_type="curobo"`, `cuRobo V2`, `attached-object`,
  `tool_frame_to_tcp`, `sim_base_to_curobo_base`, `multi_env=True`, `lock_joints`,
  `--record-save-path`, `examples/sim/planners/curobo_planner.py`. (Note:
  `tool_frame_to_tcp` and `sim_base_to_curobo_base` stay as concepts - the former
  is auto-derived from the solver's TCP, the latter is the one optional override.)

## Backward compatibility
This is an intentional API break: external-YAML escape hatches and `robot_profiles`
are removed. The example and all tests are updated to the auto-gen-only path. The
auto-gen + caching already existed; this just makes it the only path.

## Verification
1. `conda activate embodichain && black` on touched files.
2. `python -m py_compile` + import `embodichain.lab.sim.planners`.
3. `pytest tests/sim/planners/test_curobo_planner.py tests/sim/planners/test_curobo_world_yaml.py tests/docs/test_curobo_planner_docs.py -m "not slow"` (all green, no CUDA).
4. `pytest tests/sim/planners/test_curobo_integration.py tests/sim/atomic_actions/test_curobo_motion_source_e2e.py` (CUDA) - plans succeed; confirm the cached robot YAML now has ~50-100 spheres (not 668) and planning completes in seconds.
5. `python examples/sim/planners/curobo_planner.py --headless --disable-record` - completes end-to-end (auto-gen robot + world, plan, replay, TCP error) in well under a minute.
6. Sphere-count check: `python -c "import yaml; d=yaml.safe_load(open(<cached robot yaml>)); print(sum(len(v) for v in d['robot_cfg']['kinematics']['collision_spheres'].values()))"` ~50-100.

## Out of scope
- Changing the cuRobo collision backend or trajectory optimization.
- Per-link sphere tuning (the density multiplier applies uniformly).
- Keeping `collision_franka_demo.yml` for reference (deleted - no external YAML).
