# Plan: Build the cuRobo collision world from `RigidObject`s

## Goal
Let callers pass live `RigidObject` instances (e.g. `demo_block`) straight into
`CuroboWorldCfg` and have the planner generate a cuRobo V2 scene YAML from their
meshes — instead of relying on a hand-authored `collision_franka_demo.yml`.
Mirror the existing `robot_config_path=None` → auto-generate-robot-YAML idiom.

Decisions (confirmed with user):
- **Integration**: add `CuroboWorldCfg.rigid_objects`; the planner lazily
  generates + caches the world YAML on first plan. A public
  `generate_curobo_world_yaml()` is also exposed for explicit/path-based use.
- **Default representation**: `cuboid` (local-frame AABB → OBB via the object's
  pose). `mesh` and `sphere` (cuRobo `fit_spheres_to_mesh`) are supported via the
  `obstacle_representation` field.

## cuRobo schema (verified in installed curobo V2)
`SceneCfg.create()` (`curobo/_src/geom/types.py`) parses a YAML mapping:
```yaml
cuboid:
  <name>: {dims: [x,y,z], pose: [x,y,z,qw,qx,qy,qz]}
mesh:
  <name>: {vertices: [[x,y,z],...], faces: [i0,i1,i2,...], pose: [x,y,z,qw,qx,qy,qz]}  # faces = FLAT int list, 3 per triangle (types.py:867)
sphere:
  <name>: {position: [x,y,z], radius: float}
```
Pose convention `[x,y,z,qw,qx,qy,qz]` — identical to `RigidObject.get_local_pose()`.
Static YAML is authored in the cuRobo base/world frame (same convention as the
existing `collision_franka_demo.yml`); the demo robot base sits at the world
origin, so the object's world pose is correct unchanged.

## Files

### A. `embodichain/lab/sim/planners/curobo_yaml.py` — add world-YAML generator
Add `generate_curobo_world_yaml` and a pure tensor-level helper (testable w/o
CUDA/dexsim/curobo). Keep `generate_curobo_robot_yaml` as-is.

```python
def _mesh_to_obstacle_entry(name, vertices, faces, pose, *, representation="cuboid",
                             fit_type="voxel", num_spheres=None, sphere_density=1.0,
                             surface_radius=0.005, iterations=200,
                             collision_sphere_buffer=0.0, device="cuda:0"):
    # vertices: (V,3) tensor, faces: (F,3) int tensor, pose: (7,) xyz,qw,qx,qy,qz
    # returns list[(top_key, name, fields_dict)]  (1 for cuboid/mesh, N for sphere)
    # cuboid: local AABB -> OBB.
    #   vmin=verts.min(0); vmax=verts.max(0); dims=vmax-vmin; c_local=(vmin+vmax)/2
    #   R=matrix_from_quat(pose[3:7]); center_world = R@c_local + pose[:3]
    #   fields = {"dims": dims.tolist(), "pose": [*center_world, *pose[3:7]]}
    # mesh: fields = {"vertices": verts.tolist(), "faces": faces.flatten().tolist(),
    #                 "pose": pose.tolist()}
    # sphere: build trimesh, fit_spheres_to_mesh (lazy import, CUDA),
    #         emit {"position": center, "radius": r+buffer} per sphere, names f"{name}_{i}"
```
- `matrix_from_quat` already imported in `rigid_object.py` from `embodichain.utils.math`.
- Public wrapper:
```python
def generate_curobo_world_yaml(rigid_objects, output_path, *, representation="cuboid",
                               env_id=0, fit_type="voxel", num_spheres=None,
                               sphere_density=1.0, surface_radius=0.005, iterations=200,
                               collision_sphere_buffer=0.0, device="cuda:0") -> str:
    # for each RigidObject: uid (fallback f"obstacle_{i}"),
    #   verts = obj.get_vertices(env_ids=[env_id], scale=True)[0]
    #   tris  = obj.get_triangles(env_ids=[env_id])[0]
    #   pose  = obj.get_local_pose(to_matrix=False)[env_id]   # (7,) wxyz
    # group entries by top_key into one dict; yaml.dump(..., sort_keys=False)
    # makedirs(dirname); return output_path
```
- Add `"generate_curobo_world_yaml"` to `__all__`.
- Validate `representation` ∈ {"cuboid","mesh","sphere"}; error otherwise.
- Skip objects whose mesh is empty (warn), matching `generate_curobo_robot_yaml`.

### B. `embodichain/lab/sim/planners/curobo_planner.py` — auto-gen wiring
1. `TYPE_CHECKING` import `RigidObject` (avoid a circular import at module load).
2. `CuroboWorldCfg` — add fields:
   - `rigid_objects: list[RigidObject] | None = None` — live obstacles to bake
     into a generated world YAML when `world_config_path` is `None`.
   - `obstacle_representation: str = "cuboid"` — `"cuboid"` | `"mesh"` | `"sphere"`.
   - Docstrings; note `collision_cache` already covers `cuboid`/`mesh`.
3. `CuroboPlanner`:
   - In `__init__` (or `_get_backend`): error if **both** `world_config_path` and
     `rigid_objects` are set (ambiguous). Treat empty `rigid_objects` as `None`.
   - `_get_backend`: resolve the world path before the existing scene_model block:
     ```python
     world_cfg = self.cfg.world
     world_config_path = world_cfg.world_config_path
     if world_config_path is None and world_cfg.rigid_objects:
         world_config_path = self._auto_generate_world_yaml(world_cfg)
     scene_model = world_config_path
     if multi_env:
         scene_model = self._materialize_multi_env_scene_model(world_config_path, int(batch_size))
     ```
     (Existing multi-env cloning logic is unchanged — it already handles a path.)
   - `_auto_generate_world_yaml(world_cfg) -> str`: resolve cache dir from
     `self.cfg.auto_gen.cache_dir` (else `XDG_CACHE_HOME`/`~/.cache/embodichain_curobo`),
     compute content-hashed cache key, return cached path or call
     `generate_curobo_world_yaml(rigid_objects, path, representation=…,
     fit_type=auto_gen.fit_type, num_spheres=…, sphere_density=…,
     surface_radius=…, iterations=…, collision_sphere_buffer=…)`.
   - `_world_yaml_cache_key(world_cfg) -> str`: md5 over per-object
     `(uid, vertices.tobytes(), faces.tobytes(), pose.tolist(), representation,
     fit params)`. Include `auto_gen.force` bypass.

### C. `examples/sim/planners/curobo_planner.py` — use the new API
- Replace
  `world=CuroboWorldCfg(world_config_path=_demo_world_path())`
  with
  `world=CuroboWorldCfg(rigid_objects=[demo_block])`  (default `cuboid` repr).
- Remove the now-unused `_demo_world_path()` helper.
- Update the `demo_block` comment: no longer "keep synchronized with
  collision_franka_demo.yml" — the YAML is generated from the object.
- Keep `collision_franka_demo.yml` on disk (tests still reference it).

### D. Tests — `tests/sim/planners/test_curobo_world_yaml.py` (new, mostly CUDA-free)
- A tiny `_FakeRigidObject` with `uid`, `get_vertices`, `get_triangles`,
  `get_local_pose` returning synthetic tensors (a unit box).
- `test_cuboid_entry_matches_aabb`: `_mesh_to_obstacle_entry` for a centered unit
  box + identity pose → `dims=[1,1,1]`, pose at the box center. With a translated
  pose, center_world is the pose translation (centered mesh).
- `test_mesh_entry_flat_faces`: faces serialized flat; vertex/face counts correct.
- `test_generate_curobo_world_yaml_assembles_yaml`: write to tmp, reload with
  `yaml.safe_load`, assert `cuboid.<name>` / `mesh.<name>` schema.
- `test_curobo_scene_cfg_loads_generated_yaml`: `pytest.importorskip("curobo")` →
  `SceneCfg.create(data)` succeeds and lists the obstacle name(s).
- `test_sphere_mode_requires_cuda`: `pytest.importorskip("curobo")` +
  `skipif(not cuda)` → smoke-generate a sphere YAML from a small mesh.
- (Optional) extend `tests/sim/planners/test_curobo_integration.py` with a
  CUDA test that plans with `CuroboWorldCfg(rigid_objects=[block])` and asserts
  the path avoids the block — mirrors the existing static-YAML test.

### E. Docs — `docs/source/overview/sim/planners/curobo_planner.md`
- Replace the "Geometry is not extracted automatically from DexSim." sentence
  with a short note that `rigid_objects` auto-generates a cached world YAML
  (`cuboid` default; `mesh`/`sphere` available via `obstacle_representation`).
- Add a one-line example `CuroboWorldCfg(rigid_objects=[block])`.
- **Keep every string asserted by `tests/docs/test_curobo_planner_docs.py`**
  (`CuroboPlannerCfg`, `planner_type="curobo"`, `cuRobo V2`, `attached-object`,
  `tool_frame_to_tcp`, `sim_base_to_curobo_base`, `multi_env=True`, `lock_joints`,
  `--record-save-path`, `examples/sim/planners/curobo_planner.py`).

## Backward compatibility
- `world_config_path` still works exactly as before; `rigid_objects` only kicks in
  when `world_config_path is None`.
- Existing tests that pass `world_config_path=_demo_world_path()` are untouched.
- `collision_franka_demo.yml` stays on disk.

## Verification
1. `conda activate embodichain && black` on all touched files.
2. `python -m py_compile` + import `embodichain.lab.sim.planners`.
3. `pytest tests/sim/planners/test_curobo_world_yaml.py` (no CUDA for the core cases).
4. `pytest tests/docs/test_curobo_planner_docs.py` (docs strings intact).
5. `pytest tests/sim/planners/test_curobo_planner.py` (existing config tests).
6. Run the example headless: `python examples/sim/planners/curobo_planner.py
   --headless --disable-record` → first run generates a cached world YAML, plans
   around the block, replays, and reports a small TCP error; second run hits cache.

## Out of scope
- Auto-rebasing object poses into a non-origin robot base frame (static YAML stays
  authored in the cuRobo world/base frame; use `dynamic_obstacle_names` +
  `update_dynamic_obstacles` for moving/offset obstacles — composition already
  works since the generated YAML carries the obstacle name + initial pose).
- Removing `collision_franka_demo.yml` (still used by tests).
