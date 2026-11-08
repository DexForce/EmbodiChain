# Using Scene Services

These workflows start from explicit asset configurations or a prepared
`SimulationManager`. Use the normal EmbodiChain installation with its matching
DexSim runtime. CPU scene strategies use NumPy, SciPy, Trimesh, and Shapely;
using `sim.scene` does not require GenSim model services. Rendering and explicitly
selected CUDA paths have their own runtime requirements.

## Construct a scene from a recipe

The existing `scripts/tutorials/scene/tabletop_recipe.py` example builds a
workbench, two five-panel bins, six dynamic parts, and a virtual keep-out region.
It runs without Gym, robots, or model calls.

From the repository root:

```bash
python scripts/tutorials/scene/tabletop_recipe.py \
  --seed 7 --output outputs/scene_recipe/report.json
```

The example disables browser visualization and uses the `no-render` renderer.
It still prepares and steps the native physics runtime.

### Describe physical inputs and layout proxies

`SceneFragment` copies resolved `RigidObjectCfg` and `ArticulationCfg` values.
`SceneRecipe` adds a versioned identity and a `LayoutProblem`:

```{literalinclude} ../../../../../scripts/tutorials/scene/tabletop_recipe.py
:language: python
:pyobject: make_recipe
```

Each layout UID has a geometry-center seed and XY half-extents. Bounds contain
the complete AABB, rather than constraining only its center. Fixed entries can
represent existing bodies or virtual keep-out regions without physical configs.
The support UID names a fragment entity, and every movable UID needs both a
rigid-object configuration and finite sampling bounds.

`generate(recipe, seed=7, ordinal=0)` samples and solves without creating a
simulation. It changes XY translations while retaining authored heights and
orientations. Check `generated.layout.status`, `generated.fragment`, and
`generated.variant` before declaration: an unsuccessful attempt has no usable
fragment or variant. Seed and logical ordinal address the proposal independently
of physical environment slots.

For reusable assemblies, call `compose` before constructing the recipe:

```python
import numpy as np
from embodichain.lab.sim.scene import compose

# fragment already contains validated object configs and a named fixture anchor.
arena_from_fixture = np.eye(4)
arena_from_fixture[:3, 3] = (0.5, 0.0, 0.75)
assembly = compose(
    [fragment],
    namespace="station_a",
    anchor="fixture",
    arena_from_fragment=arena_from_fixture,
)
```

The anchor is aligned with the supplied arena transform; entity and anchor
names receive the `station_a__` prefix. If the assembly is used in a layout,
construct its proxies and UID mappings in this resulting frame and namespace.
Declare robots, sensors, and other resources through their normal manager or
embodiment configuration paths.

### Declare, settle, measure, and restore

The same script's `run` function connects proposal and physical validation:

```{literalinclude} ../../../../../scripts/tutorials/scene/tabletop_recipe.py
:language: python
:pyobject: run
```

The order matters:

1. Declare the generated fragment through the existing manager and call
   `sim.prepare()` after completing the initial scene.
2. Build a `SceneView` over the prepared bindings. Use a full-world selection
   for `settle_scene`, whose callback calls `sim.update(step=steps)`.
3. Request convergence mode when measured stability is required. Fixed-step
   mode only establishes that the requested steps ran.
4. Read actual poses and recompute part AABBs after settling. Even small rotations
   can invalidate authored fixed-orientation proxies.
5. Run `check_layout`, capture the measured physical state, and verify a
   restoration with `compare_states`.

On success the JSON report contains `layout_status="solved"`, `stable=true`,
`measured_xy_layout_valid=true`, and `state_restored=true`, together with the
individual checks. A failed required check raises an error. These measurements
qualify this scene preparation; robot reachability and task acceptance need
their own evidence.

## Change an existing scene

For a prepared simulation with a rigid object named `part_0`, capture a baseline
and apply an explicitly permitted pose change:

```python
from embodichain.lab.sim.scene import (
    SceneChangeSet,
    SceneView,
    apply_changes,
    capture_state,
    compare_states,
    restore_state,
)

view = SceneView(sim, env_ids=(0,))
baseline = capture_state(view)
pose = view.pose("part_0")[0].detach().cpu().numpy()
pose[0, 3] += 0.05
change = SceneChangeSet(poses={"part_0": pose}, movable_uids=("part_0",))
result = apply_changes(view, change)
# The host handles result.refresh_required before planning or observing again.

restore_state(view, baseline)
restored = compare_states(baseline, capture_state(view))
if not restored.passed:
    raise RuntimeError(restored.checks)
```

Neither pose application nor restoration advances physics. `apply_changes`
applies the same declared poses to every selected row. For distinct poses, use
`apply_changes_batch(view, changes_by_env_id)` with exactly one change set for
each selected environment ID. Unselected rows retain their state.

Restore into the original owner with the same ordered selection and unchanged
bindings. By default, saved joint positions and targets must satisfy current
joint limits. An explicit `qpos_tolerance` permits bounded clipping of measured
positions and requires caller-owned round-trip validation; targets remain strict.

Use `scene.physics.capture_rigid_physics`, `apply_rigid_physics`, and
`restore_rigid_physics` for mass/inertia and friction baselines. `SceneState`
does not capture those parameters. `restore_render_state` is a separate
physics-disabled publication API for measured frames; physical resets use
`restore_state`.

## Generate layouts for several environments

For an existing rigid-only scene, the `scene.expansion` API separates geometry
preparation from repeated proposal generation. This example assumes at least
four environments and the UIDs `bench`, `tray`, `mug`, and `target`. The explicit
support tree places the mug on the tray and both the tray and target on the bench.

```python
from embodichain.lab.sim.scene.expansion import (
    SceneRandomizationPolicyCfg,
    apply_scene_variants,
    generate_scene_variants,
    load_scene_variants,
    prepare_scene_randomization,
    save_scene_variants,
    scene_layout_input_from_sim,
)

source = scene_layout_input_from_sim(
    sim,
    support_uid="bench",
    parent_by_uid={"tray": "bench", "mug": "tray", "target": "bench"},
    scene_id="bench-layout/v1",
)
prepared = prepare_scene_randomization(
    source,
    SceneRandomizationPolicyCfg(group_targets=("tray",), device="cpu"),
)
batch = generate_scene_variants(
    prepared, seed=42, candidate_count=2, max_attempts=1000,
)
if len(batch.variants) < 2:
    raise RuntimeError(f"Insufficient layouts: {batch.status}")
applied = apply_scene_variants(
    sim,
    batch.variants[:2],
    env_ids=(1, 3),
    movable_entity_ids=prepared.movable_uids,
)
# The host refreshes applied.refresh_required and owns subsequent physics steps.
run_dir = save_scene_variants(batch, "outputs/shared-layouts", storage="packed")
loaded = load_scene_variants(run_dir, scene_input=source)
```

Preparation extracts geometry from a prototype row and validates shared asset
scales. Include every fixed obstacle that participates in the layout checks in
the support tree. Preparation retains source/policy identity and reusable support
geometry. Reprepare after asset, scale, or source changes.

The tray and its descendants move together; member XY/yaw randomization requires
an explicit `intra_group` policy. Generation can exhaust its attempt budget with
fewer candidates than requested. Inspect `status` and candidate count; use
`next_ordinal` as `start_ordinal` to continue the addressed stream.

Selected-row application preflights the complete request and publishes poses
without stepping. It does not run settling, refresh host caches, or certify task
success. If settling is needed afterwards, supply a view of the entire world,
including rows outside this application selection.

For a complete asset-backed example, run:

```bash
python -m scripts.tutorials.scene.grouped_scene_randomization \
  --output-root outputs/tabletop_four_bases_demo
```

The example downloads the registered `TabletopFourBases` asset, generates and
reloads layouts, and checks group geometry. Use a fresh output directory.
Rendering and browser playback are optional `--render` and `--preview` modes;
their requirements, policies, and outputs are described in the
{download}`scene example README <../../../../../scripts/tutorials/scene/README.md>`.

## Add robot and task checks

Use `RobotSceneWorkspace` when object placement depends on an explicit robot
control part and object-local grasp. Sampling produces candidate object poses
and joint seeds; `check_object_pose` independently checks actual pose IK, joint
limits, and full-pose FK residuals. Recheck a pose after settling or modification.

The [robot workspace tutorial](../../../tutorial/robot_workspace.rst) uses the
same scene in standalone and Task Program modes, showing rejected red starts
and accepted green starts. For complete task-conditioned layout and robot-base
search, see the [feasibility reference](../../../api_reference/embodichain/embodichain.lab.sim.scene.feasibility.rst).
The [integration map](integrations.md) identifies the hosts that own reset order,
execution, observations, acceptance, and recording.
