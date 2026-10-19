# Scene randomization demo

This example downloads the registered `TabletopFourBases` demo asset through
`get_data_path("TabletopFourBases/scene_export")`. The cached portable scene
contains a table
and four support groups: cutting board and fruit, saucer and cup, serving tray
and bottle/cup, and a small tray and two shapes. The default v4 policy uses the
shared CPU batch API for group XY and six selected members' XY/yaw. Existing
v1/v2/v3 policy examples remain available through `--config`.

Use an EmbodiChain environment with its matching DexSim runtime, NumPy, SciPy,
Trimesh, Shapely and PyYAML. Rendering additionally needs Pillow, a Vulkan-capable
GPU and `ffmpeg` with H.264 support; live browser playback uses Viser.

Run from the repository root, using a new output directory:

```bash
python -m scripts.tutorials.scene.grouped_scene_randomization \
  --output-root outputs/tabletop_four_bases_demo \
  --render --preview
```

The first run downloads `demo/tabletop_four_bases.zip` from the configured asset
host, with the official Hugging Face dataset as a fallback. Later runs reuse
the standard EmbodiChain cache. To download or list it separately:

```bash
embodichain data list --category demo
embodichain data download --name TabletopFourBases
```

Override `--source` with a local `scene_export` directory, its parent workspace,
or another registered asset reference. The example copies the selected export,
recomputes its baseline, generates 20 layouts with seed 42 and a 2 cm group
clearance, and reloads the saved candidates through the production loader.
Change `shared_scene_randomization.yaml` or pass `--config` for another policy.
The downloaded assets remain unchanged; caches, layouts and previews belong to
the separate output workspace.

The downloadable archive contains only:

```text
scene_export/
  scene_config.json
  scene_graph.json
  mesh_assets/<uid>/<uid>.glb
  articulated_assets/       # empty for this rigid-only scene
```

The configuration stores poses/scales and relative asset paths; the graph
stores the explicit support tree. The twelve GLBs include their materials and
textures. `scene.json` is optional and can be regenerated on import. Generation
logs, masks, intermediate models, `scene_usd`, existing randomization baselines,
geometry caches, generated layouts and preview files are not needed for this
demo asset. Resuming image-to-scene generation would require its separate
intermediate artifacts.

Each direct table child moves with all its descendants. The demo independently
checks actual mesh XY bounds, pairwise group AABB separation, and that every
unselected member retains its original parent-relative pose. `demo_verification.json`
records the measurements. Generation can return partial results when its attempt
budget runs out; the manifest and verification report retain that status.

Outputs under `scene_layouts/run_001/`:

- `scene_input.json`, `policy.json`, `prepared_geometry.json`, `layouts/*.json`,
  `run_result.json`: source-independent v4 run data. Legacy policies retain
  their original files, including `policy_resolved.yaml`.
- `demo_verification.json`: measured geometry and group-pose invariants.
- `preview/layout_*.png`: the original and every variant, using a fixed camera.
- `preview/contact_sheet.png`: original plus five representative variants.
- `preview/comparison.mp4` and `.gif`: original on the left, variants on the right.

`--render` loads assets once and uses native DexSim camera frames. Every layout
is shown for 0.5 seconds in the comparison; physics is never advanced. These
geometric checks and previews do not establish physical stability or task success.

`--preview` cycles the original and saved layouts at
`http://127.0.0.1:8080` (change with `--port`). Rotate and zoom in the browser;
Ctrl-C stops playback. To replay a saved run without generating again:

```bash
python -m embodichain.gen_sim.scene_engine.randomization.loading \
  --layout-dir outputs/tabletop_four_bases_demo/scene_layouts/run_001/layouts \
  --interval 1.5 --headless --viser --viser-host 127.0.0.1 --viser-port 8080
```

## CPU/CUDA batch generation

The legacy `grouped_randomization.yaml` policy retains the original CPU SLSQP
repair algorithm. For batch
rejection sampling, add `backend: batched`, `device: cpu` or `device: cuda:0`,
and `batch_size: 4096` to a version-1 input policy. Logical proposal counters
preserve the same seeded prefix when batch size changes. This distinct algorithm
records its own version/provenance. It rejects overlapping samples rather than
repairing them; dense scenes can exhaust the same attempt budget sooner. CUDA
must be available when explicitly selected.

Use the supplied 20-layout CUDA example and packed storage for a preview:

```bash
python -m scripts.tutorials.scene.grouped_scene_randomization \
  --source /path/to/tabletop_four_bases \
  --output-root outputs/tabletop_four_bases_cuda \
  --config scripts/tutorials/scene/grouped_randomization_cuda.yaml \
  --layout-format packed --render
```

For large batches, use the generation CLI with an appropriately increased
`candidate_count` and `max_attempts`; rendering is a separate operation:

```bash
embodichain scene-randomization \
  --output_root /path/to/workspace --config /path/to/batch_policy.yaml \
  --layout-format packed
```

Batch results retain compact numeric centers and poses. `result.candidates`
implements a read-only sequence; indexing or iterating constructs detached
`GeneratedScene` values. Saving works directly on the numeric pool. Loaded batch
variants are likewise lazy after all geometry, provenance and identity checks.
Original SLSQP results retain their tuple representation.

Packed runs use a version-2 manifest and one atomic `layouts/candidates.json`
container. The manifest is published after the complete container and source
revalidation. Existing replay accepts both storage formats and old version-1
policies/runs. Batch resolved policies use version 2 and include actual backend,
device and batch size. `--layout-format json` retains individual candidate files.
The saved version-2 `policy_resolved.yaml` can also be passed directly to
`--config` to reproduce the same logical proposals.

End-to-end timing, including generation, durable saving and validated reload:

```bash
python -m scripts.benchmark.scene.grouped_randomization \
  --workspace /path/to/workspace --sizes 20 500 5000
```

The benchmark compares SLSQP/JSON, CPU batches/packed and CUDA batches/packed.
Small cases repeat over three seeds; cases of 5,000 or more run once. GPU startup
is measured separately from steady-state throughput. These remain geometric
proposals; physical acceptance requires host validation.

## Intra-group XY and yaw

The v3 example selects `bottle_001`, `mug_001`, `apple_001`, `orange_001`,
`cube_001` and `cylinder_001`. The cup `mug_002` keeps its original pose relative
to the saucer. Generate and render 20 combined layouts:

```bash
python -m scripts.tutorials.scene.grouped_scene_randomization \
  --source /path/to/tabletop_four_bases \
  --output-root outputs/tabletop_four_bases_intra \
  --config scripts/tutorials/scene/grouped_randomization_intra.yaml \
  --layout-format packed --render --preview
```

Use `device: cpu` for CPU tensor batches. Set `randomize_groups: false` for
member-only augmentation. Omit `intra_group` (or leave its targets empty) to
retain group-only sampling and the original identities/storage versions. Both
scopes disabled is an error; members require `backend: batched`.

`intra_group.targets` maps explicit non-root UIDs to these settings:

| Setting | Default | Meaning |
| --- | --- | --- |
| `xy_offset_bounds_m` | `null` | Whole support region, or `[[min_x,min_y],[max_x,max_y]]` offsets from the original entity origin in support XY |
| `yaw_delta_range_deg` | `[-180,180]` | Continuous increment about the support normal; original tilt is retained |

`intra_group` also accepts `boundary_margin_m: 0.002`,
`sibling_margin_m: 0.002`, `support_gap_m: 0.001`, `grid_resolution_m: 0.001` and
`support_overrides: {}`. An override keyed by parent UID supplies a proper rigid
`parent_from_support` 4×4 matrix and a GeoJSON `polygon` (Polygon/MultiPolygon).
The polygon must be covered by coplanar upward mesh triangles. Detection uses
original child bottom geometry to choose a near-planar layer; ambiguous or
unsupported regions fail explicitly. Automatic layers must face within 18° of
parent-local +Z, with a 2 mm plane tolerance. The maximum positive height
deviation of the selected mesh layer is added to the support gap, so shallow
raised patches cannot consume that clearance. This conservative bound may leave
a small extra gap elsewhere on the layer. Use an evidence-checked override
for other frames. Rasters are limited to four million cells per support.

Whole cells must lie inside the inset support region, and every cell touched
by the projected body/subtree AABB is checked. This preserves holes and
separate islands. The conservative proxy includes scaled mesh extents, original
origin offsets and all descendants. Selected nodes are sampled in postorder;
fixed siblings remain obstacles, and unselected descendants follow their
parent. Updated group envelopes drive the outer layout. The v2 hierarchical sampler
partitions the remaining group-center area using already placed AABBs and
samples a free rectangle by area. Obstacle checks use fixed-size chunks to
avoid a temporary tensor cubic in the number of groups. Large scenes reduce
the actual batch size against a 256 MiB free-region scratch estimate; the
resolved policy records that effective batch size. An oversized single
proposal fails with a configuration error instead of attempting the allocation.
This budget does not cover all process RAM or GPU memory. It has no hidden placement retry loop.
One failed constraint
rejects the whole logical candidate without nested retries. Random streams are
addressed by seed, proposal ordinal, stage and UID. Small matrix products and
area-prefix scans use fixed arithmetic trees; batch size preserves the
same-backend logical prefix.

Support caches live in `scene_export/scene_randomization_geometry/<hash>.json`.
They are immutable and keyed by source bytes, extractor version and geometry
settings. The original `scene_randomization_base.yaml` is unchanged. V3 runs
save root XY translations, full selected parent-relative poses and cache
fingerprints. Both JSON and packed containers use the same validation and
CPU float64 reconstruction before any variant can be consumed. Saving checks
the exact cache fingerprint captured during generation both before writing and
before publishing the completion manifest. Concurrent cold preparations use
independent staging files. Replay never
resamples. Existing v1/v2 group runs and v3 runs with the original hierarchical
generator/support-cache version remain readable. New hierarchical runs record
`scene_engine.hierarchical_batch_rejection/v2` and `parent_support_grid/v2`;
changing the proposal algorithm is explicitly versioned.

Benchmark identical hierarchical policies on CPU/CUDA, including cold support
preparation, hot generation, fsynced save, validated reload and full streamed
object consumption:

```bash
python -m scripts.benchmark.scene.grouped_randomization \
  --workspace outputs/tabletop_four_bases_intra \
  --config scripts/tutorials/scene/grouped_randomization_intra.yaml \
  --sizes 20 500 5000
```

Cold preparation uses an isolated export copy so saved cache references remain
valid. Mesh loading, plane extraction and polygon rasterization occur only
during preparation. The benchmark consumes every resulting fragment but retains
only numeric results between stages. Reports include the actual configured
margin, generator version, numeric pipeline cost and full cost with consumption.
The leaderboard uses the full cost when it is measured. `consume_all` constructs generated fragments
and reloaded typed variants together to verify round-trip equality. Geometric acceptance does not certify
physical settling, arbitrary 6D pose changes, reparenting or task feasibility.

## Shared Sim API and task reset events

The source-independent API accepts prepared rigid geometry and an explicit
support tree. It does not require a Scene Engine workspace. Include the fixed
objects that participate in layout checks in the input tree.

```python
from embodichain.lab.sim.scene.expansion import (
    SceneRandomizationPolicyCfg, scene_layout_input_from_sim,
    prepare_scene_randomization, generate_scene_variants, apply_scene_variants,
    save_scene_variants, load_scene_variants,
)

source = scene_layout_input_from_sim(
    sim, support_uid="bench",
    parent_by_uid={"tray": "bench", "mug": "tray", "target": "bench"},
)
prepared = prepare_scene_randomization(
    source, SceneRandomizationPolicyCfg(group_targets=("tray",)),
)
result = generate_scene_variants(prepared, seed=42, candidate_count=20)
apply_scene_variants(
    sim, result.variants[:2], env_ids=[1, 3],
    movable_entity_ids=prepared.movable_uids,
)
run = save_scene_variants(result, "/tmp/shared-layouts", storage="packed")
loaded = load_scene_variants(run, scene_input=source)
```

Use `group_xy_offset_bounds_m={"tray": [[-0.04, -0.03], [0.04, 0.03]]}`
to restrict a selected group root to offsets from its authored pose in arena XY
coordinates. Equal lower and upper bounds fix an axis; omitted roots use the
full feasible support region. Member XY/yaw bounds use the parent support frame.

Geometry is extracted once; candidate sampling and selected-row application do
not load meshes or advance physics. Source and variant identities exclude arena
replication offsets. The application result lists the host caches requiring
refresh. Geometric acceptance does not establish robot reachability or task
success.

The main support uses a mesh-covered rectangular sampling region. Detection
retains tabletop openings and chooses a rectangle that excludes them; an
irregular support may therefore offer a smaller sampling area. Content-addressed
preparation caches from earlier extractors remain untouched and are not reused.

Handwritten expert and simulator RL environments configure the same service as
an ordinary reset event, without a Task Program or an additional environment
configuration switch:

```yaml
env:
  events:
    scene_layout:
      func: RandomizeSceneLayout
      mode: reset
      params:
        entity_cfg: {uid: bench}
        entity_cfgs:
          - {uid: tray}
          - {uid: mug}
          - {uid: target}
        parent_by_uid: {tray: bench, mug: tray, target: bench}
        source: {kind: online}
        policy:
          backend: batched
          device: cpu
          group_targets: [tray]
          intra_group:
            targets:
              mug:
                xy_offset_bounds_m: null
                yaw_delta_range_deg: [-180, 180]
        candidate_count: 64
        max_attempts: 10000
```

Place the event after other pose initialization and before state-registration
or observation events. Overlapping built-in pose/scale randomizers are rejected.
Ordinary selective reset retains cached geometry and unused candidates; an
explicit `reset(seed=...)`, including a repeated seed, rewinds the event stream.
To replay v4 layouts, use `source: {kind: saved, layout_dir: /absolute/run/path}`.
Replay cycles saved candidates and never resamples. Failed preparation or writes
block recording for the affected rows.

Reset info and episode metadata use provenance schema 2. Read one event through
`info["scene_randomization"][row]["events"]["scene_layout"]` for the event name
above. Each named event retains its own source, variant ID, seed and ordinal.
Disjoint layout events can coexist; resetting one clears only its selected rows
and provenance. A success in one event cannot clear another event's failure.

Scene Engine generation/edit APIs accept an optional `scene_randomization` job
or YAML policy. The combined CLI triggers it once after its final export:

```bash
embodichain scene-engine --image input.png --output_root /tmp/scene \
  --edit_prompt "move the mug" \
  --scene-randomization-config scripts/tutorials/scene/shared_scene_randomization.yaml
```

New pipeline jobs produce v4 packages. Existing `scene-randomization` policies
and v1/v2/v3 runs retain their original behavior. Convert a validated old run
without sampling or changing its IDs through
`embodichain.gen_sim.scene_engine.randomization.adapter.convert_layouts`.
Hierarchical v3 conversion retains the validated support certificate and the
height rules used by the old extractor, without detecting support again.

Measure the complete shared path, including four-environment native application:

```bash
python -m scripts.benchmark.scene.shared_layout \
  --workspace /absolute/scene-output --policy /absolute/v4-run/policy.json
```
