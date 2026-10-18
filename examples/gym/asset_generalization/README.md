# Asset-generalized Task Program batches

This example creates three annotated cube packages and one support platform,
selects cubes from a directory and the platform from an explicit list, then
materializes standalone UR5 Pick/Place deployments. No online generation service
is used. Package geometry is authored in meters and local Z-up coordinates.

Run from the repository root with the SDK and official task/robot extensions
installed, using a new output directory:

```bash
PYTHONPATH="$PWD" python examples/gym/asset_generalization/run.py \
  --output /tmp/asset-batches --batches 2 --seed 7
```

Preparation runs the production configuration loader and records candidate
rejections in `batches/preparation.report.json`. Each batch has `task.yaml`,
its bound program/integration, local assets, and `assets.manifest.json`. Selected
assets can repeat across seeds: a fixed seed guarantees reproducibility for the
same candidate snapshot, not uniqueness across batches.

Opt into native simulation and acceptance-gated recording:

```bash
PYTHONPATH="$PWD" python examples/gym/asset_generalization/run.py \
  --output /tmp/asset-batches-executed --batches 2 --episodes 2 --execute
```

Native execution needs the normal DexSim, robot assets and planning dependencies.
The initial validator measures support by the floor. The Task Program checks lift
and uprightness; the independent final validator measures the cube footprint and
bottom against the platform top. These checks are deliberately specific to this
example, not a universal task-success oracle. Unreachable grasps, planning errors
and physical failures may reject candidates; preparation alone does not establish
physical success.

The example explicitly sets `AssetGeneralizationCfg.record_trajectory=True`;
the host saves only accepted trajectories through the ordinary Gym reset boundary.
Every collection invocation has a fresh batch-local `recordings/run_*` directory,
so restarting recorder counters cannot overwrite another batch's data. `report.json`
distinguishes measured acceptance, a returned commit/discard reset, and successful
recorder finalization; it does not claim a persistence receipt for every episode.
No recording is enabled merely by semantic matching.

For real assets, replace the generated catalogs with your own prepared packages
following `examples/data/asset_catalog/README.md`. The integration's
`asset_requirements` owns task roles and relations. Providers supply candidates;
matching remains authoritative. External providers implement the same two-stage
protocol and emit verifiable prepared-package inventories. PR6 online generation
is deferred.

Current materialization supports upright rigid meshes with a shared visual and
collision model. Semantic support/container placement requires both objects to
have role declarations and an explicit relation. Fixed grasps, absolute placement
poses, initial local-pose overrides, non-unit scale and geometry-calibrated
expansion profiles fail explicitly. Independent objective definitions remain
unchanged; author objectives appropriate to the permitted variation.

Local custom robot resources must keep referenced mesh files within the selected
model directory; sibling/package-URI layouts outside that boundary are rejected
rather than emitted as incomplete portable artifacts. Packaged robot IDs use the
SDK asset resolver as usual.
