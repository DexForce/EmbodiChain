# Primitive baseline and PourWater asset acceptance

These two examples exercise different preparation paths. The cube/platform
baseline uses native `Cube` primitives. The PourWater case replaces both bottle
and cup through the asset Provider, matching and materialization workflow.
Neither example calls an online generation service.

Run from the repository root with the SDK and bundled task/robot extensions
installed. Every invocation requires a new output directory.

## Primitive cube/platform baseline

```bash
PYTHONPATH="$PWD" python examples/gym/asset_generalization/run.py \
  --output /tmp/primitive-baseline --batches 2 --seed 7
```

The default batches use 4 cm and 5 cm cubes and a primitive support platform.
Shapes are declared directly in physical configuration; no OBJ export, asset
catalog, Provider or fabricated asset manifest is involved. Initial bottom height
and the semantic platform placement frame follow the cube dimensions.

Add `--execute` to build native UR5 environments, execute Pick/Place and save only
accepted trajectories. The example uses the canonical scene execution gate and
independent measurements of initial support, final footprint and contact height.
An executed run exits nonzero if any episode fails acceptance.
It is a primitive parameter-variation baseline, not a file-asset retrieval test.

## Replace bottle and cup in PourWater

```bash
PYTHONPATH="$PWD" python examples/gym/asset_generalization/pour_water.py \
  --output /tmp/pour-assets --seed 1 --batches 2

PYTHONPATH="$PWD" python examples/gym/asset_generalization/pour_water.py \
  --output /tmp/pour-assets-executed --seed 1 --batches 2 --execute
```

The example reuses the official CobotMagic four-segment task: pick the bottle,
move it above the cup, tilt and return upright, then place it back on the table.
The table is a 0.7 m square primitive, keeping its collision footprint clear of
the two arm mounts. Bottle and cup remain mesh assets with their configured
convex-decomposition collision policies.

Default catalogs contain the existing open-neck PourWaterAssets bottle/cup plus
explicitly labelled dimensional derivatives: an 8% taller bottle and a 10% wider,
5% shorter cup. The example converts Y-up GLB geometry into normalized Z-up OBJ
geometry once, retaining source attribution. These are size derivatives of two
source models, not four independently retrieved models. The default seed selects
original bottle/original cup and tall bottle/wide cup; arbitrary seeds need not
provide unique combinations.

Use authored alternatives without editing the program:

```bash
PYTHONPATH="$PWD" python examples/gym/asset_generalization/pour_water.py \
  --output /tmp/custom-pour --bottle-catalog /absolute/bottles \
  --cup-catalog /absolute/cups --execute
```

Each package follows `examples/data/asset_catalog/README.md`. The integration
requires category `bottle` or `cup`, `open_top: true`, bottle capabilities
`antipodal_grasp` and `pour_source`, and cup capability `container`. The
`pours_into` relation checks an evidenced mouth radius plus task clearance
against the selected cup aperture before constructing a scene.

Structured `descriptor.provenance.pouring` annotations are part of the hashed
asset content:

- Bottle: `mouth_center_m`, `mouth_radius_m`, object-local `tilt_axis`, flattened
  SE(3) `object_to_eef`, exact embodiment `profile_id`, and nonempty `evidence`.
- Cup: `mouth_center_m`, `opening_radius_m`, and nonempty `evidence`.

Coordinates are meters in each normalized asset frame. Bottle grasp calibration
must match the selected robot profile/TCP; category matching cannot prove grasp
compatibility. The default derivatives explicitly transform their inherited
calibration and must still pass physical acceptance. External models need their
own annotations; the example does not infer missing calibration from a filename.

`AssetAdaptationCfg(pour=AssetPourAdaptationCfg(...))` explicitly enables this
four-segment adaptation. It replaces selected grasp/axis annotations, recomputes
the cup-relative staging pose from the tilted bottle mouth, and converts the
return to a semantic table support target. It updates position validators using
conservative tolerances. Other unsupported fixed calibrations remain rejected. The example requests
0.15 m nominal lower-rim clearance, preserving the original task's reachable
pouring height; actual pose evidence is checked independently.

## Measured acceptance and recording

A step observer supplies live bottle/cup poses to `PouringAcceptanceMonitor`.
Initial checks require supported upright objects and low arm speeds before
execution. During the pour segment, at least two samples must simultaneously have:

- at least 50 degrees of bottle tilt;
- the entire conservative bottle-mouth projection inside the cup aperture;
- positive bottle-mouth height within the configured band above the cup.

The monitor also rejects cup translation/tilt excursions and requires the bottle
to finish upright near its initial position. These measurements supplement
canonical program completion, phase validators and settling checks. Metadata
explicitly records `fluid_transfer_measured: false`: this scene has no fluid
simulation and does not certify water volume transferred.

Preparation writes batch configurations and immutable asset manifests. Execution
writes `report.json`, per-episode `pour-diagnostics-*.json`, and accepted `.pt`
trajectories beneath separate
`recordings/run_*` directories. Trajectories retain asset-set provenance and pour
measurement summaries. Rejected or unverified episodes are discarded. Acceptance,
returned reset requests and recorder finalization are separate reported facts;
they are not interchangeable with per-episode persistence receipts.

Online generation (planned PR6), articulated replacement and heterogeneous
parallel scenes remain outside these acceptance cases.
