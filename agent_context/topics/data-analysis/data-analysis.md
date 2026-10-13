# Data diversity analysis

`embodichain/data_analysis/` owns offline metadata, distribution queries and portable
replay. `embodichain analyze-data` dispatches lazily through `cli.py`.

- `schema.py`: version 1 episode records, measurement source/unit/frame/scope,
  explicit unknowns, identity and declared artifact checks.
- `catalog.py`: SQLite WAL current records, lifecycle event ledger and immutable
  snapshots. Equal upserts are no-ops; changed terminal records conflict.
- `statistics.py`: Polars categorical/joint counts and explicit-grid coverage.
  No target grid means no coverage percentage.
- `importers.py`: read-only LeRobot metadata adaptation. Missing facts remain
  unknown; imported metadata does not imply portable replay is available.
- `recording.py`: strict numeric NPZ loading without pickle and atomic directory
  publication; measured lift and final placement checks.
- `collection.py`: bounded subprocess attempts for the existing Franka repeated
  pick/place deployment. Uses the production configuration loader/executor;
  records real SceneExporter states and RGB. Failures stay in the catalog.
- `replay.py`: offline ViserBackend worker; scene, camera and comparison overlays
  share recorded time. One thread owns Viser mutations. Loopback by default.
- `ui.py`: optional Gradio workbench. Applied table/plot populations are frozen in
  session state for consistent slice export. Each session owns its viewer and
  its English/Chinese UI locale; English is the default. Locale changes update
  labels and plots without changing stable filter values or exported records.

`embodichain.lab` uses lazy module exports so offline replay does not import DexSim.
Install the `analysis` extra for Gradio and Matplotlib; the package reuses the
project's existing Viser, NumPy and Polars dependencies.

Portable arrays use `timestamps(T)`, `qpos(T,D)`, world-frame positions `(T,3)`,
scene positions `(T,N,3)`, scalar-first quaternions `(T,N,4)` and visibility `(T,N)`.
`scene.json` stores the existing visualization manifest. `camera.npz` stores
uint8 RGB frames and their timestamps. Artifact paths may be absolute or resolved
relative to the catalog; the UI resolves them before replay.

The preview is task-specific. Affordance is annotated; materials/light are
configured; initial position is measured. Program-reported semantic success and
geometric physical success are separate. Geometry clustering, augmentation-recipe execution and model attribution are not implemented.

`dimensions.py` owns versioned six-family feature definitions, completeness/source
summaries and typed multiselect/range/availability filters. Numeric quantities
with incompatible units are excluded from metric plots. Joint cells preserve
exact episode IDs; the UI exports the applied population, bin edges and drilldown
history. Availability distinguishes explicit unknown, missing and incompatible.

`trajectory.py` aligns recorded task segments by name plus occurrence (inclusive
endpoint states). Only curve display uses normalized time; speed/duration use
original timestamps. Arc-length path distance separates geometry from retiming.
This is descriptive comparison, not clustering. Joint plots use raw recorded
column indices; unequal robot IDs or joint dimensions are rejected. Viser retains
its original recording timeline; phase alignment applies to the analysis curves.

Validation: `tests/data_analysis/`, existing visualization protocol/backend tests,
and `analyze-data collect-preview --output <directory> --count 12`. Real-task
qualification is separate from CPU-only fixture tests. See the implementation
and acceptance records under `docs/superpowers/` for this preview's evidence.
