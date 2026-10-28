# Environment and data setup

This suite depends on [DexSim adaptation PR #829](https://github.com/DexForce/EmbodiChain/pull/829), including commit `1eef39896e8bbcb1e092bfab553fbbb8812fc478`. The report-only PR targets main and should be reviewed separately; full simulation requires that adaptation.

EmbodiChain workers use the repository Python environment. Isaac Lab workers
use a separate Isaac Lab/Newton installation. Native DexSim adapters are
reference diagnostics. Install the supporting DexSim build and the solver
versions recorded with the chosen run; a released wheel without pneumatic
cavities cannot reproduce M05 or the full C01 scene.

`assets/physics/dexuni/` contains small authored configurations and resource
references. It does not contain measured arrays or videos. Runtime inputs and readable
report evidence are published to `DexForceAI/embodichain_reports`; their
release index records exact dataset revisions, file sizes and SHA256 hashes.
Local working outputs belong in `outputs/benchmarks/tech_reports/dexuni/`.

## Current portability limits

These runners originated in a local benchmark campaign. Several advanced
task/geometry scripts retain workstation-specific interpreter, SDK source and
asset-cache paths. Review those constants before starting a new run; they are
not a portable one-command task launcher. S01/S02 pilot input and worker
environment overrides are documented in paths.py and the run_matrix scripts.
Frozen measured records retain their original paths and hashes as provenance.

## Restore runtime inputs

Run from the repository root with the existing `huggingface_hub` and `filelock`
dependencies. Downloads use the public official endpoint without login or GPU.
The release has 12 task folders (S01-C/T share one), including frozen meshes,
textures, robot conversions, trajectories and required reference metadata.
Videos and complete result archives are intentionally excluded. Small original
JSON/CSV already published on Hugging Face remain report evidence.

```bash
python -m scripts.benchmark.tech_reports.physics.experiments.dexuni.download_artifacts --list
# One task, including its runtime support files.
python -m scripts.benchmark.tech_reports.physics.experiments.dexuni.download_artifacts --task C02
# All input assets.
python -m scripts.benchmark.tech_reports.physics.experiments.dexuni.download_artifacts
# Reuse verified cached files without network access.
python -m scripts.benchmark.tech_reports.physics.experiments.dexuni.download_artifacts --task C02 --offline
```

`assets/physics/dexuni/publication.json` pins the dataset commit, sizes and SHA256.
The standard Hub cache is `~/.cache/embodichain/report-assets` (`--cache-dir` to
change it). Files restore into `outputs/benchmarks/tech_reports/dexuni/`, or an
empty `--output-root` for inspection. Existing differing files are never
replaced. Offline missing/corrupt files fail; repeat online to fetch them.

Runtime support ZIPs contain only required native manifests, initial states,
reference controller modules and integrity metadata. Some preserve historical
`preview/` or `analysis/` paths because those files are consumed as inputs by
the current runners. They contain no recordings. `runtime_checks.json` verifies
the required reference bytes for workers that previously required a video hash;
recordings are no longer a prerequisite for simulation. Frozen input/config
bytes and historical provenance remain unchanged.

S01/S02 new campaigns can consume the restored snapshots with
`run_s01.py --input-root outputs/benchmarks/tech_reports/dexuni/s01_fixed_five/inputs`
and the analogous `run_s02.py` path. These options launch new campaigns; they
are not part of download validation. Workers also support `--inputs` or
`DEXUNI_INPUT_ROOT` as documented by their existing CLI. S03 uses
`--inputs outputs/benchmarks/tech_reports/dexuni/s03_cloth_corner_lift/inputs/s03`.
Other current EmbodiChain/Isaac Lab workers read their task's restored inputs
by default. Converted W1 USD and frozen Franka URDF paths are resolved relative
to those restored inputs rather than their recorded workstation paths.

General catalog assets retain `get_data_path` and the registered asset cache.
Asset restoration does not install the simulator environments or rewrite
remaining SDK/interpreter paths; see portability limits above. Native geometry
regeneration is a separate diagnostic workflow and can still need its original
source checkout. No downloads or hashing occur in timed simulation loops.

## Offline checks

```bash
.venv/bin/python -m pytest scripts/benchmark/tech_reports/tests tests/benchmark/physics/test_dexuni_layout.py -q
.venv/bin/python -m scripts.benchmark.tech_reports.reporting.dexuni.build_overview
```

The first command exercises protocol, timing, path and report behavior without
a new simulation. The second requires the retained result files in the local
output tree and rewrites the derived overview only. Tests do not establish
physical equivalence of different solver strategies.
