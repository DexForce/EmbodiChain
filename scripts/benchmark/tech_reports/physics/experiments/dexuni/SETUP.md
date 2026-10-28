> Publication in progress: data upload has not completed. Download remains disabled until the verified dataset commit is pinned.

# Environment and data setup

This suite depends on [DexSim adaptation PR #829](https://github.com/DexForce/EmbodiChain/pull/829), including commit `1eef39896e8bbcb1e092bfab553fbbb8812fc478`. The report-only PR targets main and should be reviewed separately; full simulation requires that adaptation.

EmbodiChain workers use the repository Python environment. Isaac Lab workers
use a separate Isaac Lab/Newton installation. Native DexSim adapters are
reference diagnostics. Install the supporting DexSim build and the solver
versions recorded with the chosen run; a released wheel without pneumatic
cavities cannot reproduce M05 or the full C01 scene.

`assets/physics/dexuni/` contains small authored configurations and resource
references. It does not contain measured arrays or videos. Raw data, generated
reports and media are published to `DexForceAI/embodichain_reports`; their
release index records exact dataset revisions, file sizes and SHA256 hashes.
Local working outputs belong in `outputs/benchmarks/tech_reports/dexuni/`.

## Current portability limits

These runners originated in a local benchmark campaign. Several advanced
task/geometry scripts retain workstation-specific interpreter, SDK source and
asset-cache paths. Review those constants before starting a new run; they are
not a portable one-command task launcher. S01/S02 pilot input and worker
environment overrides are documented in paths.py and the run_matrix scripts.
Frozen measured records retain their original paths and hashes as provenance.

## Restore published inputs, raw results and videos

Run from the repository root using an environment with the existing
`huggingface_hub` and `filelock` dependencies. No simulator imports or GPU are
required for download. No login is required for public reads.

```bash
# Show input bundles and total download size without network requests.
python -m scripts.benchmark.tech_reports.physics.experiments.dexuni.download_artifacts --list
# Restore one task's inputs; default group is inputs only.
python -m scripts.benchmark.tech_reports.physics.experiments.dexuni.download_artifacts --task C02
# Also restore original measurements/reports and recordings for that task.
python -m scripts.benchmark.tech_reports.physics.experiments.dexuni.download_artifacts --task C02 --kind results --kind videos
# Restore every retained task, including the top-level comparison tables.
python -m scripts.benchmark.tech_reports.physics.experiments.dexuni.download_artifacts --kind inputs --kind results --kind videos
# Use only previously downloaded files; fail if missing or corrupt.
python -m scripts.benchmark.tech_reports.physics.experiments.dexuni.download_artifacts --task C02 --offline
```

The source `assets/physics/dexuni/publication.json` pins the complete dataset
commit and file identities. Downloads use the official HF endpoint and standard
Hub cache layout at `~/.cache/embodichain/report-assets`, overridable with
`--cache-dir`. Archives are verified before extraction; online checksum failure
retries once, offline failure stops immediately. Existing differing local files
are never overwritten. Use `--output-root /path/to/empty/dexuni` to inspect a
release separately. Missing offline files require repeating the same command
without `--offline` while connected.

Restoration preserves paths relative to `outputs/benchmarks/tech_reports/dexuni/`.
Advanced workers already read frozen robot conversions under their task's
`inputs/` directory; these snapshots are included. General robot/scene assets
still use the existing EmbodiChain resolver (`get_data_path` and the registered
asset cache); the publication does not replace that catalog. Legacy pilot-only
adapters using `DEXUNI_INPUT_ROOT` remain a separate historical workflow.

Historical absolute paths and source hashes inside measured files remain
unchanged for audit. Downloads and hashing happen explicitly before simulation,
never in timed loops. Restoring evidence does not install the original platform
environments or automatically rewrite SDK/interpreter paths for a new run.

## Offline checks

```bash
.venv/bin/python -m pytest scripts/benchmark/tech_reports/tests tests/benchmark/physics/test_dexuni_layout.py -q
.venv/bin/python -m scripts.benchmark.tech_reports.reporting.dexuni.build_overview
```

The first command exercises protocol, timing, path and report behavior without
a new simulation. The second requires the retained result files in the local
output tree and rewrites the derived overview only. Tests do not establish
physical equivalence of different solver strategies.
