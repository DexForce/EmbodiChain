# EmbodiChain technical report benchmarks

This directory groups technical-report assets, comparison experiment scaffolds,
and shared execution/reporting designs. For existing runnable benchmarks and
CLI usage, start with the [upstream benchmark guide](https://github.com/DexForce/EmbodiChain/blob/f179b5558fa17df06f172f0b1e4e4f81df63be74/scripts/benchmark/README.md).

## Technical report comparison layout

The L/R/P/G/D families follow
[issue #679](https://github.com/DexForce/EmbodiChain/issues/679). This change
creates their directory scaffold and ownership notes. The shared framework remains planned. The local DexUni physics suite adds
experiment orchestration, platform adapters and offline reporting within these owners.

```text
scripts/benchmark/tech_reports/
  assets/
    README.md                 # asset ownership and publication guide
    DOWNLOAD_DESIGN.md         # proposed input download protocol
    index.json                # published input-bundle index snapshot
    shared/                   # small shared inputs and asset references
    loading/                  # L-series input definitions
    rendering/                # R-series input definitions
    physics/                  # P-series input definitions
    expert_generation/        # G-series input definitions
    demonstrations/           # D-series input/evidence references
  core/                       # planned shared execution and measurement
  reporting/                  # planned offline analysis and report building
  loading/                    # L: loading, rebuild, reset comparisons
    experiments/
    platforms/
  rendering/                  # R: rendering and observation comparisons
    experiments/
    platforms/
  physics/                    # P: physical quality and qualified performance
    experiments/
    platforms/
  expert_generation/          # G: expert trajectory/data-production comparisons
    experiments/
    platforms/
  demonstrations/             # D: end-to-end evidence and result assembly
    experiments/
    platforms/
```

| Series | Comparison script owner | Input owner | Initial work |
| --- | --- | --- | --- |
| L | [loading](loading/README.md) | `assets/loading/` | One representative loading experiment. |
| R | [rendering](rendering/README.md) | `assets/rendering/` | R-04 batched camera rendering, then R-03/R-05/R-06/R-09. |
| P | [physics](physics/README.md) | `assets/physics/` | One representative physics experiment. |
| G | [expert_generation](expert_generation/README.md) | `assets/expert_generation/` | G-03 expert trajectory generation, then G-04. |
| D | [demonstrations](demonstrations/README.md) | `assets/demonstrations/` | D-03 assembly from existing results and evidence. |

Each family's `experiments/` holds experiment definitions, parameter matrices,
and comparison orchestration. `platforms/` holds platform-specific execution
adapters. Preserve the issue's experiment IDs in configurations and records;
use Python-compatible names for future script modules. Each platform runs in
its own process and Python environment with frozen versions, matched inputs,
and explicit comparison conditions.

The planned [core](core/README.md) owns shared run/measurement contracts; the
planned [reporting](reporting/README.md) owns simulator-free result rebuilding.
R-04 and G-03 remain planned end-to-end slices. Other experiment folders are
reserved for incremental implementation. Empty directories are tracked with
`.gitkeep`; they contain no runnable placeholder scripts or invented results.

Existing `../motion_generation/`, `../atomic_action/`, `../rl/`,
`../task_program/`, [offscreen rendering](../rendering/README.md), and other
benchmarks keep their entry points and outputs. Reuse their domain
execution and adapt legacy results as the shared framework is implemented.
Future report templates and benchmark-skill updates belong to that work;
the existing three-table report contracts remain unchanged here.

Keep small authored input descriptions and asset references under `assets/`.
Large binary inputs live in Hugging Face or the existing asset repositories;
downloads use an external cache. Run records, images, episodes, and reports go
under `outputs/benchmarks/`, not into the source asset directories.

## Technical report assets

The [report asset guide](assets/README.md) describes the public
[DexForceAI/embodichain_reports dataset](https://huggingface.co/datasets/DexForceAI/embodichain_reports),
its directory conventions, publishing process, and published input-bundle index.
The [download design](assets/DOWNLOAD_DESIGN.md) defines future shared
benchmark input preparation, revision pinning, cache behavior, and provenance.

These conventions and the proposed download helper are owned by the benchmark
scripts. The general helper and unified asset CLI remain design only. DexUni provides
an explicit [published-artifact download command](physics/experiments/dexuni/SETUP.md)
using the same official endpoint, pinned revisions and external Hub cache.

## Local DexUni results and entry points

Layout follows [PR #832](https://github.com/DexForce/EmbodiChain/pull/832).
The 13 existing task items, including unsupported cases and
C02 preliminary measurements, are indexed in the
[task comparison](reporting/dexuni/RESULTS.md).

| Content | Owner |
| --- | --- |
| Small authored inputs and references | `assets/physics/dexuni/` |
| Experiment definitions and measurement orchestration | `physics/experiments/dexuni/` |
| Platform workers and recording | `physics/platforms/dexuni/` |
| Offline reporting and video assembly | `reporting/dexuni/` |
| Raw measurements, videos, reports and frozen execution inputs/source | `outputs/benchmarks/tech_reports/dexuni/` (repository relative) |
| Cleanup and layout-migration evidence | `outputs/benchmarks/tech_reports/maintenance/` |

See the [experiment plan](physics/experiments/dexuni/PLAN.md),
[reporting commands](reporting/README.md#dexuni-maintained-tools), and
[migration record](physics/experiments/dexuni/MIGRATION.md).
The local ignored `runs` and `maintenance` symlinks only preserve historical
absolute paths in frozen records. They are not source/resource publication
directories; all maintained output defaults use `outputs/benchmarks/`.
Frozen source copies in results are execution evidence, not maintained entry points.
Existing `outputs/benchmarks/dexuni/pilot_v1` remains unchanged. The obsolete
`dexuni_benchmark` workspace has been removed; see the [retirement record](physics/experiments/dexuni/MIGRATION.md).

The [published report](https://huggingface.co/datasets/DexForceAI/embodichain_reports/tree/cafe3e14f650f93123d47fe338e49c003443f3a4/reports/dexuni-physics-2026-10-27)
contains runtime inputs, readable reports and the published raw JSON/CSV. The reviewed
[release index](assets/physics/dexuni/publication.json) pins the dataset commit,
file sizes and SHA256 hashes. Downloads restore input assets only; videos and
complete result archives remain local. General assets retain their existing
EmbodiChain resolvers.

Current plans and task descriptions live under `physics/experiments/dexuni/`.
[Initial design snapshots](physics/experiments/dexuni/history/initial_design/INDEX.md)
are retained there as history; the old root workspace has been removed.

See [environment and data setup](physics/experiments/dexuni/SETUP.md) for the
required adaptation PR, independent platform environments and portability limits.
