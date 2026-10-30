# EmbodiChain technical report benchmarks

This directory groups technical-report assets, comparison experiments,
and shared execution/reporting designs. For existing runnable benchmarks and
CLI usage, start with the [benchmark guide](../README.md).

## Technical report comparison layout

The L/R/P/G/D families follow
[issue #679](https://github.com/DexForce/EmbodiChain/issues/679). The common framework remains a design. This draft additionally imports the
existing L-02/L-03/L-04 and P-01/P-02/P-03 experiment sources, offline analysis,
HTML builder and planned-experiment specifications. See the
[current report snapshot](reporting/releases/simulation-report-2026-10-28/README.md)
for completion states, evidence locations, unresolved issues and validation.
Report data/media upload is deferred; no dataset revision is claimed.

```text
scripts/benchmark/tech_reports/
  assets/
    README.md                 # asset ownership and publication guide
    DOWNLOAD_DESIGN.md         # proposed input download protocol
    index.json                # empty Hugging Face bootstrap index
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
The shared-framework plan prioritizes R-04 and G-03. The imported local report
suite retains its own L/P experiment IDs and current completion states. Remaining
empty directories use `.gitkeep`; no measurements are invented for pending items.

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
its directory conventions, publishing process, and empty bootstrap index.
The [download design](assets/DOWNLOAD_DESIGN.md) defines future shared
benchmark input preparation, revision pinning, cache behavior, and provenance.

These conventions and the proposed download helper are owned by the benchmark
scripts. The helper and asset CLI are design only; existing benchmark commands
keep their current behavior until actual report inputs are integrated.
