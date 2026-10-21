# EmbodiChain benchmarks

Run benchmarks from a source checkout through `embodichain benchmark` or
`python -m scripts.benchmark`. Use `embodichain benchmark --help` to list the
available benchmark families and each subcommand's `--help` for its options.
Individual benchmark directories contain their own workload and result guidance.

## Technical report comparison layout

The L/R/P/G/D families follow
[issue #679](https://github.com/DexForce/EmbodiChain/issues/679). This change
creates their directory scaffold and ownership notes. The common framework,
new experiments, and cross-platform adapters are not implemented yet.

```text
scripts/benchmark/
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
    offscreen_denoising.py     # existing runnable benchmark
    settings/                 # existing denoising configuration
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
| R | [rendering](rendering/README.md#technical-report-comparisons-r-series) | `assets/rendering/` | R-04 batched camera rendering, then R-03/R-05/R-06/R-09. |
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
Start with the R-04 and G-03 end-to-end slices. Other experiment folders are
reserved for incremental implementation. Empty directories are tracked with
`.gitkeep`; they contain no runnable placeholder scripts or invented results.

Existing `motion_generation/`, `atomic_action/`, `rl/`, `task_program/`, and
other benchmarks keep their entry points and outputs. Reuse their domain
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
