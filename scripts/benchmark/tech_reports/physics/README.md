# Physics comparisons (P-series)

Directory scaffold for [issue #679](https://github.com/DexForce/EmbodiChain/issues/679).
The [DexUni suite](experiments/dexuni/README.md) runs S01-C/T with one fixed
configuration and five independent repeats per platform. Native control
implementations are retained; see its PLAN for comparison limits.

- `experiments/`: experiment definitions, physical scenarios, solver/step
  matrices, quality protocols, and comparison orchestration.
- `platforms/`: platform-specific stepping and evidence-collection adapters.
- `../assets/physics/`: small scenario inputs, input descriptions and references.

Begin with one representative physics experiment. Align timestep, solver,
geometry, materials, and other declared invariants before comparisons. Record
analytical error, convergence, physical/task quality and quality-qualified
performance separately. Changed geometry or dynamics requires physical
execution validation. Unsupported conditions retain reasons instead of zero
measurements. Each platform runs in its own process and Python environment.

Production physics, environment components, and task validators retain their
existing owners. Current DexUni run artifacts belong in repository-relative
`outputs/benchmarks/tech_reports/dexuni/`.
See the [benchmark layout](../README.md#technical-report-comparison-layout).
