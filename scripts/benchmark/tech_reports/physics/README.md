# Physics comparisons (P-series)

Directory scaffold for [issue #679](https://github.com/DexForce/EmbodiChain/issues/679).
No physics comparison implementation is added yet.

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
existing owners. Run artifacts belong in `outputs/benchmarks/`.
See the [benchmark layout](../README.md#technical-report-comparison-layout).
