# Expert generation comparisons (G-series)

Directory scaffold for [issue #679](https://github.com/DexForce/EmbodiChain/issues/679).
G-03 expert trajectory generation is the first planned slice, followed by G-04.
No new generation workflow or comparison adapter is implemented here.

- `experiments/`: experiment definitions, fixed attempt/episode budgets, task
  matrices, quality protocols, and comparison orchestration.
- `platforms/`: platform-specific execution, validation and persistence adapters.
- `../assets/expert_generation/`: small source-demo/task descriptions and references.

Consume the production trajectory-variant, execution and confirmed-persistence
path owned by [#670](https://github.com/DexForce/EmbodiChain/issues/670), and
reuse domain evaluators from existing motion-generation, Atomic Skill and Task
Program code. Keep candidate generation and persistence state machines in their
production owners. Separate execution status, physical outcomes, validation
and persistence receipts. Only validated, confirmed-persisted episodes enter
accepted yield; retries retain failures and consume the fixed budget.

Run each platform in its own process and Python environment. Run records and
generated datasets belong in caller-owned output directories under
`outputs/benchmarks/`, with lineage and duplicate-receipt handling preserved.
See the [benchmark layout](../README.md#technical-report-comparison-layout).
