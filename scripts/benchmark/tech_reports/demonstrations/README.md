# End-to-end demonstrations (D-series)

Directory scaffold for [issue #679](https://github.com/DexForce/EmbodiChain/issues/679).
The planned D-03 workflow assembles existing results, stage evidence and video
artifacts; it does not establish new performance measurements by itself.

- `experiments/`: demonstration definitions, evidence selection and assembly.
- `platforms/`: platform-specific evidence import/recording adapters when needed.
- `../assets/demonstrations/`: small input definitions and evidence references.

Preserve frozen run IDs, configurations, asset hashes, stage outcomes and
unedited recordings. Put generated indexes, reports and media in
`outputs/benchmarks/`. Pure assembly/report modules must work from frozen
artifacts without importing a simulator or initializing a GPU. RL/IL training,
generalization and sim-to-real claims remain outside this initial scaffold.

See the [benchmark layout](../README.md#technical-report-comparison-layout).
