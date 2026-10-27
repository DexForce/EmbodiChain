# Loading comparisons (L-series)

Directory scaffold for [issue #679](https://github.com/DexForce/EmbodiChain/issues/679).
No loading comparison implementation is added yet.

- `experiments/`: experiment definitions, scene-size matrices, cold/warm/reset
  protocols, and comparison orchestration.
- `platforms/`: platform-specific loading, rebuild, and reset execution adapters.
- `../assets/loading/`: small authored input descriptions and asset references.

Implement one representative experiment after the R-04/G-03 pilots. Separate
stages and cold/warm conditions, record latency, memory and failure points, and
use matched asset/configuration hashes across platforms. Run each platform in
its own process and Python environment. Write run artifacts under
`outputs/benchmarks/`; reuse the planned common measurement/reporting contracts.

See the [benchmark layout](../README.md#technical-report-comparison-layout).
