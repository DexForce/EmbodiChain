# Publication handoff pilot: independent audit

Frozen implementation: `8f101b29738598fd83c01a5b4882888b7d87a5b9`. Qualified runs: 10/10.

One seed; fixed 128 updates, 4,096 chunks and 131,072 valid action targets per run. Evaluation is disabled; no policy-quality or learning-efficiency conclusion.

| Mode | Vision | Total s | First update s | Trained / saved | Early updates | All-source publish→train p50 / p95 s | Post-initial p50 / p95 s | Staging wait s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| eager | dual | 409.41 | 65.98 | 32 / 40 | 48 | 6.289 / 24.338 | 6.289 / 19.067 | 75.53 |
| generation_control | dual | 335.00 | 66.48 | 32 / 32 | 0 | 24.029 / 47.079 | 24.030 / 47.079 | 100.37 |
| ods_overlap | dual | 334.54 | 65.49 | 32 / 32 | 48 | 7.698 / 24.569 | 7.698 / 22.576 | 66.86 |
| ods_serial | dual | 334.32 | 64.70 | 32 / 32 | 0 | 20.357 / 37.282 | 20.357 / 37.282 | 65.60 |
| offline32 | dual | 350.91 | 330.62 | 32 / 32 | 0 | 208.472 / 288.735 | 186.739 / 240.146 | 68.60 |
| eager | single | 242.15 | 55.39 | 32 / 40 | 48 | 0.285 / 19.467 | 0.279 / 0.335 | 16.77 |
| generation_control | single | 209.34 | 57.57 | 32 / 32 | 0 | 9.997 / 20.279 | 9.662 / 19.650 | 18.53 |
| ods_overlap | single | 204.84 | 56.99 | 32 / 32 | 48 | 0.287 / 19.691 | 0.271 / 0.310 | 15.88 |
| ods_serial | single | 200.09 | 54.79 | 32 / 32 | 0 | 9.668 / 19.345 | 9.668 / 19.269 | 6.17 |
| offline32 | single | 216.94 | 201.23 | 32 / 32 | 0 | 97.669 / 165.106 | 79.794 / 125.089 | 17.81 |

## Interpretation boundaries

- Publication latency and early-update counts are independently reconstructed from completed optimizer events and the last publication timestamp in each generation.
- All-source latency includes initial fill and shared normalization. Post-initial latency includes only sources from generation > 1 that actually completed training. Untrained published sources remain separately counted and do not receive invented zero latency.
- The four recorder substages partition SDK episode service. Their sums must not be added to overlapping generation/training wall time.
- Serial preserves recording/generation overlap and reproduces numeric sources and ordered update work. Raw RGB equality is not established, even if encoded or decoded YUV hashes happen to agree.
- One seed and shared-host measurements do not establish a stable speedup. State coverage and accepted source counts remain separate metrics.
- Earlier smoke-v1 used the older serial barrier and is engineering qualification only; it is excluded here.
- single paired serial contract: PASS; overlap time saving -2.38% (negative means slower).
- dual paired serial contract: PASS; overlap time saving -0.06% (negative means slower).
