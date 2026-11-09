# Test-suite runtime measurements

This internal CI benchmark measures fresh-process wall time, including imports,
collection, native test discovery, assertions, cleanup and process exit. Run it
from a configured source checkout with the same dependencies, assets, display,
GPU selection and compiler/cache settings for both compared versions.

## Measure the root Python suite

```bash
python scripts/benchmark/testing/suite_runtime.py \
  --root /path/to/EmbodiChain \
  --python /path/to/runtime/bin/python \
  --docs-python /path/to/docs-runtime/bin/python \
  --output /tmp/test-runtime-candidate-1 \
  --label candidate \
  --cpu-workers 8 \
  --cuda-visible-devices 0 \
  --freeze-native-imports
```

Use a new output directory outside the checkout, or a Git-ignored directory.
The runtime needs pytest-xdist and the project's qualified native dependencies.
The separate docs interpreter avoids conflicts between documentation and SDK
dependencies. CPU workers are configurable; GPU and simulation tests stay serial.
CI uses `EMBODICHAIN_CPU_TEST_WORKERS` to override its default of eight workers.

`--freeze-native-imports` applies only to the owned native test processes that
already exit without interpreter finalization. Static dependencies are frozen
before importing project or test modules; lifecycle collection stays active for
the latter. Successful optimized exits also check that native resources have
been released. For the original runner, use four CPU workers with neither
duration history nor import freezing, as shown below.

## Include the independent suites

The default command covers the root Python fast, distributed, GPU, simulation
and documentation lanes. A complete repository measurement must also pass an
extra plan with `--extra-plan /path/to/all-suites.json`:

```json
[
  {"name": "exporter", "argv": ["python", "-m", "pytest", "-q", "-c", "tools/lerobot_export/pyproject.toml", "tools/lerobot_export/tests"]},
  {"name": "architecture-unit", "argv": ["npm", "test"], "cwd": "docs/architecture/web"},
  {"name": "architecture-browser", "argv": ["npm", "run", "test:browser"], "cwd": "docs/architecture/web"},
  {"name": "taskgraph-unit", "argv": ["npm", "test"], "cwd": "web/task-engine"},
  {"name": "taskgraph-browser", "argv": ["npm", "run", "test:browser"], "cwd": "web/task-engine"},
  {"name": "sphinx-browser", "argv": ["npm", "run", "test:docs"], "cwd": "docs/architecture/web"}
]
```

Install each frontend's dependencies and Playwright browsers first. The exporter
must be installed in the selected runtime. `python` in extra commands resolves
through that runtime's executable directory. The Sphinx browser command may
need an explicit docs-interpreter `PATH` in its `argv` when the native runtime
cannot also contain the documentation dependencies.

## Compare versions

Freeze each checkout and use the same harness and metrics plugin for both
versions. The original runner supports only four CPU workers. Its baseline
command uses the candidate's measurement harness and metrics plugin, with no
duration history or import freezing:

```bash
python /path/to/candidate/scripts/benchmark/testing/suite_runtime.py \
  --root /path/to/baseline \
  --python /path/to/baseline-runtime/bin/python \
  --docs-python /path/to/docs-runtime/bin/python \
  --output /tmp/test-runtime-baseline-1 \
  --label baseline \
  --cpu-workers 4 \
  --cuda-visible-devices 0 \
  --metrics-plugin /path/to/candidate/scripts/ci/pytest_metrics.py \
  --extra-plan /path/to/all-suites.json
```

For optimized repeats, use eight workers and import freezing as above, plus
`--cpu-history /path/to/fixed-history` when measuring duration-based ordering.
Keep that history fixed throughout the optimized repeats. Duration hints reorder
whole CPU groups without changing order inside a group. Worker count, history
and freezing are intentional policy differences and must be reported with the
result; all other environment and input settings should match.

Run at least three complete samples per version and record the actual order,
cache policy, first-use compilation, worker counts and any interruptions. Compute
the total reduction as `1 - median(candidate seconds) / median(baseline seconds)`.
Per-test phase sums help locate costs but do not equal parallel wall time, and
percentages from different stages must not be added.

Each sample writes `result.json`, a three-table `report.md`, environment/source
fingerprints, step logs and phase metrics. Failed steps remain failed while
later steps still run. A native abort may leave incomplete or unattempted cases;
do not treat those as passes or infer a complete-flow time from a partial run.
Disclose all failed/interrupted attempts rather than replacing them silently.

Review failures, skips, removed/merged coverage and native resource observations
alongside timing. The coverage decisions for this change are recorded in
[`.ci/test-efficiency-coverage.json`](../../../.ci/test-efficiency-coverage.json).
The benchmark does not measure process-tree or GPU memory peaks; raw process RSS
high-water marks must not be described as either.
