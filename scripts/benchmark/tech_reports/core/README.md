# Shared benchmark core

Reserved for the common infrastructure proposed in
[issue #679](https://github.com/DexForce/EmbodiChain/issues/679): result/run
contracts, matrix expansion, isolated workers, budgets, measurement boundaries,
resource sampling, artifacts, recovery and comparison eligibility.

This directory contains documentation only. Extract shared helpers incrementally
from existing benchmark implementations after establishing protocol fixtures.
Preserve domain execution, physics, Task Program and persistence ownership.
Support staged operations, continuous stepping, and attempts/episodes with
explicit units, denominators, unsupported states and retained failures.

Platforms execute in separate processes and environments. Pure contracts and
artifact readers must not import simulators or initialize GPUs. Future core
implementation tests belong under `tests/benchmark/core/`.
See the [benchmark layout](../README.md#technical-report-comparison-layout).
