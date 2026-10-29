# Technical-report benchmark maintenance

Follow the directory ownership in README.md and upstream PR #832.

- Definitions and orchestration: physics/experiments/dexuni.
- Platform execution: physics/platforms/dexuni; each simulator runs in its own environment.
- Offline analysis and media assembly: reporting/dexuni; imports must not initialize a simulator.
- Small authored inputs, publication indexes and provenance references: assets.
- Generated data, videos, reports and local diagnostic evidence: outputs/benchmarks/tech_reports.
- Publish large report artifacts to DexForceAI/embodichain_reports, with verified revisions and hashes. Do not put them in Git.

Current task definitions are S01-C/T, S02–S04, M01–M06 and C01–C02.
C03 has been removed. Historical specifications are reference material only;
PLAN.md and each task's current README own execution scope.

Preserve accepted inputs and raw measured outputs. Keep measurement and media
recordings separate. Report unsupported and failed conditions explicitly.
C02 contains preliminary timing, not five-process formal measurements; its
24×12 full video fails the task and must not be presented as a successful run.
Iteration counts across different solver strategies are not equal units of work.

Use existing CPU protocol/path tests for orchestration or reporting changes.
Run GPU validation only when changing geometry, controllers or physical settings.
Do not terminate unrelated jobs or silently run measurements under GPU contention.
Do not rewrite frozen source hashes when current source changes.
