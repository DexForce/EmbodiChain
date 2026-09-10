# Simulation CLI implementation plan

Goal: separate standalone simulation arguments from Gym config overrides and make example seeds explicit and testable.

Architecture: embodichain.cli.sim owns common simulation options and opt-in seed parsing/resolution. The existing Gym launcher composes this API and retains omission sentinels for config overrides. Standalone examples depend on the simulation CLI and expose seeds only for real consumers.

- [x] Add failing tests for CLI ownership, seed validation/resolution, and config preservation.
- [x] Extract common arguments; migrate all standalone consumers including shared tutorial builders and preview-asset.
- [x] Fix cuRobo, grasp-cup, and open-drawer seed consumption; seed standalone Gym tutorials.
- [x] Extract parser builders for independent CLI tests; exercise example parser construction without native simulation.
- [x] Run CLI/Gym/example regression tests, cuRobo smoke runs, formatting and context/API checks.

Validation: standard-library-only --help and default parser validation across 31 scripts; Default/cuRobo single-world and Newton/cuRobo two-world planning and replay succeeded. Fourteen pre-existing Gym config-fixture failures (missing mandatory physics) are excluded from the focused run. Independent review found and corrected fold_tshirt math import ordering.
