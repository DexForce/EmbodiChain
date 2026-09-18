# Task Catalog and Objectives Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Deliver a shared task catalog/gallery and a separately reported physical objective validated with an expert/replay sample.

**Architecture:** Retain task-first ownership and existing loaders/runtimes. Internal catalog helpers serve CLI/gallery; optional physical objectives integrate with Gym step/reset hooks; a sample runner records independent outcomes.

**Tech Stack:** Python 3.11, PyTorch, YAML, Gymnasium, DexSim, pytest, static HTML.

**Spec:** docs/superpowers/specs/2026-09-18-task-catalog-objectives-design.md

## Global Constraints

- Worktree: /root/sources/EmbodiChain/.worktrees/task-catalog-objectives.
- Interpreter: /root/miniconda3/envs/py311/bin/python. Run with this worktree on PYTHONPATH where needed; verify imports do not resolve production changes from the original checkout.
- Preserve existing task IDs and no-objective behavior, file-owned physics and component field ownership.
- Independent physical outcome never changes authoritative expert completion, segment validation or data persistence.
- New source uses Apache DexForce 2021–2026 header, future annotations, public annotations/docstrings and __all__; configuration classes use @configclass.
- black==26.3.1; root runs black . before any commit. Workers do not commit or spawn agents; root integrates and reviews isolated file ownership before committing.
- Do not read docs/source unless needed under the API documentation skill's explicit coverage workflow; ordinary docs live in task READMEs and generated gallery.
- Tests must exercise real contracts, not merely assert source text. No tutorial tests.

## Task 1: Shared catalog, CLI and gallery

Files: add embodichain/cli/_task_catalog.py and embodichain/cli/show_task.py; modify embodichain/cli/list_task.py and main.py; add tests/test_task_catalog.py and update tests/test_main.py. Add task-local catalog.yaml/README.md for repeated_pick_place, push_cube and stack_cups. Add gallery export from the same model and update package data only if necessary.

Interfaces: catalog loads logical tasks and named deployments with config reference, effective backend/embodiment, supported uses and agent references. show-task accepts a unique task key or qualified package:key. list-task accepts optional category filter and --export-html PATH while preserving default output. Do not add a new launcher shortcut yet: emit existing --gym_config commands.

- [x] Write failing tests with two deployments and one trainer.gym_config; assert only the linked deployment acquires inferred RL support. Test absent catalog fallback and pure config Task Program listing.
- [x] Run `/root/miniconda3/envs/py311/bin/python -m pytest tests/test_task_catalog.py tests/test_main.py -q` and record expected failures.
- [x] Extract current discovery helpers and add strict metadata loading. Keep registered-class capability augmentation separate from static data parsing. Example authored catalog:

```yaml
task_key: repeated_pick_place
title: Repeated Pick and Place
summary: Move a cube between two locations.
tags: [pick_place, single_arm]
default_deployment: franka
deployments:
  franka:
    config: task.franka.yaml
```

- [x] Implement detail output and escaped static HTML cards/deployment tables using the same records. Config commands and source links must refer to packaged real resources.
- [x] Run catalog/CLI/layout tests and record results in the task report.

## Task 2: Optional physical objective

Files: add focused modules below embodichain/lab/gym/envs/objectives/; modify embodied_env.py and gym_utils.py for optional declaration decode and step/reset integration; add tests/gym/envs/test_objectives.py and lifecycle tests.

Interfaces: deployment `objective: {component: objective.yaml}` lowers to an optional EmbodiedEnvCfg field. Runtime exposes reset(env_ids), update(measured_state, dt), snapshot(); typed declaration uses @configclass. `info['physical_objective']` reports independent per-environment success, progress and metrics. Shared report consumer can request a JSON-compatible per-row result. Exact exported names are recorded in the implementation report before Task 3 starts.

- [x] Write tests feeding two rows through A→B→A with required hold time; assert a transient visit does not count, a later exit invalidates final truth, querying twice is pure, partial reset preserves the other row.
- [x] Run focused tests to establish missing behavior.
- [x] Implement strict region/config validation and minimal tensor state machine. Advance after interval events once per control step; reset after episode reset events, before reset info is exposed. Validate object existence during construction.
- [x] Add separate physical namespace to info and demo metadata without changing compute_task_state or is_task_success authority. No-objective tests prove unchanged behavior.
- [x] Run objective tests and existing expert bridge/completion tests, report exact commands and outcomes.

## Task 3: Expert/replay sample and run artifact

Files: modify replay.py only for demonstrated controller-schema compatibility needs; add production sample runner under embodichain/lab/scripts/ and optional CLI adapter if justified; add benchmark deployment/objective config to repeated_pick_place; add focused replay and runner tests.

Interfaces: consume Task 2's objective snapshots and existing demo metadata. Consume runnable config paths, optional seed and bounded initial pose variation. Emit local JSON result and recorded trajectory suitable for dynamic replay; snapshot resolved config/components. Four outcome fields retain unavailable/not-applicable distinctly.

- [x] Establish replay regression tests for existing raw policy actions and expert qpos/qvel controller layouts; cadence mismatch fails before stepping.
- [x] Implement minimal compatibility decoding only if required by recorded schema. Preserve kinematic behavior.
- [x] Add sample objective with measured object regions and explicit stable-duration/final-state semantics. Add baseline/perturbation runner that uses existing reset event path and captures actual post-reset pose.
- [x] Test artifact provenance and independent outcome fields using real objective logic with a small environment double at simulator boundary.
- [ ] Complete one small real headless expert run and dynamic replay. Attempted: native initialization is blocked by the installed DexSim descriptor rejecting `com_quaternion`, before any rollout. The checked-in startup report records unavailable physical outcome; qualification remains pending a compatible DexSim build.

## Task 4: Integration, documentation and review

Files: task READMEs, setup.py only if needed, affected agent_context owners, public API documentation if coverage requires it.

- [x] Generate gallery from sample task catalog and confirm links/deployments/qualification are truthful.
- [x] Run focused integrated regressions, black ., API docs coverage and package build/resource inspection.
- [x] Run `python .agents/skills/project-dev-context/scripts/context.py affected --base origin/main --explain`; update only materially changed ownership guidance and validate context.
- [x] Obtain independent review of the complete diff, fix evidence-backed findings and run covering tests.
- [x] Commit focused changes with validation evidence; report branch/worktree, results, artifact locations and any remaining qualification limitations. Do not push or publish without authorization.

## Validation result

The final targeted regression command passed 123 tests. The broader affected suite
passed 952 tests, with 6 skipped and 18 deselected; 2 failures and 2 setup errors
all originate in the installed DexSim API rejecting `com_quaternion`. The same
descriptor failure was reproduced on the unchanged base checkout. Real physical
qualification remains outstanding; the startup record explicitly reports no
measurement. Independent catalog, objective/replay integration, and runner
reviews have no unresolved findings.

Black 26.3.1 formatted the full repository. API coverage is 2091/2091. Package,
layout and context checks passed, including an unpacked-wheel offline gallery
smoke that imports no simulator modules. Context ownership guidance was revised
only in env-framework; other affected topic contracts remain accurate.
