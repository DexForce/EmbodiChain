# Simulation-Level Affordance Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible geometric Affordance sampling to batched simulation Atomic Actions and demonstrate it in the parallel Atomic Action tutorials without Task Program or Gym integration.

**Architecture:** A host-owned `AffordanceSamplingContext` flows through `PlanningContext`. Each `Affordance` samples legal pose candidates into one `AffordanceSample`; Atomic Actions intersect that success mask with IK and trajectory feasibility and report namespaced diagnostics. The direct Atomic Action tutorials map logical branches one-to-one to simulation rows through shared host helpers.

**Tech Stack:** Python 3.11, PyTorch, pytest, EmbodiChain Atomic Actions, Sphinx.

**Spec:** `docs/superpowers/specs/2026-09-16-sim-affordance-sampling-design.md`

## Global Constraints

- Do not modify or import Task Program, Gym, task deployment, dataset, recorder,
  or episode collection production code. Compatibility-only integration-test
  updates are allowed when an Atomic Action internal contract changes.
- Keep Slide distance, Twist angle, and OpenDoor opening fraction exact; do not add action/goal range sampling.
- Keep existing tensor-returning Affordance pose helpers compatible.
- Expose candidate sampling only through `Affordance.sample_candidates()`.
- Use tests first and observe the expected failure before each production change.

---

### Task 1: Sampling Value Contracts and Diversity Selection

**Files:**
- Create: `embodichain/lab/sim/atomic_actions/affordance_sampling.py`
- Create: `tests/sim/atomic_actions/test_affordance_sampling.py`

**Interfaces:**
- Produces: `AffordanceSamplingContext`, `AffordancePoseCandidates`, and `AffordanceSample`.
- Produces: private candidate selection and contact-frame roll helpers used by `affordance.py`.

- [ ] **Step 1: Write failing tests for validation and immutable result ownership**

Add tests constructing valid and invalid contexts, candidates, and samples. Assert
`success` has shape `(B,)`, `poses` has shape `(B, 4, 4)`, tensors share a
device, metadata is copied, and candidate padding stays invalid.

- [ ] **Step 2: Run the new module test and verify import failure**

Run: `conda run -n open pytest -q tests/sim/atomic_actions/test_affordance_sampling.py`

Expected: collection fails because `affordance_sampling` does not exist.

- [ ] **Step 3: Implement the three public value types**

Implement self-contained validation without an environment configuration type.
Use private CPU generators and owned output snapshots.

- [ ] **Step 4: Write and run failing behavioral tests for deterministic sampling**

Cover branch-zero nominal behavior, distinct candidate selection, row-order
independence, attempt/key stream changes, duplicate reuse, empty rows, and no
global RNG mutation. Expected: failures because no public Affordance entry point
exists yet.

- [ ] **Step 5: Keep the algorithm private for Task 2**

Add `_sample_pose_candidates()` and `_roll_poses()` as non-exported helpers.
Rerun the value-contract tests; the behavioral entry-point tests remain red.

### Task 2: Affordance-Owned Sampling API

**Files:**
- Modify: `embodichain/lab/sim/atomic_actions/affordance.py`
- Modify: `embodichain/lab/sim/atomic_actions/__init__.py`
- Modify: `tests/sim/atomic_actions/test_affordance.py`
- Modify: `tests/sim/atomic_actions/test_affordance_sampling.py`

**Interfaces:**
- Consumes: sampling value types from Task 1.
- Produces: `Affordance.sample_candidates(...) -> AffordanceSample`.
- Produces: `AntipodalAffordance.get_grasp_candidates(...)`.
- Produces: specialized Press, Twist, Assemble, and InteractionPoints samplers.

- [ ] **Step 1: Add failing tests for the exact public API**

Assert candidate selection is invoked through `Affordance`, returns
`AffordanceSample`, preserves success/metadata, and has no public candidate
`select()` method. Add tests proving existing exact pose helpers remain exact.

- [ ] **Step 2: Run the focused tests and verify missing-method failures**

Run: `conda run -n open pytest -q tests/sim/atomic_actions/test_affordance_sampling.py tests/sim/atomic_actions/test_affordance.py`

- [ ] **Step 3: Implement the base and antipodal APIs**

Add `sample_candidates()` to `Affordance` and candidate generation to
`AntipodalAffordance`. Preserve generator ownership outside the Affordance.

- [ ] **Step 4: Implement specialized geometric samplers**

Add sampling helpers for press roll, declared twist roll, assembly symmetry,
and interaction points. Return success and metadata in every path.

- [ ] **Step 5: Export and rerun tests**

Export the three public sampling values from the package. Run both focused test
modules and require green output.

### Task 3: Planning Context Propagation

**Files:**
- Modify: `embodichain/lab/sim/atomic_actions/state.py`
- Modify: `embodichain/lab/sim/atomic_actions/engine.py`
- Modify: `tests/sim/atomic_actions/test_core.py`
- Modify: `tests/sim/atomic_actions/test_engine.py`

**Interfaces:**
- Consumes: `AffordanceSamplingContext`.
- Produces: optional `PlanningContext.affordance_sampling` preserved by `project()`.
- Produces: `AtomicActionEngine.initial_context(..., affordance_sampling=...)`.

- [ ] **Step 1: Add failing propagation and validation tests**

Test wrong types, direct initial-context injection, and preservation across
offline projection.

- [ ] **Step 2: Run the focused tests and verify expected constructor failures**

Run: `conda run -n open pytest -q tests/sim/atomic_actions/test_core.py tests/sim/atomic_actions/test_engine.py`

- [ ] **Step 3: Implement minimal context and engine changes**

Store the immutable context without introducing global action hooks or
Task Program providers.

- [ ] **Step 4: Rerun focused tests**

Require both modules to pass.

### Task 4: Atomic Action Integrations

**Files:**
- Modify: `embodichain/lab/sim/atomic_actions/primitives/pick_up.py`
- Modify: `embodichain/lab/sim/atomic_actions/primitives/axis_align.py`
- Modify: `embodichain/lab/sim/atomic_actions/primitives/slide.py`
- Modify: `embodichain/lab/sim/atomic_actions/primitives/open_door.py`
- Modify: `embodichain/lab/sim/atomic_actions/primitives/hand_over.py`
- Modify: `embodichain/lab/sim/atomic_actions/primitives/press.py`
- Modify: `embodichain/lab/sim/atomic_actions/primitives/twist.py`
- Modify: `embodichain/lab/sim/atomic_actions/primitives/place.py`
- Modify: `tests/sim/atomic_actions/test_actions.py`
- Modify: `tests/lab/task_program/test_semantic_compiler.py`

**Interfaces:**
- Consumes: Affordance sampling helpers and `PlanningContext.affordance_sampling`.
- Produces: row-local success intersection and `affordance_sample` plan diagnostics.

- [ ] **Step 1: Add failing action tests**

Use CPU fakes to prove PickUp/AxisAlign/Slide/OpenDoor candidate selection,
Press/Twist pose sampling, and Assemble symmetry preserve row-local sample
failure and metadata. Explicit grasp overrides must remain exact.

- [ ] **Step 2: Run the targeted action tests and observe failures**

Run selected node IDs in `tests/sim/atomic_actions/test_actions.py` covering the
new sampling behaviors.

- [ ] **Step 3: Integrate grasp-candidate actions**

Build `AffordancePoseCandidates`, apply action-owned geometric/IK validity, call
`affordance.sample_candidates()`, and carry the returned sample through plan
construction.

- [ ] **Step 4: Integrate direct-pose actions**

Use specialized Affordance samplers for Press, Twist, and Assemble. Merge
metadata under `diagnostics.metadata["affordance_sample"]`.

- [ ] **Step 5: Rerun all Atomic Action tests**

Run: `conda run -n open pytest -q tests/sim/atomic_actions`

Expected: existing and new tests pass; sim/GPU-specific tests retain their
normal skip/deselection behavior.

### Task 5: Parallel Atomic Action Tutorials

**Files:**
- Modify: `scripts/tutorials/atomic_action/tutorial_utils.py`
- Modify: `scripts/tutorials/atomic_action/axis_align.py`
- Modify: `scripts/tutorials/atomic_action/hand_over.py`
- Modify: `scripts/tutorials/atomic_action/open_door.py`
- Modify: `scripts/tutorials/atomic_action/pickup.py`
- Modify: `scripts/tutorials/atomic_action/press.py`
- Modify: `scripts/tutorials/atomic_action/slide.py`
- Modify: `scripts/tutorials/atomic_action/twist.py`

**Interfaces:**
- Consumes: engine sampling-context injection.
- Produces: CLI arguments `--affordance_branches`, `--sampling_seed`, and `--sampling_attempt` for direct simulation demonstration.

- [ ] **Step 1: Add shared host wiring**

Centralize branch-count validation, branch-to-`num_envs` wiring, deterministic
context construction, and diagnostic logging in `tutorial_utils.py`.

- [ ] **Step 2: Integrate the supported tutorials**

Use the shared helpers in PickUp, AxisAlign, HandOver, OpenDoor, Press, Slide,
and Twist while leaving task-level parameter ranges exact.

- [ ] **Step 3: Run static tutorial validation**

Run the repository's existing syntax and import checks. Do not add or expand
tests whose primary subject is a tutorial script.

### Task 6: Public and Project Documentation

**Files:**
- Modify: `docs/source/api_reference/public_api.rst`
- Create: `docs/source/overview/sim/atomic_actions/affordance_sampling.md`
- Modify: `docs/source/overview/sim/atomic_actions/index.md`
- Modify: `agent_context/MAP.yaml`
- Create: `agent_context/topics/atomic-actions/affordance-sampling.md`
- Modify: `agent_context/topics/atomic-actions/atomic-actions.md`

**Interfaces:**
- Documents: public sampling values, ownership boundary, direct tutorial, and deferred Task Program/Gym work.

- [ ] **Step 1: Run the API checker and capture missing exports**

Run: `python docs/scripts/check_api_docs.py --format json`

- [ ] **Step 2: Add scoped API, overview, and project-context documentation**

Document only implemented sim behavior. Do not claim episode collection,
recording, retry, or Task Program support.

- [ ] **Step 3: Validate docs and project context**

Run:

```bash
python docs/scripts/check_api_docs.py
conda run -n open pytest -q tests/docs/test_check_api_docs.py --confcutdir=tests/docs
conda run -n open python -m sphinx -b dummy docs/source docs/build/api-docs-check
python .agents/skills/project-dev-context/scripts/context.py check
conda run -n open python -m pytest -q -c /dev/null --noconftest tests/test_agent_context_map.py tests/test_agent_context_tools.py
```

### Task 7: Final Validation and Pull Request

**Files:**
- Review all files changed relative to `origin/main`.

**Interfaces:**
- Produces: one focused commit series and one PR targeting `main`.

- [ ] **Step 1: Run formatting and structural checks**

Run `conda run -n open black .`, inspect the diff, then run
`conda run -n open black --check --diff --color ./`.

- [ ] **Step 2: Run the complete focused validation set**

Rerun Atomic Action tests, API docs tests/checker, Sphinx dummy build, and
project-context validation from Tasks 4 and 6.

- [ ] **Step 3: Audit scope**

Confirm `git diff --name-only origin/main...HEAD` contains no Task Program, Gym,
task deployment, dataset, recorder, or episode-collection production files.
Any Task Program test-only change must be limited to compatibility with the
Atomic Action contract.

- [ ] **Step 4: Commit, push, and create the PR**

Use conventional commits, push `codex/sim-affordance-sampling`, create a PR with
base `main`, report exact validation, and apply available feature/agent labels.
