# EmbodiChain Review Matrix

Use the common checks for every review, then load only the sections matching
the changed files and their contract neighbors. Prefer `agent_context/MAP.yaml`
and current source code when this matrix and the repository diverge.

## Contents

- [Common checks](#common-checks)
- [Simulation core and scene objects](#simulation-core-and-scene-objects)
- [Gym environments, managers, functors, and randomization](#gym-environments-managers-functors-and-randomization)
- [Task Programs and configured deployments](#task-programs-and-configured-deployments)
- [Atomic actions, motion planning, and IK](#atomic-actions-motion-planning-and-ik)
- [Robots, sensors, and visualization](#robots-sensors-and-visualization)
- [Reinforcement learning and data pipelines](#reinforcement-learning-and-data-pipelines)
- [Official tasks and configuration packages](#official-tasks-and-configuration-packages)
- [Packaging, CLI, workflows, native code, and docs](#packaging-cli-workflows-native-code-and-docs)
- [Agent skills and project context](#agent-skills-and-project-context)

## Common checks

- Preserve the distribution name `embodichain`, the core import package
  `embodichain`, and the bundled task import package `embodichain_tasks`.
- Check the complete call or configuration-resolution path, not only the
  changed definition. Inspect its closest tests and at least one caller or
  consumer.
- Check public Python modules for the Apache header,
  `from __future__ import annotations`, typed public APIs, meaningful Google
  docstrings, static `__all__`, and matching API documentation when the public
  contract changes.
- Check configuration objects for `@configclass`, required fields expressed
  with `dataclasses.MISSING`, safe nested defaults, and `to_dict` / `from_dict`
  round trips where serialization is supported.
- Check file paths relative to the owning config or package, not the reviewer's
  current working directory. Reject unchecked path escape when inputs are not
  trusted.
- Check failure behavior as closely as success behavior: validation should fail
  at the earliest owning boundary with a useful error and without partially
  initialized state.
- Check that tests target observable contracts and include boundary, negative,
  partial-batch, or cleanup cases appropriate to the change.
- Prefer focused tests. Treat GPU, renderer, real-simulation, and distributed
  paths as separate resource classes and report them as residual risk when they
  cannot be run.

## Simulation core and scene objects

**Paths:** `embodichain/lab/sim/**`, plus environment code that owns a
`SimulationManager`.

Check:

- Simulator creation, update, reset, destruction, and cleanup-queue ordering;
  ensure exception paths release scenes, GPU resources, and renderer state.
- Vectorized state shape and `env_ids` semantics, including non-contiguous,
  empty, singleton, and partial-reset selections.
- Tensor dtype/device consistency and avoidance of implicit CPU copies or
  per-step allocations on hot paths.
- Physics-step versus render/update ordering, stale handles after resets, and
  cache invalidation after topology or asset changes.
- World/local/link coordinate frames, meters/radians conventions, quaternion
  order, joint/link indices, mimic joints, limits, and asset scale.
- Scene UID uniqueness, object lookup, articulation/rigid/soft/cloth ownership,
  and behavior when an asset is absent or only partly initialized.

Evidence surfaces: the matching object/config module, `sim_manager.py`,
`base_env.py` or `embodied_env.py`, and the nearest `tests/sim/**` or
simulation-marked test.

## Gym environments, managers, functors, and randomization

**Paths:** `embodichain/lab/gym/**`, especially `envs/managers/**`,
`action_bank/**`, and wrappers.

Check:

- Gymnasium reset/step contracts, observation and action spaces, reward shape,
  `terminated` versus `truncated`, episode counters, and reset ordering.
- Manager execution order and data dependencies. Verify that observation,
  action, reward, event, record, and dataset managers see the intended state.
- Function-style functors accept `(env, env_ids, ...)`; class-style functors
  implement `__init__(cfg, env)` and `__call__(env, env_ids, ...)`.
- All functors respect the selected `env_ids` and do not overwrite unaffected
  rows. Check broadcast parameters and device placement.
- Startup/reset/interval randomization modes, deterministic seeding, legal
  physics/geometry ranges, and independence across environments.
- Wrappers preserve spaces, metadata, reset options, info dictionaries, and
  the underlying completion semantics.
- Task discovery occurs through installed task packages and init hooks; do not
  assume importing `embodichain.lab.gym.envs` registers official tasks.

Evidence surfaces: manager base/config, production registration and component
loaders, the owning task config, and focused `tests/gym/**` tests.

## Task Programs and configured deployments

**Paths:** `embodichain/lab/task_program/**`,
`embodichain/lab/gym/envs/task_program/**`, and task `program.yaml`,
`integration.yaml`, `task.*.yaml`, or component files.

Check:

- Strict language decoding rejects unknown or malformed fields before compile
  or runtime; AST validation, compiler lowering, and runtime execution agree on
  node semantics.
- Registered Semantic Calls have coherent descriptors, schemas, lowerers,
  catalog entries, runtime services, tests, and public exports when applicable.
- Physical and semantic ownership stays separated: reusable `env.yaml` and
  scene components remain physical; integration-owned `scene_binding` contains
  semantic roots and affordances.
- A deployment chooses exactly one of environment component or inline
  environment/scene, and exactly one of embodiment component or inline
  robot/sensor. Component files do not add compatibility `version` fields.
- An embodiment owns one robot, sensors, and optional `skill_profile`; programs
  remain embodiment-independent and trusted integration binds concrete IDs.
- Scene-binding targets exist in the physical scene and preserve canonical
  entity identity through composition, planning, effects, and evidence.
- Parallel branches claim disjoint resources or intentionally synchronize;
  completion masks, cancellation, effect verification, and failure propagation
  work per environment row.
- Relative component paths, execution policies, integration fingerprints,
  package data, and dynamic environment registration survive installation.

Evidence surfaces: language decoder/validation, compiler, semantic catalog,
configured composition/services, Gym registration/bridge, package-data tests,
and the `$add-task-program` read-only deployment inspector.

## Atomic actions, motion planning, and IK

**Paths:** `embodichain/lab/sim/atomic_actions/**`, `planners/**`,
`solvers/**`, and grasp/workspace utilities that feed plans.

Check:

- Goal, options, affordance, requirement, binding, plan, command, effect, and
  evidence types remain coherent across registration, planning, compilation,
  execution, tracking, and verification.
- Planning is side-effect free; execution does not silently re-plan or bypass
  typed validation. Failures preserve useful causes and do not emit partial
  unsafe commands.
- Resource claims, endpoint resolution, invocation revisions, per-row
  eligibility, cancellation, safe stop/hold, and recovery policies remain
  consistent under partial failure.
- Trajectory timing is explicit and consistent (`dt`, interpolation, velocity,
  acceleration); concatenation does not duplicate or skip boundary samples.
- Planner collision worlds use canonical logical IDs, refresh dynamic
  obstacles, handle empty geometry, and preserve batch-mode expectations.
- IK/FK respects base/tool frames, joint order and limits, batch shapes,
  convergence/failure signaling, singularities, dtype/device, and backend
  differences.
- New actions or solvers include registration, package exports, documentation,
  focused unit tests, and benchmarks when performance claims are part of the
  contract.

Evidence surfaces: typed core contracts, engine/runner, simulation adapter,
planner/solver base class, registration/export modules, and the nearest
`tests/sim/atomic_actions/**`, planner, solver, or toolkit test.

## Robots, sensors, and visualization

**Paths:** `embodichain/lab/sim/robots/**`, `objects/robot.py`,
`sensors/**`, and `embodichain/lab/visualization/**`.

Check:

- `RobotCfg` defaults, URDF paths, serial-chain construction, control-part
  definitions, joint-name/index mapping, drive properties, end effectors, and
  supported variants stay aligned.
- Robot and embodiment configuration round trips do not lose nested solver,
  drive, sensor, or skill-profile data.
- Sensor outputs match declared shape, dtype, device, frame, intrinsics,
  extrinsics, clipping/range semantics, and update frequency for every backend.
- Camera/depth/point-cloud conversion handles batched environments, invalid
  pixels, axis conventions, and headless or renderer-unavailable operation.
- Visualization is observational: starting, refreshing, or disconnecting the
  browser must not change simulation state. Topology changes and resource
  teardown must not leave stale nodes, callbacks, or background tasks.

Evidence surfaces: base and concrete cfg/runtime classes, embodiment component
composition, asset-preview or visualization entry points, and focused robot,
sensor, or visualization tests.

## Reinforcement learning and data pipelines

**Paths:** `embodichain/learning/**`, `embodichain/data_pipeline/**`, and
environment managers that record datasets.

Check:

- Rollout time axes, environment axes, actions, rewards, values, log-probs,
  dones, truncations, bootstrap values, masks, and advantage normalization
  remain aligned.
- Differentiable paths avoid unintended `detach`, in-place autograd mutation,
  device transfers, or non-differentiable environment adapters; ordinary PPO
  paths do not accidentally retain graphs.
- Collector/trainer routing selects the intended algorithm and rollout kind;
  checkpoint save/resume restores models, optimizers, counters, normalizers,
  RNG state, and configuration needed for equivalent continuation.
- Evaluation does not mutate training state and handles vector completion,
  deterministic policies, and task discovery consistently with training.
- Distributed workers initialize devices and seeds correctly, synchronize only
  intended state, propagate failures, and clean up process groups.
- Dataset schemas, episode boundaries, timestamps, image encoding, buffering,
  asynchronous finalization, error propagation, and partial writes preserve
  data integrity and bounded memory.

Evidence surfaces: algorithm, buffer, collector, trainer, RL environment,
evaluation, dataset manager/writer, and focused `tests/learning/**` or dataset
tests. Run GPU/distributed tests serially and only when justified.

## Official tasks and configuration packages

**Paths:** `embodichain_tasks/**` and reusable task components.

Check:

- Preserve the task-first layout under a task family and optional subdomain;
  do not organize tasks by solution method.
- Keep `@register_env` in the task-named module when a Python entry point is
  required. Do not add a same-named package or Python `scenario`/`mdp` layers
  when configuration and manager functors own the behavior.
- Configuration-defined Task Programs may omit a Python task module only when
  their runnable deployment supplies the required environment, embodiment,
  program, integration, and execution policy composition.
- Environment IDs and registration are unique and discoverable after package
  installation, not only from the source tree.
- JSON/YAML is validated through production strict loaders and component
  composition, not only `yaml.safe_load`. Relative files and scene targets must
  resolve after wheel installation.
- New or moved config/assets are present in wheel contents; deleted files are
  not retained accidentally. Imports use `embodichain_tasks`, not a repository
  folder assumption.

Evidence surfaces: the task-named module, deployment/component files,
registration utilities, `pyproject.toml`/`setup.py`, task layout tests, package
data tests, and wheel-content validation.

## Packaging, CLI, workflows, native code, and docs

**Paths:** `pyproject.toml`, `setup.py`, `embodichain/__main__.py`,
`.github/**`, `scripts/**`, `docs/**`, and C++/CUDA extension sources.

Check:

- CLI commands discover task packages and initialize them before consuming
  environment IDs; exit status, stderr/stdout, path handling, and optional
  dependency failures remain useful.
- Python dependencies, optional extras, entry points, versions, build backend,
  native extension flags, wheel membership, and import paths agree across
  source and installed artifacts.
- Native interfaces preserve shape/dtype/device/contiguity contracts, lifetime
  ownership, CPU/GPU fallback, error propagation, and build compatibility.
- Workflow changes preserve event filters, permissions, secrets isolation,
  cache keys, artifacts, concurrency, and the intended separation of lint,
  docs, non-simulation, simulation, distributed, GPU, and release jobs.
- Public API changes are reachable from documented import paths and pass the
  read-only API docs checker. Documentation examples use current configuration
  and package names.

Evidence surfaces: CLI dispatch and tests, package metadata/build scripts,
workflow-specific tests or `actionlint`, API docs checker, Sphinx dummy build,
and a built wheel for packaging changes.

## Agent skills and project context

**Paths:** `.agents/skills/**`, `agent_context/**`, `.claude/skills/**`,
`.github/copilot/**`, and `AGENTS.md`.

Check:

- `.agents/skills/<name>/SKILL.md` remains the canonical implementation;
  Claude and Copilot adapters stay thin and point to it instead of duplicating
  instructions.
- Skill names are lowercase kebab-case, frontmatter contains only `name` and
  `description`, trigger wording is specific, and `agents/openai.yaml` matches
  the canonical behavior.
- References are linked directly from `SKILL.md`, scripts are deterministic and
  tested, and no auxiliary README or process-history files are added.
- `agent_context/MAP.yaml` resolves topics by ID, aliases, then keywords; paths
  and `source_of_truth` remain current. Behavior changes update the mapped topic
  and routing adapters when required by the context update contract.
- New canonical skills pass `quick_validate.py`; adapters and the project skill
  index expose them consistently.

Evidence surfaces: the canonical skill, metadata, thin adapters, context map
and topic files, project instructions, and the Skill Creator validator.
