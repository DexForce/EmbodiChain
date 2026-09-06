# EmbodiChain — Developer reference

## Project context

Read `agent_context/MAP.yaml` first for project context or code navigation,
then follow [.agents/skills/project-dev-context/SKILL.md](.agents/skills/project-dev-context/SKILL.md).
Load only the matched overview; follow its detail links when the request needs
them. Verify relevant facts against current code before recommending a change.
`MAP.yaml` is the sole topic inventory; conventions are loaded for context
maintenance, not ordinary reads. Do not read `docs/source/` unless the user
asks for Sphinx documentation.

Canonical skills live under `.agents/skills/`. `.claude/skills/` and
`.github/copilot/` contain thin adapters, not independent instructions.

## Package and architecture

The distribution and primary import package are **`embodichain`**, lowercase
and one word. The repository directory is `EmbodiChain`; the same wheel
bundles official tasks as the **`embodichain_tasks`** import package.

| Area | Owner |
|---|---|
| CLI dispatch / task discovery | `embodichain/cli/` |
| Shared numerical algorithms | `embodichain/compute/` |
| Assets and download registry | `embodichain/data/` |
| Online datasets and depth video | `embodichain/data_pipeline/` |
| Scene Engine and SimReady generation | `embodichain/gen_sim/` |
| Simulation world, objects, sensors, solvers, planning | `embodichain/lab/sim/` |
| Gym environments and manager functors | `embodichain/lab/gym/` |
| Task Program language, semantics, compiler, runtime and integrations | `embodichain/lab/task_program/` |
| Browser visualization | `embodichain/lab/visualization/` |
| RL algorithms, policies, collectors and trainers | `embodichain/learning/rl/` |
| Real-device controllers / standalone tools | `embodichain/lab/devices/`, `embodichain/toolkits/` |
| Shared config and math utilities | `embodichain/utils/` |
| Official task entry points / components / deployments | `embodichain_tasks/` |

Shared numerical algorithms belong to `compute/<domain>/`, with Warp kernels
under private `_warp/` packages. Compute must not import `lab`, simulation
objects or environment managers. Stateful solvers remain in `lab/sim/motion/solvers`;
contact-data adaptation belongs to sensors. New trajectory consumers import
`embodichain.compute.trajectory`; existing `utils/warp` and pure
`lab/sim/utility/action_utils` exports remain compatibility surfaces.

Motion APIs live under `embodichain.lab.sim.motion.{solvers,planners,workspace,trajectory_augmentation}`.
`motion/motion_generator.py` owns `MotionGenerator`, `MotionGenCfg`, and
`MotionGenOptions`, composing the planner backends. Planners do not re-export it.
The motion parent and workspace analyzer exports remain lazy to preserve Robot
initialization. Trajectory augmentation owns candidates, operators, coverage and
generation bookkeeping; execution, reset and persistence belong to host integrations.

## Task ownership

Organize official tasks by family, optional subdomain and task identity.
Keep import registration in the task-named module at
`embodichain_tasks/embodichain_tasks/<category-path>/<task>.py`; configs belong
under `embodichain_tasks/configs/tasks/<category-path>/<task>/`. Do not add a
same-named Python package for one entry point or `scenario` / `mdp` modules when
existing config fields and manager functors express the task.

Physical environments, robot/sensor embodiments, and Task Program semantic
integration have separate owners. Component selections cannot duplicate the
inline fields they own. A supported configuration-defined Task Program may
omit the Python task module. Follow the authoritative
[configuration and registration contract](agent_context/topics/env-framework/configuration.md)
and `/add-task-env` before adding a deployment.

## Code and validation

- Run **`black==26.3.1`**, using `black .`, before every commit.
- Use `/pre-commit-check` for proportional checks. New fixes/features need
  focused tests that prove their behavior; use `/add-test` for project patterns.
- New source files use the Apache 2.0 copyright header from
  [.agents/skills/pre-commit-check/SKILL.md](.agents/skills/pre-commit-check/SKILL.md),
  with DexForce's 2021–2026 copyright. Preserve existing third-party licenses.
- Add `from __future__ import annotations`; fully annotate public APIs, prefer
  `A | B`, and guard circular imports with `TYPE_CHECKING`.
- Use `@configclass` for configuration objects. Annotated `MISSING` defaults
  require `cfg.validate()` after assembly; construction may leave them unresolved.
- Define `__all__` for public modules. Use Google-style docstrings with Sphinx
  directives where they explain non-obvious behavior.
- Manager functors have manager-specific signatures; use `/add-functor`.
- Update public API docs with `/update-api-docs`; the read-only coverage gate is
  `python docs/scripts/check_api_docs.py`. Sphinx sources live in `docs/source/`;
  build with `pip install -r docs/requirements.txt`, then `make -C docs html`.
  For locale errors use `LC_ALL=C.UTF-8` and `LANG=C.UTF-8`.
- When routed behavior changes, review affected context in the same change:
  `python .agents/skills/project-dev-context/scripts/context.py affected --base origin/main`.
  Follow the context skill for updating and checking it.

## Contribution routes

Use `.github/ISSUE_TEMPLATE/bug.md` for `[Bug Report]` issues with a minimal
reproduction and commit/OS/GPU/CUDA/driver details; check duplicates first.
Use `.github/ISSUE_TEMPLATE/proposal.md` for `[Proposal]` motivation and scope.
Keep branches and PRs focused on one logical change and use the PR template.

| Skill | Use |
|---|---|
| `/project-dev-context` | Navigate, read, refresh or add project context |
| `/add-task-env` | Select environment-only, handwritten expert, Task Program or RL task route |
| `/add-task-program` | Author, integrate, deploy, validate or repair a program |
| `/add-embodiment-component` | Reuse a robot, sensors and optional skill profile |
| `/add-semantic-call` | Expose an Atomic Skill as a Semantic Call |
| `/add-atomic-action` | Add typed action planning/execution |
| `/add-robot` | Add robot config and kinematic-chain wiring |
| `/add-solver` | Add IK/FK solver, docs, tests and benchmark |
| `/add-functor` | Add observation, reward, event, action, dataset or randomization operation |
| `/add-test` | Add focused tests with project conventions |
| `/benchmark` | Write a benchmark using project patterns |
| `/update-api-docs` | Synchronize public API documentation |
| `/pre-commit-check` | Validate readiness proportionally |
| `/review-pr` | Review a change for evidence-backed regressions |
| `/pr` | Draft or create a single or stacked PR |
| `/release` | Prepare or publish release artifacts |
