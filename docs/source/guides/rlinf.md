# Using EmbodiChain with RLinf

RLinf is an optional external training backend for EmbodiChain environments.
This page explains the boundary between the two stacks and the EmbodiChain
contracts that an RLinf worker consumes. It does not repeat RLinf's
installation, launcher, or task recipe; use the [official RLinf
EmbodiChain guide](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/embodichain.html)
for that workflow.

## Choose a training stack

The two stacks serve different experiment lifecycles. Use this as a workflow
guideline when deciding where to put the policy and training runtime:

| Experiment profile | Recommended stack | Typical lifecycle |
| --- | --- | --- |
| Locomotion or a simple visual policy | EmbodiChain RL (`embodichain train-rl`) | Configure the task, policy, algorithm, and trainer in EmbodiChain and usually train from scratch |
| A pretrained VLA or WAM policy | RLinf | Start from pretrained weights, run SFT when needed, then continue with RL using RLinf's actor, rollout, and distributed training stack |

This is a recommended workflow boundary rather than a hard capability
restriction. EmbodiChain owns the Gym task, observations, actions, rewards, and
simulation configuration. RLinf is useful when the experiment needs a
pretrained model lifecycle or RLinf's distributed training runtime.

For EmbodiChain's native RL workflow, see the
{doc}`RL tutorial </tutorial/rl>` and the
{doc}`RL overview </overview/rl/index>`. For the external training workflow,
see the [official RLinf EmbodiChain guide](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/embodichain.html).

## EmbodiChain integration contract

An RLinf worker uses the same installed task package and runnable Gym
deployment as the EmbodiChain CLI:

- The RLinf environment selects `env_type: embodichain` and points
  `gym_config_path` at a runnable deployment with a non-empty `id`.
- The task package must be installed in the Python environment used by RLinf
  workers and expose the `embodichain.tasks` entry point.
- Package initialization hooks must be safe to run more than once because
  workers construct environments independently.
- The deployment's observation and action spaces remain the source of truth;
  the RLinf policy dimensions and mapping must match them.

See the [task package registration contract](https://github.com/DexForce/EmbodiChain/blob/main/embodichain_tasks/README.md#how-registration-works)
for discovery and initialization details. See the
{doc}`configuration guide </guides/configuration>` for inline and componentized
Gym deployment paths.

The official RLinf recipe may move at a different pace from EmbodiChain. Older
versions of that recipe use the retired
`configs/agents/rl/basic/<task>/gym_config.json` layout. Current EmbodiChain
deployments live under
`embodichain_tasks/configs/tasks/<category>/<task>/env.json` or a corresponding
task deployment selected by the project configuration conventions.

## Current compatibility boundary

The adapter is intentionally conservative. Verify these boundaries before
scaling a training run:

| Capability | Current status |
| --- | --- |
| Inline `env.json`/`env.yaml` deployment | Supported first |
| Python `@register_env` task | Supported when its package is installed in the RLinf runtime |
| External manager or asset resolver | Supported through an `embodichain.init` hook |
| Componentized deployment | Experimental; verify relative component paths |
| Configuration-defined Task Program | Experimental; verify the adapter version |
| Default physics backend | Supported first |
| Newton backend and custom simulation settings | Verify that the adapter preserves EmbodiChain's `sim_cfg` |
| Low-dimensional robot state | Supported through explicit adapter state mapping |
| Object, camera, language, or nested observations | Requires an explicit mapping |
| Flat continuous `Box` actions | Supported first |
| Dict, semantic, or VLA/chunked actions | Requires an action converter |

Before a long run, verify a worker-local `reset()` and `step()` smoke test,
the policy observation and action dimensions, the selected physics backend,
and any reusable component paths. Task discoverability alone does not prove
that observation semantics, action semantics, or simulation settings are ready
for RLinf.

## Ownership and support

| Area | Owner |
| --- | --- |
| Task registration, Gym deployment, robot and scene configuration, managers, and physics settings | EmbodiChain |
| Actor, rollout, trainer, placement, checkpoints, and distributed execution | RLinf |

Report task registration, configuration composition, robot or scene, manager,
and physics issues to the EmbodiChain integration. Report actor, rollout,
placement, checkpoint, and trainer issues to RLinf.
