(task-program-deployment)=

# Configure and verify a deployment

A runnable Task Program deployment selects the program, semantic integration,
physical environment, embodiment, and execution policy for one Gym ID. Follow
{doc}`quickstart` to replace the program in an existing deployment. Use
{doc}`/tutorial/task_program` for a worked integration from the physical scene
through execution.

## Select a task and environment

From an installed SDK, use the task catalog to find deployment keys and their
runnable configuration paths.

```bash
embodichain list-task --category manipulation
embodichain show-task embodichain.extensions.tasks:repeated_pick_place
```

For example, `franka` selects `task.franka.yaml` with Gym ID
`TaskProgramRepeatedPickPlace-Franka-v1`. Use the path of that runnable file
with `run-task --gym_config`. `--task-program` replaces the program with a
local file compatible with the selected `program_id`, scene and resources.
The deployment supplies the trusted integration selection.

Each reusable physical environment declares its own `physics` backend.
With an `environment.default/newton` mapping, `--physics newton` selects the
Newton environment file. With one `environment.component`, `--physics`
confirms the backend declared by that file. The Gym ID comes from the runnable
deployment in both cases. See {doc}`/guides/run_task` for the launcher options.

## Find the configuration to edit

| Change | Configuration owner |
|---|---|
| Reorder calls, move a target, or change subgoal language | `task_program.program` selects the file containing the flow, named targets, and instructions. |
| Change a scene object's shape, pose, dynamics, or physics backend | The selected `environment` file owns physical simulation entities and ordinary Gym settings. |
| Expose a physical object to a program or select a grasp affordance, call option, or effect monitor | `task_program.integration` selects the file owning the nested `scene_binding` and task-specific semantic settings. |
| Change the robot, sensors, or reusable resource definitions | `embodiment.component` supplies the robot, sensors, and skill profile. |
| Tune motion, tracking, recovery, or effect assurance | `task_program.execution_policy` selects the policy component. |
| Define a reusable measured condition or episode acceptance rule | The optional `evaluation.component` supplies named rules, final acceptance, and process constraints. |
| Choose a compatible combination or give it a new Gym ID | Edit the runnable `task.<embodiment>.yaml`. |

Relative component paths in the runnable YAML resolve from that file's
directory. When moving a deployment to a new directory, update its selectors
to point to the corresponding components. Reusing one program across robots
requires compatible scene identities, embodiment contracts, and call resources.
Use {doc}`scene_registry` to map `entity_id` to `simulation_uid` and
{doc}`robot_profiles` to bind resources to robot endpoints.

## Choose expert execution or policy progress

`task_program.execution` defaults to `expert`. The Task Program runtime lowers
Semantic Calls to Atomic Skills and the Gym demo executor steps the generated
actions. Use this mode to run and record expert trajectories.

With `task_program.execution: policy`, the policy supplies the ActionManager
commands. The Task Program defines ordered stages evaluated from observations
after each step. Set `action_owners` to `policy` for each skill-profile resource
and declare a `program_stage_condition` in `rl_task.success`. Named
instantaneous conditions can be shared by expert validators and policy
progress. Policy programs use explicit segments in a sequential flow with
instructions and contracts. A segment authored with `segment:` is explicit
for this purpose.

The `StackBlocksTwoPolicy-v1` example pairs a measured program and evaluation
component with policy-owned resources. Run its progress example from a source
checkout with the required simulator and assets installed.

```bash
python -m examples.learning.task_program_progress --seed 7 --steps 16
```

This example reports ordered stage achievements, rewards, and Gym flags from
a seeded, untrained policy. See [define segments and their outcomes](authoring.md#define-segments-and-their-outcomes)
for the program's lift stage and the {download}`example README <../../../../examples/learning/task_program_progress/README.md>`
for supported stage and reward configurations.

## Inspect the resolved deployment

Use a Python environment with the SDK installed from your source checkout.
Run the inspector from the checkout root before changing controller or
acceptance settings.

```bash
python .agents/skills/add-task-program/scripts/inspect_deployment.py \
  embodichain/extensions/tasks/configs/tasks/manipulation/repeated_pick_place/task.franka.yaml \
  --explain --json
```

The inspector resolves the physical components and semantic bindings, decodes
the program, and runs provider-free catalog preflight.
Check `physics` and `resolved_configuration` for the selected backend and
authored owners of effective parameters. `calls` lists expanded call indices,
segment IDs, authored paths, and resource bindings. `normalized_program`
shows the same node structure for either YAML form. Add `--strict-contracts`
when you want inspection to require an instruction and contract for each
segment. Use the default inspection checks for the Quickstart deployment.

## Verify the run

Start with one environment and a fixed seed. Check each result before relying
on it for data collection or policy evaluation.

| Check | Evidence to inspect |
|---|---|
| Configuration and planned sequence | Static inspection passes, and the resolved backend, calls, targets, and resources match the task. |
| Expert command execution | Inspect the first failing segment or call in `--debug-mode` output when grounding, planning, or execution fails. |
| Effect assurance | A `verified` policy advances call effects from measured evidence. A `projected` policy advances expected symbolic effects after command completion. |
| Task acceptance | Inspect segment `metadata.post_policies` and `metadata.validation` for expert runs. For policy runs, inspect ordered stage achievements and the final and process requirements. |
| Dataset persistence | Check the recorder's saved path and read back a committed episode with its segment metadata. |

Physical qualification requires live observations under the conditions you
intend to use. Follow {doc}`quickstart` for a worked one-episode check and
{doc}`troubleshooting` to locate the first failing stage.
