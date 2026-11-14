# Task Program progress during policy execution

Run from the repository with its simulator dependencies and assets installed:

```bash
python -m examples.learning.task_program_progress --seed 7 --steps 16
```

The seeded uniform policy supplies the complete CobotMagic ActionManager vector.
The deployment `task.policy.cobotmagic.yaml` loads the measured program and rules
without creating an expert adapter or executing its declarative pick/place calls.
The output JSON retains seed, episode context, program/evaluation identities,
per-step rewards, progress and Gym flags. Terminal outcomes include configuration
and episode identities, and separate execution, physical, acceptance, termination
and persistence namespaces. The short deadline produces a reproducible truncation;
this untrained policy is a lifecycle smoke example, with no task-success claim.

The policy environment variant owns its action and reward managers. The existing
expert deployment remains available. Both consumers can select
`task_program/program.measured.yaml` and `evaluation.measured.yaml`: its named
conditions serve expert condition validators and policy progress without copied
thresholds. The lift condition here means arena height at least 0.95 m, an explicit
scene-specific clearance, rather than an inferred expert segment-start baseline.

Configuration requires `task_program.execution: policy`, explicit `action_owners`
for every skill-profile resource, and `rl_task.success` containing
`program_stage_condition` with `state: achieved|current`. Every endpoint's actuated
joints must be covered by ActionManager. The execution-policy component supplies
static semantic preset metadata only. Expert bridge/demo and ControllerAction
entry points reject policy-owned execution.

Use `program_stage_reward` in ordinary RewardCfg terms:

```yaml
lift_bonus:
  func: program_stage_reward
  mode: add
  weight: 2.0
  params:
    stage: lift_block
    mode: first_achievement
    scope: occurrence
```

Reward modes are `satisfied` (eligible current truth), `rising` (false-to-true),
`first_achievement` (sticky ordered milestone) and `metric` (exact scalar metric
key plus linear `scale`). Weight or scale may be negative for a penalty. A first
bonus must declare `scope: occurrence` or `episode`; episode scope pays once per
authored stage name across repeated occurrences. Reads in the same transition
return the same reward. Missing measurements return zero, and ambiguous or
nonscalar measured metrics fail. Unknown metric keys have no measured contribution.

Supported declarations are explicit sequential segments, including finite repeats,
with named instantaneous condition validators, instructions and contracts.
Condition prerequisites use the same measured frame. Occurrence achievements are
ordered; the next can achieve only after a later qualified transition. Current
truth may revoke while historical achievement remains true. Final and process
requirements remain mandatory even after all achievements are reached. Missing or
invalid evidence fails closed through TerminationManager. Temporal graphs, expert
segment-start baselines, parallel/implicit segments, post-policies and required
symbolic/call-effect outcomes are rejected by preflight.
