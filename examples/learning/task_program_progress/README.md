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


## Composed skill rewards

Read the [system architecture and reward contract](../../../docs/architecture/task-program-reward-composition.md)
for the component diagram, default skill formulas, configuration resolution,
runtime ordering and reuse with future learned skills.

The existing RewardManager remains the execution and weighting entry point.
`joint_task.yaml` is a standalone MoveJoints objective; `program_task.yaml` uses
`program_rewards.yaml` to share PickUp/Place templates with a policy-owned Task
Program. The same profiles are independent of planning, Real2Sim and any future
learned skill executor. This change supplies reward interfaces, not a learned
skill execution backend.

```bash
python -m embodichain train-rl --config examples/learning/task_program_progress/joint_ppo.yaml
python .agents/skills/add-task-program/scripts/inspect_deployment.py examples/learning/task_program_progress/program_task.yaml
```

The training configuration intentionally performs three PPO updates for pipeline
qualification. Increase iterations for an actual training experiment; successful
updates are not evidence of task convergence. The CLI validates reward discount
against algorithm gamma before constructing simulation. Saved configurations pin
the installed reward-profile revisions and resolved objective thresholds.

A standalone objective declares the exact robot/control-part binding, a goal,
and acceptance thresholds. Its measured success is independent of shaping weight.
Manipulation objectives additionally bind target/support actors and a calibrated
gripper with two opposing pads and its full collision-link selection. Contact evidence and collision
geometry are required; unsupported backends fail explicitly. Geometry is bound
for the environment lifetime; rebuild after changing assets or topology.
Measured physical goals reject `replace_assets_from_group` before reset events
run. Program hand links come from the selected resource's calibrated `grasp`
endpoint; omitted lists are derived and explicit lists must match the full hand.

Program objectives select an authored `stage`, or a global expanded `call_index`.
A multi-call segment requires explicit objectives for every call before its
internal progress can be expanded. Sequential repeats retain separate physical
windows. Object, control resource and semantic placement targets must agree with
the compiled calls. Parallel programs and expert post-policies remain outside
policy progress support. An omitted skills list selects generic program progress.

Held and stable windows sample every physics substep, so a brief contact at the
end of a control step does not earn the full control interval. The first valid
sample establishes a boundary; interrupted evidence restarts the window. Missing
samples and pose changes after sampling invalidate the affected evidence.

The composition generates additive potential, outcome and milestone RewardCfg
terms. Existing explicit success/first-achievement terms suppress their matching
automatic bonuses. `success`, `failure`, `milestone_budget`, `time_cost` and overall
shaping `weight` may be specified in `reward_composition`; defaults are 10, -10,
1, 0 and 1. Disabling shaping weight does not disable the physical goal. Every
skill target and sampling/acceptance parameter is explicit in the effective config.
Equivalent stage names, occurrence IDs and `call:N` selectors suppress the same
automatic milestone. Standalone `rl_task.skill` requires `reward_composition`
with an explicit discount.
Episode-scoped explicit bonuses suppress only the first occurrence of their
authored name, leaving later automatic milestones available.

Program `scene_relative` call targets freeze their rigid reference at activation
per row/occurrence; repeats and reset rows rebind without retargeting peers.
Place orientation preservation applies to both `at` and `on` destinations.
Articulation/link references are not supported by measured skill targets.
Dynamic replay seeds reward baselines from the final restored observation.
Policy-only inference does not require an algorithm block.

Dense formulas accept scalar role scores or error channels with finite `target`
and finite positive `scale`; vector errors reduce by Euclidean norm. Named templates can
be registered by trusted code with `register_skill_reward_profile` and selected
with `profile`. Extensions can subclass `SkillRewardProfile.potential`; templates
receive measured tensors and never step simulation. Unknown names and mismatched
saved revisions reject. Custom model entropy/KL losses remain algorithm-owned.

Use `aggregate_reward_sequence` for VA/VLA action chunks: supply per-control-step
rewards, executed masks and termination/truncation flags. It returns discounted
return, continuation discount and actual prefix length. Keep the original
pre-reset successor observation for bootstrap and never concatenate episodes.
Image/language preprocessing and action/token log-probabilities belong to the
model's collector. No specific VA/VLA training backend is included here.

## Native acceptance and videos

These fixed fixtures qualify physical evidence; they are not learned-policy
success-rate experiments. `supported` must be accepted after a complete stable
window; `floating` and `empty_grip` must not be accepted.

```bash
python -m examples.learning.task_program_progress.qualify_rewards --case floating --steps 15
python -m examples.learning.task_program_progress.qualify_rewards --case empty_grip --steps 15
xvfb-run -a python -m examples.learning.task_program_progress.qualify_rewards --case supported --video
xvfb-run -a python -m examples.learning.task_program_progress.qualify_rewards --run outputs/YOUR_RUN --num-envs 1 --video
python -m scripts.benchmark.learning.rewards
```

Use a working native display instead of xvfb-run when available. The commands
write JSON traces and diagnostic MP4s under `outputs/reward_qualification`.
Checkpoint video loads the actual saved policy with deterministic inference;
fixture videos explicitly identify zero-action physical scenes. The performance
benchmark excludes contact acquisition, rendering and policy inference, and
reports correctness agreement separately from physical task success.


A complete planning-expert grasp/place trace can qualify the same measured
objectives independently of learned-policy performance:

```bash
xvfb-run -a python -m examples.learning.task_program_progress.qualify_rewards --expert --steps 1100 --video
```

This calibrated single-episode fixture uses CPU physics to avoid the native
Default backend's direct-GPU controller-fetch issue seen during qualification.
The JSON explicitly labels the planning source and separately reports the
expert executor result, measured grasp acceptance and measured placement
acceptance. A positive support witness may persist through sleeping-body reports
only while both physical poses remain unchanged; motion, invalid acquisition or
reset invalidates it. No reward bonus or symbolic attachment establishes support.
A measured objective may select a trusted `reward_profile` while retaining the
same independent physical acceptance thresholds.


## Task Program reward acceptance without training

Use the focused acceptance harness when only Task Program reward correctness is
needed. It constructs no policy model or optimizer. Planning commands are
collected from the calibrated expert, then supplied through ordinary absolute
JointPositionAction terms to the configuration-owned Task Program reward host.
The host's `execution: policy` declaration identifies external action ownership;
it does not require policy training.

```bash
python -m examples.learning.task_program_progress.accept_program_rewards --phase collect
xvfb-run -a python -m examples.learning.task_program_progress.accept_program_rewards --phase replay --case success --video
python -m examples.learning.task_program_progress.accept_program_rewards --phase replay --case hold
python -m examples.learning.task_program_progress.accept_program_rewards --phase replay --case truncated
python -m examples.learning.task_program_progress.accept_program_rewards --phase replay --case failure
python -m examples.learning.task_program_progress.accept_program_rewards --phase report
```

Every native env.step return and each reward term is checked against independent
scalar reward-composition arithmetic. The checks cover milestone weights,
once-only outcome events, discounted shaping, true termination versus timeout,
repeated reads, two asynchronous environments and the next reward after a partial
reset. Local skill potentials come from canonical measured observations; the
arithmetic audit is not an independent geometry or skill-potential-formula oracle.

Reports under `outputs/task_program_reward_acceptance` bind the exact controller
JSON hash, planning configuration hashes, replay evaluation declaration and
effective reward/objective contracts. Inputs changing after collection reject
replay. The annotated MP4 displays rewards from that same rollout and freezes
the last frame for one second to make terminal rewards readable. Video hashes,
captured step indices and the reward-trace hash bind footage to its replay;
rerunning without `--video` produces no preview, even if old footage is present.
The focused
production regression is in `tests/lab/task_program/test_policy_progress.py`.


A recorded native reward acceptance result and annotated preview are available
in [validation/README.md](validation/README.md).
