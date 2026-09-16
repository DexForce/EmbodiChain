# Affordance sampling

Affordance sampling lets batched simulation actions explore multiple legal
geometric poses without moving sampling policy into the task orchestrator.
The first row in each branch group keeps the nominal or lowest-cost pose;
the remaining rows choose geometrically distinct alternatives when available.

## Responsibility boundary

The implementation separates four concerns:

| Concern | Owner |
|---|---|
| Legal grasp, contact, interaction, and assembly pose space | `Affordance` |
| Reproducible branch identity | `AffordanceSamplingContext` |
| IK, reachability, and complete trajectory feasibility | Atomic Action |
| Parallel simulation rows and replay | Direct simulation host |

`Affordance` objects do not retain sampling state. A caller optionally adds an
immutable `AffordanceSamplingContext` to `PlanningContext`; actions pass it to
the Affordance for each named sampling operation.

```python
from embodichain.lab.sim.atomic_actions import AffordanceSamplingContext

sampling = AffordanceSamplingContext(
    count=4,
    seed=7,
    episode_id=0,
    attempt_id=0,
)
context = engine.initial_context(
    control_dt=control_dt,
    affordance_sampling=sampling,
)
```

The public result is `AffordanceSample`. It retains a row-local success mask,
the selected poses, and metadata such as candidate IDs, valid-candidate counts,
duplicate reuse, and sampling identity. Actions intersect its success mask with
their own feasibility result and publish the metadata under
`plan.diagnostics.metadata["affordance_sample"]`.

## Supported geometric freedom

The current simulation implementation covers:

- antipodal candidates for PickUp, AxisAlign, Slide, and OpenDoor;
- interaction points;
- assembly symmetry for AssembleGoal placement;
- Press contact-frame roll; and
- explicitly declared Twist grasp-roll symmetry.

Explicit caller-provided grasp poses remain exact. Slide travel distance, Twist
angle, and OpenDoor opening fraction are action or goal parameters and are not
sampled by the Affordance layer.

## Parallel PickUp tutorial

The PickUp tutorial maps one Affordance branch to one simulation environment:

```bash
python scripts/tutorials/atomic_action/pickup.py \
  --affordance_branches 4 \
  --sampling_seed 7 \
  --sampling_attempt 0 \
  --auto_play
```

The tutorial logs the selected candidate and reuse status for every row before
replaying the batched trajectory. `--num_envs`, when supplied, must match
`--affordance_branches`.

This is intentionally a direct simulation integration. Task Program lowering,
Gym episode lifecycle, retry policy, commit behavior, and dataset recording are
deferred until those hosts have an explicit expansion policy.

```{seealso}
{doc}`Atomic actions <index>` and
{doc}`Built-in actions <builtin_actions>`.
```
