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

- antipodal candidates for PickUp, AxisAlign, HandOver, Slide, and OpenDoor;
- interaction points;
- assembly symmetry for AssembleGoal placement;
- Press contact-frame roll; and
- explicitly declared Twist grasp-roll symmetry.

Explicit caller-provided grasp poses remain exact. Slide travel distance, Twist
angle, and OpenDoor opening fraction are action or goal parameters and are not
sampled by the Affordance layer.

## Parallel simulation tutorials

The following tutorials map one Affordance branch to one simulation environment.
They share argument validation, sampling-context construction, and per-row
diagnostic logging in `scripts/tutorials/atomic_action/tutorial_utils.py`.

| Entry point under `scripts/tutorials/atomic_action/` | Geometric augmentation | Sampling metadata | Preserved task inputs |
|---|---|---|---|
| `pickup.py` | Select antipodal candidates returned by `AntipodalGraspPoseGenerator.get_valid_grasp_poses()`, after pickup IK and symmetric-roll screening | `grasp`: candidate ID, candidate counts, reuse | Approach direction, pre-grasp distance, lift height |
| `axis_align.py` | Select generator candidates within the action's preferred alignment-grasp set, then apply the existing symmetric/upright pose adjustments | `grasp`: candidate ID, candidate counts, reuse | Object internal axis and requested target axis |
| `hand_over.py` | Independently select generator candidates for pickup and receiving grasps; the first hand uses its near projected object end and the receiver uses the opposite end at the predicted middle pose | `pickup_grasp` and `receive_grasp`: candidate ID, candidate counts, reuse, executing arm | Nearest-arm assignment, opposite-end constraint, lift height, final target |
| `open_door.py` | Select generator candidates on the door handle and replan the hinge-following motion | `grasp`: candidate ID, candidate counts, reuse | Hinge geometry and requested opening fraction |
| `press.py` | Sample a contact-frame roll in `[-pi, pi]` radians around the press pose's local z-axis | `roll`: per-row contact roll in radians | Contact position, local z-axis, approach distance, press distance |
| `slide.py` | Select generator candidates on the handle for both pull and push; push plans from the handle pose observed after pull replay | `grasp`: candidate ID, candidate counts, reuse | Pull/push direction and translation distance |
| `twist.py` | Explicitly declare `grasp_roll_range=(-pi, pi)` for parallel branches and sample around the grasp pose's local z-axis | `roll`: per-row grasp roll in radians | Grasp position, local z-axis, twist-axis origin, commanded twist angle |

The antipodal tutorials use `get_grasp_candidates()` to retain all returned
poses, costs, and validity before `sample_candidates()` selects a pose for each
row. HandOver uses separate named streams for pickup and receiving and reports
metadata from the arm assignment selected for each row. Candidate selection
does not replace each action's subsequent trajectory-feasibility checks.

For Press and Twist, a sampled angle changes only the local x/y directions:
`R_sample = R_nominal @ Rz(angle)`. The contact/grasp position and local z-axis
remain fixed. Branch zero retains the nominal angle; the remaining branches
use stratified random angles over the declared range. Existing action-level
symmetric-pose selection can subsequently add a 180-degree roll. The Twist
range describes the tutorial's allowed grasp variation; other callers must
explicitly declare a range appropriate to their own target geometry.

Every listed entry point accepts the same branch controls:

| Argument | Default | Meaning |
|---|---|---|
| `--affordance_branches` | `--num_envs` (normally 1) | Positive number of logical branches and simulation rows; it is not the number of raw grasp candidates |
| `--sampling_seed` | `0` | Non-negative base seed for Affordance selection streams |
| `--sampling_attempt` | `0` | Non-negative resampling-attempt identity; changing it changes the stream, but it neither runs retries nor increments automatically |

For example:

```bash
python scripts/tutorials/atomic_action/pickup.py \
  --affordance_branches 4 \
  --sampling_seed 7 \
  --sampling_attempt 0 \
  --headless \
  --auto_play
```

Replace `pickup.py` with any entry point in the table to run the corresponding
four-branch example. `press.py --rigid_object` and `twist.py --rigid_object`
also support the same roll augmentation. Press, Slide, and OpenDoor use
`strategy="ik_interp"` to preserve their required Cartesian contact/path
samples; their exact-path segments do not use the `--planner` backend.

Each tutorial logs the selected candidate and reuse status, or the sampled roll,
for every row before replay. HandOver also logs the arm for each grasp stage.
The tutorials replay only when every row has a successful plan. Planning
success is not a physical task-success measurement.

The shared parser accepts `--num_envs=1` as the default sentinel; otherwise
`--num_envs` must match `--affordance_branches`. With one branch, the context is
`None` and nominal/lowest-cost behavior is retained. Increase the branch count
for more parallel trajectories, or change `--sampling_attempt` for another
batch. Neither operation guarantees previously unseen trajectories.

## Selection and reproducibility

For discrete candidates, selection ranks valid poses by cost and considers at
most the first `max(32, 4 * count)` entries. It retains the best pose as branch
zero, then favors geometric diversity using seeded random weights. When a
row's distinct candidate set is exhausted, selection cycles through it and
reports `reused=True`. Candidate IDs are local to each row's generator output;
equal IDs across rows need not identify equal poses. `reused=False` is not a
cross-row or cross-batch uniqueness guarantee.

The private selection generator is derived from
`(seed, episode_id, attempt_id, key, group)`. These tutorials use `episode_id=0`.
The same identity reproduces selection from the same candidate pool. However,
the antipodal generator's approach-direction perturbations currently use
PyTorch's global random stream and generate candidates separately for each row.
`--sampling_seed` does not seed that upstream stage or the simulator. Reproducing
an entire antipodal run also requires controlling upstream randomness, geometry,
initial state, and cached annotations. Press/Twist roll sampling itself uses the
private Affordance stream.

This is intentionally a direct simulation integration. Task Program lowering,
Gym episode lifecycle, retry policy, commit behavior, and dataset recording are
deferred until those hosts have an explicit expansion policy.

```{seealso}
{doc}`Atomic actions <index>` and
{doc}`Built-in actions <builtin_actions>`.
```
