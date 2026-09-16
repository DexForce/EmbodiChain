# Atomic-Action Benchmark Standard

This document defines how atomic-skill benchmarks in
`scripts/benchmark/atomic_action/` measure success and cost, and why each
threshold has the value it has. It is written so that another engineer can
reproduce the numbers and extend the suite to a new skill without inventing new
vocabulary.

The vocabulary is inherited from
`scripts/benchmark/motion_generation/BENCHMARK_DESIGN.md` (sections 4.3 and 7).
Nothing here renames or replaces those terms.

---

## 1. The four-level success ladder

Every case records all four stages, not just the final verdict. A case that
fails at stage *n* still reports stages `1..n-1` as passed, so a report
distinguishes "the planner could not solve this" from "the plan was fine but
the robot did not achieve the task".

| Stage | Meaning | How it is decided |
|---|---|---|
| `planning_success` | The engine compiled the invocation. | `compiled.plan_success.all()` is true and `AtomicActionEngine.compile` did not raise. |
| `motion_valid` | The planned trajectory is usable on its own terms. | Non-empty, all values finite, and every waypoint inside the robot's joint position limits (tolerance `1e-3` rad). Implemented in `common.check_motion_valid`. |
| `execution_success` | The physical replay completed and the arm actually followed the plan. | The replay ran without raising **and** the maximum per-joint tracking error stayed at or below `0.10` rad. |
| `task_success` | The skill's semantic goal was achieved. | Per-skill rule; see section 4. |

The ladder is implemented once, in `common.SuccessLadder`. Calling
`ladder.fail(stage, reason)` records the **first** failing stage and its reason;
later failures do not overwrite it.

### Why `execution_success` includes a tracking check

`Robot.set_qpos` writes **drive targets**, not teleported state. If the replay
advances to the next waypoint before the position controller has converged, the
arm silently lags the plan, and any scene measurement then describes the
controller rather than the skill. Gating `execution_success` on tracking error
means a case is never credited or blamed for a motion the robot did not
actually perform. `controller_tracking_failure` is the reason code used, taken
from the design document's taxonomy.

---

## 2. Failure taxonomy

`common.FAILURE_TAXONOMY` is the design document's section-7 list, used
verbatim. `SuccessLadder.fail` rejects any reason outside it, so reports cannot
drift into ad-hoc strings.

```
invalid_case, unsupported_capability, checkpoint_load_failure,
planner_exception, planner_reported_failure, timeout,
non_finite_trajectory, waypoint_miss, joint_limit_violation,
dynamic_limit_violation, self_collision, environment_collision,
controller_tracking_failure, object_not_grasped, object_dropped,
release_failure, task_goal_miss
```

Reasons actually emitted by this suite today:

| Reason | Emitted when |
|---|---|
| `planner_exception` | `compile` raised. |
| `planner_reported_failure` | `compile` returned `plan_success == False`. |
| `non_finite_trajectory` | Trajectory empty or containing non-finite values. |
| `joint_limit_violation` | A waypoint lies outside the robot's joint limits. |
| `controller_tracking_failure` | Replay raised, or tracking error exceeded `0.10` rad. |
| `object_dropped` | The manipulated object fell below the drop threshold. |
| `task_goal_miss` | Replay was faithful but the skill's goal was not met. |

Every report's "Success & Other Metrics" table carries `failure_stage` and
`failure_reason` columns. No fourth table is added, per the design contract.

---

## 3. Measuring the physical replay

### 3.1 Where the target is measured

Contact skills must be scored **while the robot still controls the target**.
Each skill's plan is a sequence of named segments; the benchmark scores the
last waypoint of the segment that actually actuates the target.

| Skill | Segments emitted | Scored segment |
|---|---|---|
| `open_door` | `approach, reach, close, open, release, retract` | `open` |
| `press` | `close, approach, contact, press, retract` | `press` |
| `slide` | `approach, reach, close, pull, open, [return]` | `pull` |
| `twist` | `approach, reach, close, twist, open, retract` | `twist` |
| `axis_align` | `approach, close, manipulate` | `manipulate` |
| `pour` | `pour` (single segment, second invocation) | `pour` (peak, see 4.6) |
| `hand_over` | unified pick-and-transfer plan | whole trajectory + settle (see 4.7) |

This matters concretely. Measuring OpenDoor after the full trajectory reads the
hinge **after** the hand has opened and the arm has retracted. The tutorial
microwave's hinge is undriven (`drive_type="none"`), so it then swings freely:
in a measured 60-degree case the hinge reached 0.979 rad at the end of the
`open` segment but settled at 0.667 rad after release, which would have been
scored as a 0.38 rad miss for a door the skill had opened correctly.

Each report therefore carries three readings:

- the value at the scored waypoint (the skill's achievement),
- `*_peak_signed_*` — the extreme value reached at any point, sign preserved,
- `*_settled_*` — the resting value after release and retract, **diagnostic
  only, never a success gate**.

"Peak signed displacement" is the design document's own term for contact
skills, where "final rebound is allowed".

### 3.2 Replay rate, and why it is 16 physics steps per waypoint

The planners emit roughly one waypoint per physics step, but the simulated
position drive cannot converge that fast. Sweeping the replay rate on OpenDoor's
60-degree case gave:

| Steps per waypoint | Max arm tracking error (rad) | Hinge error (rad) |
|---|---|---|
| 1 | 1.3844 | 0.2273 |
| 2 | 0.9242 | 0.1254 |
| 4 | 0.4390 | 0.0689 |
| 8 | 0.1599 | 0.0506 |
| 16 | 0.0532 | 0.0306 |
| 32 | 0.0230 | 0.0245 |

The measured target error is dominated by tracking error, and both converge
together. The same sweep on Slide's 180 mm pull went from 0.0525 m error at 1
step to 0.0035 m at 8 steps.

`DEFAULT_REPLAY_STEPS_PER_WAYPOINT = 16` is chosen because the target-joint
measurement has converged to within 0.006 rad of the 32-step value while
costing half the simulation time. `REPLAY_TRACKING_TOLERANCE_RAD = 0.10` sits
comfortably above the 0.053 rad worst case observed at that rate, so it does not
fire on healthy runs but does catch a genuinely unfollowed trajectory.

**Replaying at the planner's own `dt` is not correct here.** It was tried: it
produces tracking errors above 1 rad and makes every contact skill look broken.
The planned timing describes what a well-tuned controller would do, not what
this simulation's drive achieves.

> Known inconsistency: the five originally shipped benchmarks (`move_joints`,
> `move_end_effector`, `pick_up`, `place`, `move_held_object`) still replay at a
> fixed 4 steps per waypoint through
> `common.replay_trajectory_for_physical_validation`, and do not perform a
> tracking check. They pass at that rate, and their behaviour was deliberately
> left unchanged. Aligning them to 16 steps plus the tracking gate is a
> recommended follow-up, not something this suite changed silently.

### 3.3 The shared replay primitive

`common.replay_and_track_scalar` is the single physical-replay implementation.
It takes a `read_value(waypoint_index)` callback, so a skill can track an
articulation joint, an object's axis angle, or a relative rotation, and returns
a `JointDisplacementTrace` with the initial, scored, peak-signed, and settled
values plus `max_tracking_error_rad`.

`common.replay_and_track_joint` is a thin wrapper for the common
articulation-joint case. `common.run_articulated_contact_case` wraps the whole
plan / validate / replay / score ladder for contact skills.

---

## 4. Per-skill `task_success` rules and thresholds

Every threshold below is stated with the measurement that justifies it. Where a
threshold is still open, section 6 says so explicitly.

### 4.1 `open_door`

- **Rule**: the hinge angle at the end of the `open` segment is within
  `0.15` rad of the commanded absolute opening angle.
- **Measured**: 0.034 / 0.032 / 0.041 rad error for 30 / 60 / 90-degree targets.
- **Basis**: the tolerance is roughly 4x the observed error, leaving room for the
  error growth seen at larger angles while still rejecting a door that did not
  open. It is *not* tuned to the measurements — 0.15 rad predates them and the
  measurements simply fit inside it.

### 4.2 `press`

- **Rule**: the peak signed button displacement reaches at least 80 % of the
  button joint's own usable stroke.
- **Measured**: the `start_button_press` joint has a 6.00 mm stroke, so the gate
  is 4.80 mm; all commanded distances reached the full 6.00 mm.
- **Basis**: expressing the gate as a fraction of the asset's stroke keeps it
  portable to another button instead of hard-coding a millimetre figure.
- **Caveat**: on this asset the gate does not discriminate. Commanding 3 mm
  still drove the button to 5.66 mm, because the button joint has near-zero
  stiffness (`1e-3`) and bottoms out on contact. Press's commanded
  `press_distance` does not control achieved depth here. See section 6.

### 4.3 `slide`

- **Rule**: the drawer travel at the end of the `pull` segment matches the
  commanded translation within `0.03` m.
- **Measured**: 0.0002 / 0.0006 / 0.0010 m error for 100 / 180 / 250 mm pulls.
- **Basis**: the measured error is three orders of magnitude inside the
  tolerance, so 0.03 m is very loose. It is retained because it was set before
  the replay rate was calibrated; section 6 proposes tightening it.

### 4.4 `twist`

- **Rule**: the knob rotation at the end of the `twist` segment matches the
  commanded angle magnitude within `0.15` rad.
- **Measured**: **fails**. Commanded 0.52 / 0.79 / 1.57 rad produced 2.38 / 2.53 /
  3.14 rad of knob rotation.
- **This is not a planning defect.** The planned end-effector rotation about the
  resolved knob axis was measured at exactly 0.7854 rad for a 0.7854 rad
  command, with the correct sign. Arm tracking error at the benchmark replay
  rate is 0.026 rad. The knob nevertheless over-rotates, and continues to be
  dragged a further 0.52 rad during `retract`, after the hand has opened.
- **Interpretation**: the gripper wedges against rather than rigidly holding the
  5 cm knob, and the knob joint (`stiffness=1e-3`, `damping=1e2`,
  `max_effort=1e-2`) offers no resistance, so it is cranked past the commanded
  angle. The measurement is confounded by the scene's physics, not by Twist.
- The benchmark reports this as `task_goal_miss` rather than relaxing the rule.
  Section 6 asks how Twist should be gated.

### 4.5 `axis_align`

- **Rule**: the angle between the object's internal axis and the commanded
  target axis is at most `0.15` rad at the end of the `manipulate` segment,
  while the object is still held.
- **Measured**: 0.048 rad (upright) and 0.041 rad (horizontal), both starting
  from a 1.571 rad (90-degree) misalignment.
- **Basis**: matches the angular tolerance used by the other rotation skills.
  The reported `initial_axis_angle_rad` shows the case starts genuinely
  misaligned, so success is not vacuous.

### 4.6 `pour`

- **Rule**: the **peak** signed rotation of the object about its own internal
  axis during the `pour` segment matches the commanded angle within `0.15` rad.
- **Measured**: 0.0020 / 0.0035 / 0.0057 rad error for 30 / 45 / 90-degree pours.
- **Why peak, not end-of-segment**: Pour plans two end-effector targets inside
  one `pour` segment — rotate to the poured pose, then return to the pose it
  started from. The end of the segment is therefore back at zero rotation *by
  construction*; measured end-of-segment rotation was 0.014 rad for a 1.571 rad
  command. The peak is the achieved pour. This matches the design document's
  "peak signed displacement" treatment of contact skills.
- Pour is scored as a two-invocation episode (`pick_up` then `pour`) compiled
  together, per the design rule that sequence success must come from one
  sequential episode. The rotation reference is captured at the pour segment
  start so the preceding PickUp rotation is not credited to Pour.

### 4.7 `hand_over`

- **Rule**: the object settles within `0.12` m of the commanded delivery
  position **and** never falls below `0.35` m (0.15 m under the support
  surface) at any point during the replay.
- **Measured**: delivery distance 0.1123 m (vertical can) and 0.0678 m
  (horizontal pencil); minimum object height 0.531 m and 0.532 m, so neither
  case dropped.
- **Basis**: the 0.12 m delivery tolerance is borrowed from this suite's
  existing `PHYSICAL_MOVE_HELD_OBJECT_XYZ_TOLERANCE_M`. The drop threshold is
  set well below the surface the object was lifted from, so a normal
  mid-transfer dip does not read as a drop.
- **Why the minimum, not the final, height**: an object that is dropped
  mid-transfer can still roll to rest near the target. Tracking the minimum
  height over the whole replay catches that; `object_dropped` is reported as
  its own column and maps to the `object_dropped` taxonomy reason.
- **Caveat**: the vertical-can case passes with only 7 % margin. See the
  report's open questions.
- HandOver picks the object up itself, so no prior PickUp invocation is needed.
- The object's mesh and orientation are fixed when the scene is built, so each
  case runs in its own simulation. `SimulationManager` is a singleton, so the
  benchmark tears the scene down between cases rather than creating a second
  instance.

---

## 5. Cost metrics

### 5.1 Warm-up is mandatory

The first `compile` in a process pays Warp/Torch kernel compilation and
allocator warm-up. Without a warm-up the first case absorbs that cost and the
cases are not comparable. Every benchmark calls `common.warmup_planning` with
one discarded compile of the first case before the measured loop, and resets the
scene afterwards. Reports state this in their notes.

### 5.2 What is measured

`common.timed_call` wraps the measured `compile` call and returns:

- `cost_time_ms` — wall-clock planning time. Includes grasp-pose generation for
  skills that sample grasps; the reports say so.
- `cpu_delta_mb` — process RSS delta across the call (`psutil`, falling back to
  `resource`; the backend is named in the report notes).
- `gpu_delta_mb` — Torch CUDA allocated-memory delta.
- `peak_gpu_mb` — Torch CUDA peak allocated, after
  `reset_peak_memory_stats`.

CUDA is synchronized before and after timing. `trajectory_waypoints` is reported
alongside so planning time can be read per waypoint.

Planning time **excludes** the physical replay; replay is not a property of the
planner.

### 5.3 Reproducing

```
PYTHONPATH=<repo> python -m scripts.benchmark.atomic_action.<skill>_benchmark \
    --profile coverage --device cpu [--n_sample 2000]
```

Skills without grasp sampling (`press`, `twist`) do not accept `--n_sample`.
Reports are written to `outputs/benchmarks/`. `run_benchmark.py` dispatches all
of them via `--action`.

---

## 6. Thresholds that need a decision

These are stated as open questions rather than settled values.

1. **Twist's success rule.** The knob joint measurement is confounded by the
   scene's physics (section 4.4). Options: (a) keep gating on the knob joint and
   accept that Twist fails on this asset until the knob's joint drive is given
   realistic detent stiffness; (b) gate on the executed end-effector rotation
   about the twist axis, which is measurable and currently exact; (c) give the
   benchmark scene a stiffer knob drive than the tutorial's. Option (c) changes
   scene defaults to make a test pass and is not recommended without a reason
   independent of the benchmark. **Suggested: (b) as the gate, keeping the knob
   joint as a reported diagnostic — but this is the user's call.**

2. **Press's discrimination.** The 80 %-of-stroke gate is currently always
   satisfied on this asset regardless of commanded travel (section 4.2). Either
   accept it as a liveness check ("the button was pressed"), or add an asset
   with a stiffer button before treating Press's number as meaningful.
   **Suggested: keep as a liveness check and say so in the report.**

3. **Slide's tolerance.** 0.03 m is ~30x the measured error. Tightening to
   0.01 m would still pass every measured case with 10x margin and would make
   the metric informative. **Suggested: 0.01 m, pending agreement.**

4. **OpenDoor at larger angles.** Error grew monotonically with the commanded
   angle (0.034 / 0.032 / 0.041 rad at 30 / 60 / 90 degrees). A 120-degree case
   was not tested and may approach the 0.15 rad tolerance.

5. **Replay-rate alignment for the five original benchmarks.** They still use 4
   steps per waypoint with no tracking check (section 3.2). They pass today, but
   their contact measurements carry the same systematic bias that the
   calibration exposed.

6. **Repeat count and confidence intervals.** Everything here is `--repeat 1`.
   The design document asks for confidence intervals; single-sample numbers are
   point estimates only.

7. **HandOver's delivery tolerance.** The vertical-can case lands 0.1123 m from
   the target against a 0.12 m gate. Decide whether the tolerance is wrong for
   this skill or the delivery is genuinely marginal.

---

## 7. Adding a new skill

1. Reuse the tutorial's scene helpers; do not re-model the scene.
2. Call `sim.prepare()` after adding robots and objects, before reading
   `body_data` (for example `get_qpos` or `get_local_pose`).
3. Build invocations in a `_build_invocation` helper so the warm-up and the
   measured loop share one code path.
4. Identify the segment that actuates the target and score at its end; use the
   peak instead when the skill returns to its starting pose inside that segment.
5. Drive the ladder through `common.run_articulated_contact_case`, or follow the
   same order explicitly: compile, `check_motion_valid`, replay with tracking,
   then the skill rule.
6. Report the scored, peak, and settled values plus `max_tracking_error_rad`.
7. State the threshold and its measured basis in the report notes, and add it to
   section 4 here.
8. Register the module in `run_benchmark.ACTION_MODULES`.
