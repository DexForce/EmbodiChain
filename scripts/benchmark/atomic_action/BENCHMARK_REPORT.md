# Atomic-Action Benchmark: Results and Technical Report

Branch `xinyi/atomic-bench-all` in `/tmp/bench_night`, based on `main`
(`7bc7271c`). Nothing is committed; the working tree holds all changes.

Environment: `embodichain311` (Python 3.11), `--device cpu`, DexSim default
backend, physics timestep 10 ms, single environment. Reports in
`outputs/benchmarks/`. Metric definitions and threshold justifications live in
[`BENCHMARK_STANDARD.md`](BENCHMARK_STANDARD.md).

---

## 1. One-page summary

**12 atomic-action benchmarks now run end to end.** Five existed but could not
run on `main`; seven are new.

| Skill | Status | plan | valid | exec | task |
|---|---|---|---|---|---|
| `move_joints` | restored | ok | ok | ok | ok |
| `move_end_effector` | restored | ok | ok | ok | ok |
| `pick_up` | restored | ok | ok | ok | ok |
| `place` | restored | ok | ok | ok | ok |
| `move_held_object` | restored | ok | ok | ok | ok |
| `open_door` | **new** | 3/3 | 3/3 | 3/3 | **3/3** |
| `press` | **new** | 3/3 | 3/3 | 3/3 | **3/3** |
| `slide` | **new** | 3/3 | 3/3 | 3/3 | **3/3** |
| `axis_align` | **new** | 2/2 | 2/2 | 2/2 | **2/2** |
| `pour` | **new** | 3/3 | 3/3 | 3/3 | **3/3** |
| `hand_over` | **new** | 2/2 | 2/2 | 2/2 | **2/2** |
| `twist` | **new** | 3/3 | 3/3 | 3/3 | **0/3** |

**Twist is the only skill that does not reach `task_success`, and the cause is
not Twist's planning.** Its planned end-effector rotation about the knob axis is
exactly the commanded angle; the simulated grasp cranks the freely-spinning knob
past it. Details in section 4.

**Not completed:** `push_object`, `coordinated_pickment`,
`coordinated_placement`. Section 6.

**Needs your decision:** seven threshold/scope questions, section 7.

---

## 2. What was broken, and what fixing it required

### 2.1 Restoring the five shipped benchmarks

PR #620's fixes were re-applied by cherry-picking `42728c12` with `-n` (staged,
not committed): missing tutorial exports, missing planning context, missing
simulator teardown, a stale `get_hand_open_close_qpos` signature, and a
too-tight `move_end_effector` tolerance.

That alone was not enough. **Additional fix required by the current DexSim:**
the extracted `create_robot` helpers never called `sim.prepare()`, so
`robot.body_data` was `None` and every benchmark died at the first
`robot.get_qpos()`.

- *Before*: `AttributeError: 'NoneType' object has no attribute 'qpos'` in
  `articulation.py:1619`, for all five.
- *Fix*: `sim.prepare()` inside each tutorial `create_robot`, and inside
  `common.create_benchmark_object` after `add_rigid_object` (adding a body
  changes the spawn topology).
- *After*: all five run and write reports. `move_joints` 123 ms, success, clean
  exit.

### 2.2 The replay-rate defect (affects every contact skill)

This was the most consequential finding of the night.

`Robot.set_qpos` writes **drive targets**, not teleported state. The benchmark
replay advanced one waypoint per 4 physics steps with no check that the arm had
actually got there. Sweeping the rate on OpenDoor:

| Steps/waypoint | Max arm tracking error (rad) | Hinge error (rad) |
|---|---|---|
| 1 (planner's own dt) | 1.3844 | 0.2273 |
| 4 (original) | 0.4390 | 0.0689 |
| 16 (**chosen**) | 0.0532 | 0.0306 |
| 32 | 0.0230 | 0.0245 |

The measured "task" error was largely the controller failing to follow the plan.
I first tried replaying at the planner's own `dt`, which is *worse* — it made
OpenDoor and Slide fail outright.

**Fix**: `DEFAULT_REPLAY_STEPS_PER_WAYPOINT = 16`, plus a new
`execution_success` gate on `max_tracking_error_rad <= 0.10`, so a case is never
credited or blamed for motion the robot did not perform.

*Evidence, Slide 180 mm pull:* 0.0525 m error at 1 step → 0.0098 m at 4 →
**0.0006 m at 16**.

### 2.3 Measurement-point defects (my benchmark code, found by disbelieving green/red)

Three skills were initially mis-measured. In each case the skill was correct and
the benchmark was wrong.

**`open_door`** — first run reported `task_success=False`, hinge error 0.380 rad.
Instrumenting each segment boundary showed the hinge reached 0.979 rad at the
end of the `open` segment and then **settled back to 0.667 rad** after the hand
released and the arm retracted: the tutorial microwave's hinge is undriven
(`drive_type="none"`) and swings freely. Scoring moved to the end of the `open`
segment. Settled value retained as a diagnostic only.

**`pour`** — reported `task_success=False` with ~0.014 rad achieved for a
1.571 rad command. The planned `pour` segment showed
`max_joint_change_over_pour_segment = 0.0000 rad`. Reading
`primitives/pour.py` explained it: Pour plans **two** end-effector targets in
one segment — rotate to the poured pose, then *return to the pose it started
from*. The end of the segment is zero rotation by construction. Scoring moved to
the peak signed rotation. Error dropped to 0.002–0.006 rad.

**`pour` (second defect, mine)** — PickUp refused to plan in the Pour scene
(`plan failure [pick_up]: planning failed`). Cause: I passed
`strategy="motion_gen"` on the PickUp invocation. Removing it (matching the
working `pickup_benchmark`) made it plan. See 2.4 for the related tutorial bug.

**`axis_align`** — `KeyError: "axis_align_object_pose references unknown scene
entity 'cube'"`. AxisAlign resolves the object pose from the scene by entity id,
so the engine must be built with
`create_simulation_atomic_action_engine(scene_entities=(obj,))`, not a bare
`AtomicActionEngine`. Fixed; both cases now pass from a 90-degree misalignment.

### 2.4 A pre-existing bug found in the Pour tutorial

Running `scripts/tutorials/atomic_action/pour.py` **on this branch, unmodified**,
crashes:

```
ValueError: Quantity sampling requires at least one sample per retained
waypoint; received 108, requires at least 277.
```

`PICK_MOTION_SAMPLE_COUNT = PICK_SAMPLE_INTERVAL - HAND_INTERP_STEPS` = 108, but
the retained waypoint count is 277. The tutorial's `_create_pick_motion_policy`
forces `strategy="motion_gen"` with an under-sized `sample_interval`.

I did **not** change the tutorial — the benchmark sidesteps it by leaving the
strategy at its default, and the tutorial fix is a separate decision (raise the
sample count, or stop forcing `motion_gen`). Flagging it because it means Pour's
tutorial is currently unusable.

---

## 3. Results

All numbers from the final verification run after `black` formatting. Planning
time excludes a discarded warm-up compile.

### 3.1 Success ladder and task metrics

| Skill | Case | plan | valid | exec | task | task metric (measured vs rule) |
|---|---|---|---|---|---|---|
| `open_door` | open_30 | ok | ok | ok | **ok** | hinge err 0.0318 rad ≤ 0.15 |
| | open_60 | ok | ok | ok | **ok** | hinge err 0.0332 rad ≤ 0.15 |
| | open_90 | ok | ok | ok | **ok** | hinge err 0.0415 rad ≤ 0.15 |
| `press` | press_10mm | ok | ok | ok | **ok** | peak 6.00 mm ≥ 4.80 mm (80 % of stroke) |
| | press_20mm | ok | ok | ok | **ok** | peak 6.00 mm ≥ 4.80 mm |
| | press_30mm | ok | ok | ok | **ok** | peak 6.00 mm ≥ 4.80 mm |
| `slide` | pull_100mm | ok | ok | ok | **ok** | travel 0.0998 m, err 0.0002 m ≤ 0.03 |
| | pull_180mm | ok | ok | ok | **ok** | travel 0.1794 m, err 0.0006 m ≤ 0.03 |
| | pull_250mm | ok | ok | ok | **ok** | travel 0.2444 m, err 0.0056 m ≤ 0.03 |
| `axis_align` | upright | ok | ok | ok | **ok** | 1.5709 → 0.0480 rad ≤ 0.15 |
| | horizontal | ok | ok | ok | **ok** | 1.5708 → 0.0412 rad ≤ 0.15 |
| `pour` | pour_30 | ok | ok | ok | **ok** | peak 0.5217 rad, err 0.0019 ≤ 0.15 |
| | pour_45 | ok | ok | ok | **ok** | peak 0.7796 rad, err 0.0058 ≤ 0.15 |
| | pour_90 | ok | ok | ok | **ok** | peak 1.5649 rad, err 0.0059 ≤ 0.15 |
| `hand_over` | vertical_can | ok | ok | ok | **ok** | delivery 0.1123 m ≤ 0.12; min z 0.531 m, not dropped |
| | horizontal_pencil | ok | ok | ok | **ok** | delivery 0.0678 m ≤ 0.12; min z 0.532 m, not dropped |
| `twist` | twist_ccw_30 | ok | ok | ok | **FAIL** | knob 2.0286 rad vs 0.5236 commanded, err 1.5050 |
| | twist_ccw_45 | ok | ok | ok | **FAIL** | knob 2.4024 rad vs 0.7854, err 1.6170 |
| | twist_ccw_90 | ok | ok | ok | **FAIL** | knob 3.1416 rad vs 1.5708, err 1.5708 |

Restored benchmarks (smoke profile): `move_joints` success, joint err 0.000000;
`move_end_effector` success, err 0.0000; `pick_up` success, object lift
0.158 m, XY drift 0.004 m; `place` success; `move_held_object` success.

### 3.2 Cost

| Skill | Planning time (ms) | Waypoints | CPU delta (MB) | Peak GPU (MB) |
|---|---|---|---|---|
| `open_door` | 270–287 | 120 | 0.0–7.6 | 29.1 |
| `press` | 316–321 | 140 | 0.0 | 0.0 |
| `slide` | 315–325 | 140 | 0.0–7.7 | 45.7 |
| `twist` | 229–237 | 140 | 0.0 | 0.0 |
| `axis_align` | 321–332 | 180 | 0.0–8.4 | 24.5 |
| `pour` (pick+pour) | 278–284 | 200 | 0.8–6.8 | 42.8 |
| `hand_over` | 665–685 | 220 | — | — |
| `move_joints` | 123 | — | — | — |
| `move_end_effector` | 69 | — | — | — |
| `pick_up` | 579 | 120 | — | — |

Run on CPU, so GPU deltas are Torch-side only. `press` and `twist` allocate no
GPU memory because they sample no grasps. The CPU delta of the first case in
each sweep is larger than subsequent ones (allocator growth) even after warm-up;
this is visible in the per-case tables.

---

## 4. Twist: root-cause analysis

`twist` is the one skill whose `task_success` fails. The evidence says the
failure is in the scene's contact physics, not in Twist.

**What was ruled out:**

1. *Replay rate.* Arm tracking error at the benchmark rate is **0.026 rad** —
   the arm follows the plan almost exactly. Over-rotation persists at every rate
   from 2 to 32 steps per waypoint.
2. *Grasp not closing.* The hand is commanded to 0.036 and stalls at 0.0136,
   i.e. it is physically blocked by the 5 cm knob. Contact is real.
3. *Planning.* Measured directly: the resolved twist axis is
   `(-1, 0, 0)` in world, and the **planned end-effector rotation about that
   axis over the `twist` segment is 0.7854 rad for a 0.7854 rad command, with
   the correct sign.** Twist plans exactly what it was asked to.

**What the instrumentation shows instead:** per-segment knob angle for a
0.7854 rad command —

```
end_of_close     knob=+0.0108
end_of_twist     knob=+1.8736   <- 2.4x commanded, opposite sign
end_of_open      knob=+1.8524   (hand has opened)
end_of_retract   knob=+2.3749   <- still being dragged after release
after_hold       knob=+2.3749   drift=0.0000
```

The knob keeps being driven a further **0.52 rad during `retract`, after the
hand opened**, and then stops dead (zero drift). It is not spinning on
momentum — it is being cranked by contact with the withdrawing fingers.

**Mechanism:** the gripper wedges against the knob rather than holding it
rigidly, and the knob joint has `stiffness=1e-3`, `damping=1e2`,
`max_effort=1e-2` — essentially no restoring torque — so any tangential contact
force turns it freely past the commanded angle.

**I did not "fix" this by loosening the tolerance or by stiffening the knob's
joint drive.** Either would manufacture a pass. The benchmark reports
`task_goal_miss` with the measured value, and section 7 asks how you want Twist
gated.

---

## 5. Changes made

Working tree on `xinyi/atomic-bench-all`, uncommitted.

**New benchmarks** (`scripts/benchmark/atomic_action/`): `open_door_benchmark.py`,
`press_benchmark.py`, `slide_benchmark.py`, `twist_benchmark.py`,
`axis_align_benchmark.py`, `pour_benchmark.py`, `hand_over_benchmark.py`. All
registered in
`run_benchmark.ACTION_MODULES`.

**`common.py`** — new shared infrastructure:
`SuccessLadder`, `FAILURE_TAXONOMY`, `LADDER_STAGES`, `check_motion_valid`,
`JointDisplacementTrace`, `replay_and_track_scalar` (the single replay
primitive), `replay_and_track_joint`, `run_articulated_contact_case`,
`waypoint_step_counts`, `warmup_planning`, `build_ladder_leaderboard`,
`DEFAULT_REPLAY_STEPS_PER_WAYPOINT`, `REPLAY_TRACKING_TOLERANCE_RAD`.
Plus the `sim.prepare()` fix in `create_benchmark_object`.

**Tutorials** — cherry-picked #620 exports, plus `sim.prepare()` in each
extracted `create_robot`. No tutorial behaviour was otherwise changed; in
particular the Pour tutorial's bug (2.4) is reported, not patched.

**Formatting** — `black==26.3.1` clean across
`scripts/benchmark/atomic_action/` and `scripts/tutorials/atomic_action/`.

All 12 modules import and expose `run_all_benchmarks` / `add_benchmark_args`.

---

## 6. Not completed

| Skill | Status | Reason |
|---|---|---|
| `push_object` | not built | No tutorial. The only reference usage is `tests/sim/atomic_actions/test_actions.py`, which drives it with a **mocked** motion generator and synthetic `SceneSnapshot` poses — no physical scene exists to reuse, so a benchmark needs a scene designed from scratch (object, support surface, contact offsets). Deferred after the priority-2 skills. |
| `coordinated_pickment` | not built | Dual-arm, `left`/`right` binding. Not reached. |
| `coordinated_placement` | not built | Dual-arm, and the most involved: requires two sequential PickUps, then re-deriving held state from actual measurements (`compute_actual_held_state`) before planning placement. Not reached. |

The API facts needed for all four were gathered and are summarized in this
session; none is blocked by a missing dependency. **No skill was blocked by
cuRobo** — every benchmark here uses the TOPPRA/trapezoidal path, so the
"needs cuRobo, cannot verify locally" case did not arise.

---

## 7. Decisions needed from you

1. **How should `twist` be gated?** Options: (a) keep the knob-joint rule and
   accept Twist failing on this asset; (b) gate on the executed end-effector
   rotation about the twist axis, which is currently exact; (c) give the
   benchmark scene a stiffer knob drive than the tutorial's. *Suggested: (b),
   keeping the knob joint as a reported diagnostic. (c) is tuning a scene to make
   a test pass and I'd advise against it without an independent reason.*
   **Status: undecided.**

2. **Is `press`'s gate meaningful?** Commanding 3 mm still drove the button to
   5.66 mm of its 6 mm stroke, so the 80 %-of-stroke rule passes regardless of
   commanded travel. *Suggested: keep it as an explicit liveness check ("the
   button was pressed") and stop treating commanded depth as a controlled
   variable on this asset.* **Status: undecided.**

3. **Tighten `slide`'s tolerance?** 0.03 m is ~30x the measured error (max
   0.0056 m). *Suggested: 0.01 m, which still leaves ~2x margin on the worst
   measured case.* **Status: undecided.**

4. **Should the five original benchmarks adopt the calibrated replay?** They
   still use 4 steps/waypoint via
   `replay_trajectory_for_physical_validation`, with no tracking check. They
   pass, but carry the same systematic bias section 2.2 quantifies. I left their
   behaviour unchanged deliberately. *Suggested: align them, in a separate
   change.* **Status: undecided.**

5. **`open_door` at larger angles.** Error grows with commanded angle
   (0.0318 / 0.0332 / 0.0415 rad at 30 / 60 / 90 degrees). A 120-degree case was
   not tested and could approach the 0.15 rad tolerance. Worth adding before
   trusting the tolerance at the extremes.

6. **Repeat count.** Everything is `--repeat 1`; these are point estimates. The
   design document asks for confidence intervals, which needs repeats and a
   seeding policy.

7. **`hand_over`'s delivery tolerance.** 0.12 m was borrowed from the existing
   MoveHeldObject tolerance. The vertical-can case lands at 0.1123 m, i.e. only
   7 % inside the gate, so it would flip to a failure under small perturbation.
   Either the tolerance is too tight for HandOver, or the delivery is genuinely
   marginal and worth investigating. *Suggested: investigate before loosening.*
   **Status: undecided.**

Also awaiting your call: whether to fix the Pour tutorial bug (2.4) here or in a
separate change.

---

## 8. Reproducing

```
cd /tmp/bench_night
export PYTHONPATH=/tmp/bench_night
PY=/home/lqin/miniconda3/envs/embodichain311/bin/python

$PY -m scripts.benchmark.atomic_action.open_door_benchmark  --profile coverage --device cpu --n_sample 2000
$PY -m scripts.benchmark.atomic_action.slide_benchmark      --profile coverage --device cpu --n_sample 2000
$PY -m scripts.benchmark.atomic_action.axis_align_benchmark --profile coverage --device cpu --n_sample 2000
$PY -m scripts.benchmark.atomic_action.pour_benchmark       --profile coverage --device cpu --n_sample 2000
$PY -m scripts.benchmark.atomic_action.press_benchmark      --profile coverage --device cpu
$PY -m scripts.benchmark.atomic_action.twist_benchmark      --profile coverage --device cpu
$PY -m scripts.benchmark.atomic_action.hand_over_benchmark  --profile coverage --device cpu --n_sample 2000
```

`press` and `twist` sample no grasps and do not accept `--n_sample`. A DexSim
exit code of 139 on teardown is the known-harmless segfault; judge each run by
its printed output and the written report.
