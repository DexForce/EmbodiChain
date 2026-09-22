# Atomic-Action Benchmark: Results

Twelve skills measured under
[`BENCHMARK_STANDARD.md`](BENCHMARK_STANDARD.md). Every number below is a
`--profile coverage --device cpu` run on `embodichain311` (Python 3.11), DexSim
default backend, 10 ms physics timestep, one environment, `--repeat 1`. Per-case
tables are in `outputs/benchmarks/`.

`success_rate` is the last link of a chain. Each stage is scored only on the
cases that reached it, so a rate says what fraction of the cases that got there
passed, and the product of the chain is the suite's `success_rate`:

```
stage_success_rate[i] = passed(i) / reached(i)      reached(i) = passed(i-1)
success_rate          = passed(last) / total
```

Plan validity and replay tracking are reported per case and fail nothing.

---

## 1. Results

| Skill | Cases | Stage chain | `success_rate` |
|---|---|---|---|
| `move_joints` | 3 | reached 100% | **100%** |
| `move_end_effector` | 4 | reached 100% | **100%** |
| `open_door` | 3 | grasped 100% → opened 100% → released 100% | **100%** |
| `slide` | 3 | grasped 100% → slid 100% → released 100% | **100%** |
| `press` | 3 | contacted 100% → pressed 100% | **100%** |
| `pour` | 3 | poured 100% | **100%** |
| `twist` | 3 | grasped 100% → twisted 100% → released 100% | **100%** |
| `axis_align` | 2 | grasped 100% → aligned 100% → held 100% | **100%** |
| `pick_up` | 48 | grasped 87.50% → lifted 59.52% → held 100% | **52.08%** |
| `place` | 48 | released 87.50% → placed 42.86% → stable 100% | **37.50%** |
| `move_held_object` | 48 | transported 27.08% | **27.08%** |
| `hand_over` | 2 | grasped 100% → transferred 100% → handed_over 100% → placed 0% | **0%** |

Measured values behind the passing skills, against the tolerance each is scored
at:

| Skill | Measured | Tolerance |
|---|---|---|
| `move_joints` | 0.002996 rad, all three cases | 0.00524 rad |
| `move_end_effector` | 0.0006 – 0.0015 m | 0.003 m |
| `open_door` | hinge error 0.0232 – 0.0289 rad | 0.08727 rad |
| `slide` | travel error 0.0002 – 0.0010 m | 0.01 m |
| `press` | peak 6.00 mm of a 6 mm stroke, all three cases | 80 % of stroke |
| `pour` | peak-rotation error 0.0035 – 0.0054 rad | 0.08727 rad |
| `twist` | end-effector rotation error 0.0024 – 0.0085 rad | 0.08727 rad |
| `axis_align` | residual axis angle 0.0480, 0.0485 rad | 0.08727 rad |

`move_joints` and `move_end_effector` are read off the robot after a physical
replay and a terminal settle. Read from the planner's last waypoint, as the
suite did before, they report 0.000000 rad and 0.00002 m, which describes the
solver.

---

## 2. Where the four failing skills lose their cases

### `pick_up` — 52.08 %, and the grasp state disagrees with the scene

| Outcome | Cases |
|---|---|
| Passed every stage | 25 |
| Failed `lifted` | 17 |
| Failed `grasped`, `planner_reported_failure` | 6 |

Of the 17 that failed `lifted`, **15 never moved the object at all** — the
measured lift is under 5 mm. The skill's own held-object state reports a grasp
in every one of them, which is what `grasped` scores; the object stays on the
table. That gap between the held-object state and the scene is the finding, and
it is only visible because the two are separate stages.

`held` is 100 %: nothing that came up ever fell.

### `place` — 37.50 %, and the tighter tolerance is not the main cause

| Outcome | Cases |
|---|---|
| Passed every stage | 18 |
| Failed `placed` | 24 |
| Failed `released`, PickUp precondition raised | 6 |

XY error over the 42 scored cases has a median of 0.0163 m. At the standard's
1 cm, 18 pass; at the 10 cm the suite used before, 25 would. **The tolerance
change accounts for 7 cases; the other 17 miss by more than 10 cm.** `stable`
is 100 %: whatever landed on the target stayed there.

### `move_held_object` — 27.08 %, and the misses are half a metre

| Position and rotation error, 40 scored cases | Cases |
|---|---|
| Within 1 cm and 5° | 13 |
| Rotation only outside | 1 |
| Both outside | 26 |

Median position error is **0.5411 m**, maximum 1.6343 m. At the 12 cm the suite
used before, only 15 of the 40 would pass, so this is not a tolerance effect:
the skill puts the object somewhere else entirely in most coverage poses. The
remaining 8 of the 48 produced no measurement because the case failed earlier.

This skill also gained an orientation criterion. Its goal commands a pose, not
a point, and only the position was scored before; exactly one case fails on
rotation alone, so the criterion is not what moved the number.

### `hand_over` — 0 %, failing only at the last stage

Both cases pass `grasped`, `transferred` and `handed_over`, then miss `placed`:
delivery distance 0.2372 m and 0.0764 m against the derived budget of
3 × `TASK_POSITION_TOLERANCE_M` = 0.03 m. Neither case dropped the object.

An earlier run of the same two cases measured 0.1123 m and 0.0678 m. Grasp
sampling is stochastic and every number here is `--repeat 1`, so these are point
estimates, not confidence intervals.

---

## 3. The replay rate does not decide these results

The physical-validation replay moved from 4 to the 16 physics steps per
waypoint the standard calibrates at. Running `pick_up` coverage at both rates:

| Steps per waypoint | grasped | lifted | held | `success_rate` |
|---|---|---|---|---|
| 4 (previous) | 87.50% | 59.52% | 100.00% | 52.08% |
| 16 (standard) | 87.50% | 59.52% | 100.00% | 52.08% |

Identical in aggregate. Six of the 48 cases swap verdicts, three in each
direction — margin noise. The rate changes measurement fidelity, not the thing
being measured.

---

## 4. Two criteria that do not discriminate

- **`press`** passes at every commanded depth. Commanding 10, 20 or 30 mm all
  drive the button to 6.00 mm of its 6 mm stroke, so the 80 % rule shows that
  the button was pressed and nothing about how far it was asked to go.
- **`twist`'s knob joint** is a diagnostic, not the criterion. It reaches
  2.0261, 2.5326 and 3.1416 rad for 0.5236, 0.7854 and 1.5708 rad commanded:
  the gripper wedges against the 5 cm knob and the joint offers no resistance,
  and the 90° case is simply at the asset's limit. The executed end-effector
  rotation about the knob axis, which the standard scores, is within 0.0085 rad
  in all three cases. The axis is probed off the asset rather than assumed.

---

## 5. Not covered

`push_object`, `coordinated_pickment` and `coordinated_placement` have stages
defined in the standard but no benchmark module.

`robustness` and `adaptability` are not measured here. This report is section 1
of the standard only.

---

## 6. Reproducing

```
PY=/home/lqin/miniconda3/envs/embodichain311/bin/python

$PY -m scripts.benchmark.atomic_action.move_joints_benchmark       --profile coverage --device cpu
$PY -m scripts.benchmark.atomic_action.move_end_effector_benchmark --profile coverage --device cpu
$PY -m scripts.benchmark.atomic_action.pickup_benchmark            --profile coverage --device cpu --n_sample 2000
$PY -m scripts.benchmark.atomic_action.place_benchmark             --profile coverage --device cpu --n_sample 2000
$PY -m scripts.benchmark.atomic_action.move_held_object_benchmark  --profile coverage --device cpu --n_sample 2000
$PY -m scripts.benchmark.atomic_action.open_door_benchmark         --profile coverage --device cpu --n_sample 2000
$PY -m scripts.benchmark.atomic_action.slide_benchmark             --profile coverage --device cpu --n_sample 2000
$PY -m scripts.benchmark.atomic_action.axis_align_benchmark        --profile coverage --device cpu --n_sample 2000
$PY -m scripts.benchmark.atomic_action.pour_benchmark              --profile coverage --device cpu --n_sample 2000
$PY -m scripts.benchmark.atomic_action.press_benchmark             --profile coverage --device cpu
$PY -m scripts.benchmark.atomic_action.twist_benchmark             --profile coverage --device cpu
$PY -m scripts.benchmark.atomic_action.hand_over_benchmark         --profile coverage --device cpu --n_sample 2000
```

`press` and `twist` sample no grasps and do not accept `--n_sample`. A DexSim
exit code of 139 on teardown is the known-harmless segfault; judge a run by its
printed output and the written report.
