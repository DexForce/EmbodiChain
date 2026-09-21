# Atomic-Action Benchmark Standard

These benchmarks measure **skills**, not trajectories. Whether a plan is finite,
stays inside the joint limits, avoids collisions, or is tracked faithfully by
the drive belongs to `scripts/benchmark/motion_generation/`. Those properties
are reported here as diagnostic columns only; none of them fails a skill.

A skill is measured along three dimensions. `robustness` and `adaptability` are
`success_rate` recomputed over a different case set, so they inherit every
definition in section 1.

---

## 0. Tolerances

Two sets, because the two groups are scored at physically different moments --
not because different skills deserve different standards. The split follows the
binding contract: `MoveEndEffector` and `MoveJoints` bind only a `motion`
endpoint, so they are scored after the terminal settle with the drive
converged; the other thirteen also bind a `grasp` endpoint and are scored at
the end of the segment that actuates the target, with the drive in transient
and the arm contact-loaded. Every value is 1.8x the worst error measured for
its set.

| Set | Skills | Scored at | Position | Rotation |
|---|---|---|---|---|
| `PRIMITIVE_*` | motion endpoint only (2) | trajectory end + settle | **3 mm** | **0.3°** |
| `TASK_*` | motion and grasp endpoints (13) | end of the actuating segment | **1 cm** | **5°** |

```python
PRIMITIVE_POSITION_TOLERANCE_M   = 0.003
PRIMITIVE_ROTATION_TOLERANCE_RAD = 0.00524    # 0.3 deg
TASK_POSITION_TOLERANCE_M        = 0.01       # replaces PHYSICAL_PLACE_XY_TOLERANCE_M
TASK_ROTATION_TOLERANCE_RAD      = 0.08727    # 5 deg
                                              # and PHYSICAL_MOVE_HELD_OBJECT_XYZ_TOLERANCE_M

PHYSICAL_PICK_MIN_LIFT_M         = 0.04       # unchanged; not an error against a command
PHYSICAL_PRESS_MIN_STROKE_RATIO  = 0.80
PHYSICAL_DROP_MARGIN_M           = 0.15
```

Calibration. `PRIMITIVE_*`: `MoveEndEffector` settles 0.64-1.54 mm from the
commanded pose, `MoveJoints` 0.00294 rad (0.169°) -- measured by physical
replay, not from `traj[:, -1]`, which reports 0.00002 m and 0.000000 rad
because it describes the solver rather than the robot. `TASK_*`: the worst
measured errors are `Slide` 0.0056 m and `AxisAlign` 0.0480 rad (2.75°),
against a contact-loaded tracking error of 0.0532 rad (3.05°) -- a task
rotation tolerance below ~3° is unreachable in this simulation at any replay
rate. Each pair is one physical error at a characteristic lever arm: 0.57 m
(arm reach) and 0.115 m (handle, drawer front, grasped object). Tolerances are
absolute, never scaled by commanded magnitude.

A skill that breaks and re-establishes contact gets a derived budget rather
than its own constant: `n * TASK_POSITION_TOLERANCE_M` for `n` contact phases.

---

## 1. success_rate

Each skill declares ordered **stages**, each a scene measurement taken at the
end of the segment where the skill stops acting on what the stage is about. A
stage is reached only when every earlier stage passed.

```
stage_success_rate[i] = passed(i) / reached(i)      reached(i) = passed(i-1)
success_rate          = passed(last) / total
```

Closing stages count: a skill that lifts an object and then drops it did not
pick it up.

| Skill | Tolerances | Stages (in order) | Criterion |
|---|---|---|---|
| `MoveEndEffector` | `PRIMITIVE_*` | `reached` | end-effector position error ≤ 3 mm |
| `MoveJoints` | `PRIMITIVE_*` | `reached` | joint position error ≤ 0.3° |
| `PickUp` | `TASK_*` | `grasped` → `lifted` → `held` | held / lift ≥ 4 cm / still held after settle |
| `AxisAlign` | `TASK_*` | `grasped` → `aligned` → `held` | held / axis angle ≤ 5° / still held |
| `MoveHeldObject` | `TASK_*` | `transported` | position ≤ 1 cm and orientation ≤ 5° |
| `Place` | `TASK_*` | `released` → `placed` → `stable` | hand empty / XY ≤ 1 cm / drift ≤ 1 cm·5° |
| `Pour` | `TASK_*` | `poured` | **peak signed** rotation about the object's own axis within 5° |
| `PushObject` | `TASK_*` | `contacted` → `pushed` → `on_support_plane` | object displaced > 0 / in-plane pose ≤ 1 cm / tilt ≤ 5° |
| `Press` | `TASK_*` | `contacted` → `pressed` | button displaced > 0 / peak signed ≥ 80 % of stroke |
| `Slide` | `TASK_*` | `grasped` → `slid` → `released` | handle held / travel within 1 cm / released |
| `OpenDoor` | `TASK_*` | `grasped` → `opened` → `released` | handle held / hinge angle within 5° / released |
| `Twist` | `TASK_*` | `grasped` → `twisted` → `released` | knob held / **end-effector rotation about the knob axis** within 5° / released |
| `CoordinatedPickment` | `TASK_*` | `dual_grasped` → `lifted` → `moved` → `held` | both hands contact / lift ≥ 4 cm / pose ≤ 1 cm·5° / still pinched |
| `CoordinatedPlacement` | `TASK_*` | `aligned` → `released` → `stable` | relative pose ≤ 1 cm·5° / both hands open / drift ≤ 1 cm·5° |
| `HandOver` | `TASK_*` | `grasped` → `transferred` → `handed_over` → `placed` | handing arm holds / both arms hold / released and still held / delivery ≤ 3 cm |

Cross-cutting: for any skill holding an object, falling more than
`PHYSICAL_DROP_MARGIN_M` below the support plane at any point fails the current
stage with `object_dropped`. The minimum over the replay is used, not the final
height, because an object dropped mid-motion can still roll to rest near the
target.

Failure stage and reason are stored per case. Reasons are the task-level subset
of the taxonomy in `motion_generation/BENCHMARK_DESIGN.md` section 7:
`invalid_case`, `unsupported_capability`, `planner_exception`,
`planner_reported_failure`, `timeout`, `object_not_grasped`, `object_dropped`,
`release_failure`, `task_goal_miss`.

---

## 2. robustness

The same task under perturbation, reported against the nominal `success_rate`.

| Axis | What varies | Applies to |
|---|---|---|
| Initial pose | robot start `qpos` noise | all |
| Target pose | object or articulated part translated and rotated on its support plane | motion-and-grasp skills |
| Command parameter | opening angle, travel, pour angle jittered within the valid range | `Pour`, `Press`, `Slide`, `OpenDoor`, `Twist` |
| Friction | contact friction coefficient scaled | motion-and-grasp skills |
| Mass | manipulated object mass scaled | object-holding skills |
| Grasp sampling | same case, different grasp-pose seed | skills that sample grasps |

---

## 3. adaptability

A different object, scene, or embodiment. Reported as the minimum and the
variance of `success_rate` across the set.

| Axis | What varies | Applies to |
|---|---|---|
| Object geometry | mesh swapped | object-holding skills |
| Articulated asset | door / drawer / knob / button asset swapped | `Press`, `Slide`, `OpenDoor`, `Twist` |
| Scene layout | support plane height, obstacles, workspace-edge poses | all |
| End effector | gripper swapped (PGI → other) | all |
| Embodiment | arm swapped (UR5 → other) | all |
| Command extrapolation | beyond the calibrated range (e.g. a 120° door) | `Pour`, `Press`, `Slide`, `OpenDoor`, `Twist` |

---

## 4. Notes

- **Replay rate is 16 physics steps per waypoint.** This is a replay
  configuration, not a criterion. Replaying at the planner's own `dt` is the
  worst setting: it produces tracking errors above 1 rad and makes every
  articulated-contact skill look broken. `max_tracking_error_rad` is reported
  per case as a diagnostic so a reader can judge whether a scene measurement is
  trustworthy.
- **Skills are scored while the robot still controls the target.** After
  release and retract, an undriven hinge, drawer, or knob moves under its own
  dynamics: a 60° `OpenDoor` case reached 0.979 rad at the end of `open` and
  settled back to 0.667 rad.
- **`Twist`'s knob joint is a diagnostic, not the criterion.** The knob
  over-rotates (2.38-3.14 rad for 0.52-1.57 rad commanded) because the gripper
  wedges against the 5 cm knob and the joint offers no resistance. The executed
  end-effector rotation about the knob axis is exact, and is what `twisted`
  measures.
- **`Press`'s criterion only shows that the button moved.** Commanding 3 mm
  still drives the button to 5.66 mm of its 6 mm stroke, so the 80 % ratio does
  not discriminate commanded depth on this asset.
- **Not yet benchmarked:** `PushObject`, `CoordinatedPickment`,
  `CoordinatedPlacement` have stages defined above but no benchmark module.
- **Single-sample estimates.** Everything is `--repeat 1`; these are point
  estimates, not confidence intervals.
