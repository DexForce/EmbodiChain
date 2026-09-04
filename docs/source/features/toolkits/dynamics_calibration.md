# Dynamics Calibration

The dynamics-calibration toolkit tunes effective robot drive properties for a
specific EmbodiChain application. It produces a reviewable YAML overlay and
qualification evidence; it never rewrites the source URDF.

This V1 intentionally does **not** claim physical parameter identification.
Without torque/current or equivalent real-system evidence, mass, center of
mass, inertia, friction, stiffness, and damping are not uniquely identifiable
from position tracking alone.

## Workflow and ownership

The workflow has three commands:

1. `audit` delegates generic URDF and inertia checks to `dexsim.simready`.
2. `tune-drive` runs each candidate in a fresh process, ranks candidates on
   training trajectories, then checks the winner on a held-out trajectory.
3. `qualify` rechecks an existing overlay against the current asset hashes and
   held-out gates.

DexSim owns simulation-readiness facts. EmbodiChain owns application
trajectories, control-group selection, drive-parameter search, qualification
policy, and report assembly. An error-level DexSim finding blocks tuning;
warnings remain visible in the final report but permit application evaluation.

## Configuration

Save a YAML file such as `calibration.yaml`:

```yaml
schema_version: 1
assets:
  - /absolute/path/to/robot.urdf
backend: physx
device: cpu
physics_dt: 0.004166666666666667  # 240 Hz
control_frequency_hz: 60
seed: 7
candidate_count: 9

evaluator:
  target: embodichain.toolkits.dynamics_calibration.tracking_evaluator:evaluate
  timeout_seconds: 120
  payload:
    control_part: arm
    robot_cfg:
      control_parts:
        arm: [joint1, joint2, joint3]
    training_trajectory:
      duration_seconds: 3
      warmup_seconds: 0.5
      amplitude: [0.10, 0.08, 0.06]
      frequencies_hz: [0.25, 0.35, 0.45]
    qualification_trajectory:
      duration_seconds: 4
      warmup_seconds: 0.5
      amplitude: [0.07, 0.11, 0.09]
      frequencies_hz: [0.30, 0.40, 0.55]

parameters:
  - name: arm_stiffness
    field: stiffness
    selector: arm
    lower: 100
    upper: 20000
    initial: 5000
    scale: log
  - name: arm_damping
    field: damping
    selector: arm
    lower: 10
    upper: 2000
    initial: 500
    scale: log

qualification:
  aggregate_rmse_max: 0.05
  per_joint_rmse_max: 0.08
  per_control_group_rmse_max: 0.06
  cvar95_max: 0.12
  # Custom step-response evaluators can also gate overshoot_max and
  # settling_time_seconds_max when they return those metrics.
  saturation_fraction_max: 0.02
  velocity_saturation_fraction_max: 0.02
  joint_limit_violation_max: 0
  control_frequency_relative_error_max: 0
  expected_target_qvel_write_count: 0
  require_stable: true
```

Parameter `selector` values use the same exact-name, regular-expression, and
Robot control-part resolution as `RobotCfg.drive_pros`. The built-in evaluator
only writes qpos targets. Its qvel write count comes from public
`Articulation.set_qvel` instrumentation and counts successful batched target
write calls; it is not inferred from private engine state.

The requested control frequency must map to an integral number of physics
updates. Set `allow_approximate_control_frequency: true` only when a changed
actual rate is acceptable and covered by the configured frequency-error gate.
The default qpos-only policy also requires `target_qvel_write_count` evidence;
an evaluator that omits it fails that gate rather than being assumed to have
written zero velocity targets. Set `expected_target_qvel_write_count: null`
only when that instrumentation is intentionally unavailable.

## Commands

```bash
# Inspect one or more URDFs without changing them.
embodichain calibrate-dynamics audit robot.urdf --output-dir audit_output

# Audit, search, write the overlay, and run held-out qualification.
embodichain calibrate-dynamics tune-drive \
  --config calibration.yaml \
  --output-dir calibration_output

# Re-qualify a previously generated overlay.
embodichain calibrate-dynamics qualify \
  --config calibration.yaml \
  --overlay calibration_output/drive_overlay.yaml \
  --output-dir qualification_output
```

An audit failure, worker exception, timeout, stale asset hash, or qualification
failure exits nonzero. Candidate results are content-addressed by the assets,
overlay, evaluator, runtime, backend, timestep, control schedule, seed, and
phase.

## Artifacts

`tune-drive` writes:

- `drive_overlay.yaml`: the selected `drive_pros` values and exact asset hashes;
- `report.json`: all candidates, raw metrics, hard-gate decisions, versions,
  timing, device, backend, and cache provenance;
- `report.md`: a compact human review;
- `cache/`: reusable isolated-candidate results.

The report claim is always `effective_drive_tuning`. Confidence intervals and
domain-randomization ranges are marked as not estimated in V1; those require a
separate physical-identification workflow and suitable measurements.

## Custom application evaluator

Set `evaluator.target` to `module:function` or `/path/to/file.py:function`.
The callable receives `(overlay, context)` and returns a dictionary containing:

```python
{
    "joint_names": ["joint1", "joint2"],
    "target_qpos": [[0.0, 0.0], [0.1, -0.1]],
    "actual_qpos": [[0.0, 0.0], [0.08, -0.12]],
    "requested_control_hz": 60.0,
    "actual_control_hz": 60.0,
    "target_qvel_write_count": 0,
    # Optional: control_groups, effort/effort_limits, qvel/qvel_limits,
    # qpos_lower/qpos_upper, overshoot, settling_time_seconds, stable,
    # and JSON-serializable metadata.
}
```

Tracking-error, saturation, limit metrics, and all hard gates remain centralized
in the toolkit. A custom step-response evaluator may additionally supply
application-defined `overshoot` and `settling_time_seconds` observations; the
toolkit validates them as finite non-negative values and applies the configured
gates without transforming them.
