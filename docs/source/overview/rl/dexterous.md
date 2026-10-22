# Pen and cube manipulation

Five vector tasks use the standard learning-environment interface and PPO trainer.
Each deployment selects its solver explicitly and obtains models and initial
states through the EmbodiChain asset registry.

| Task directory | Registered environment | Actions | Actor / critic observations | Asset |
|---|---|---:|---:|---|
| `shadow_pen_fixed` | `ShadowPenFixedDexSim-v1` | 18 | 87 / 87 | `ShadowHandPen` |
| `shadow_pen_continuous` | `ShadowPenContinuousDexSim-v1` | 18 | 87 / 87 | `ShadowHandPen` |
| `leap_cube_face_turn` | `LeapCubeFaceTurnDexSim-v1` | 16 | 65 / 65 | `LeapHandCubeFaceTurn` |
| `leap_cube_reorient` | `LeapCubeReorientDexSim-v1` | 16 | 57 / 128 | `LeapHandCube` |
| `leap_cube_rotate_z` | `LeapCubeRotateZAxisDexSim-v1` | 16 | 32 / 105 | `LeapHandCube` |

Training configurations are under
`extensions/embodichain_tasks/configs/tasks/manipulation/dexterous/<task>/agents/`.
`ppo.mujoco_warp.yaml` and `ppo.dexuni.yaml` select the corresponding solver.
The default configurations run 32 environments for ten updates and save
checkpoints. For example:

```bash
embodichain data download --name ShadowHandPen
embodichain train-rl --config extensions/embodichain_tasks/configs/tasks/manipulation/dexterous/shadow_pen_fixed/agents/ppo.mujoco_warp.yaml
```

Increase `trainer.iterations` and `trainer.num_envs` for a longer experiment.
Policy and optimizer state are saved by the standard trainer. Checkpoint
evaluation uses the generated RUN manifest through `embodichain eval-policy`.
Continue training with `embodichain train-rl --config CONFIG --resume CHECKPOINT`.
Set `trainer.iterations` to the cumulative target, for example 15 after an initial
10 updates. Resume restores policy normalization, optimizer, scheduler and update
counters, then starts fresh episodes. The shared schema-v2 loader checks task,
physics, policy and algorithm compatibility. Select a manifest checkpoint with
`--resume latest --run RUN_DIRECTORY`; `--resume-scheduler restart` starts the
configured schedule over the remaining updates while preserving optimizer
momentum. Legacy checkpoints without metadata emit a compatibility warning.
Each resumed invocation writes a new RUN directory.

The fixed and continuous pen tasks share their model. Continuous spinning
samples from 22 retained initial poses. Cube face turning retains the passive
top-layer joint and updates its detent on every physics substep. Reorientation
and Z-axis rotation use separate actor and critic observation groups.
Selective reset preserves unselected physical rows; completed rows return the
next initial observation with terminal reward and done flags.

Evaluation reports `eval/success_rate` for pen spinning, cube face turning and
cube reorientation. An episode succeeds if it reaches at least one task success
event before termination or truncation, including an event on the final step.
Pen spinning requires one complete positive revolution while retaining the pen;
face turning requires holding the target face angle for 20 steps; reorientation
requires reaching the target orientation. The environment retains these events
in `info["success"]` until the corresponding episode resets. Per-step events
remain available in `info["metrics"]["success"]`.

Continuous Z-axis rotation has no binary success criterion. Its success rate is
unavailable; evaluate `rotation`, `angular_velocity_z`, `retained`, episode length
and return instead. A numeric zero success rate for the other tasks means no
evaluated episode met its success criterion. Short training verifies execution
and checkpoint compatibility, not policy convergence.

The runtime currently contains version-specific MJCF import and compact-state
adapters for DexSim/Newton. Model audits check imported masses, inertia,
actuators, tendons and collision masks before stepping. These adapters need
engine-owned public interfaces before this task collection can be treated as
a backend-independent release. The local validation versions and publication
status must accompany a release; short-run checkpoints are not converged policies.

The face-turn task reserves 2048 constraint rows per world. A 512-row budget
can truncate constraints during large-batch training. Solver iteration limits
remain distinct from capacity limits: source defaults for LEAP tasks use
5 solver iterations and 8 line-search iterations on MJWarp; increase the
iteration budget explicitly when validating convergence.
