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
The core training CLI does not currently expose an optimizer-resume option;
programmatic continuation must restore policy, optimizer and trainer counters.

The fixed and continuous pen tasks share their model. Continuous spinning
samples from 22 retained initial poses. Cube face turning retains the passive
top-layer joint and updates its detent on every physics substep. Reorientation
and Z-axis rotation use separate actor and critic observation groups.
Selective reset preserves unselected physical rows; completed rows return the
next initial observation with terminal reward and done flags.

The runtime currently contains version-specific MJCF import and compact-state
adapters for DexSim/Newton. Model audits check imported masses, inertia,
actuators, tendons and collision masks before stepping. These adapters need
engine-owned public interfaces before this task collection can be treated as
a backend-independent release. The local validation versions and publication
status must accompany a release; short-run checkpoints are not converged policies.
