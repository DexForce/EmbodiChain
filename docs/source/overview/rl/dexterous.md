# Pen and cube manipulation

Five USD-native dexterous manipulation tasks inherit from `EmbodiedEnv` (`DexterousManipulationEnv`) and integrate with the task catalog, `run-task`, `train-rl`, and `eval-policy`. Each task loads its physical scene directly from packaged USD (`.usdc`) via `SimulationManager.add_usd()` with zero runtime MJCF dependency.

| Task directory | Registered environment | Actions | Actor / critic observations | Asset |
|---|---|---:|---:|---|
| `shadow_pen_fixed` | `ShadowPenFixedRL-v1` | 18 | 87 / 87 | `ShadowHandPen` |
| `shadow_pen_continuous` | `ShadowPenContinuousRL-v1` | 18 | 87 / 87 | `ShadowHandPen` |
| `leap_cube_face_turn` | `LeapCubeFaceTurnRL-v1` | 16 | 65 / 65 | `LeapHandCubeFaceTurn` |
| `leap_cube_reorient` | `LeapCubeReorientRL-v1` | 16 | 57 / 128 | `LeapHandCube` |
| `leap_cube_rotate_z` | `LeapCubeRotateZAxisRL-v1` | 16 | 32 / 105 | `LeapHandCube` |

## Configuration and CLI Usage

Each task directory under `extensions/embodichain_tasks/configs/tasks/manipulation/dexterous/<task>/` provides a single `default` catalog deployment (`env.yaml` and `agents/ppo.yaml`) configured with `physics: newton` and `solver_type: dexuni` (`use_mujoco_cpu: false`, `use_mujoco_contacts: true`, GPU MuJoCo Warp on CUDA with full scene joint ownership).

```bash
embodichain data download --name ShadowHandPen
embodichain show-task embodichain_tasks:shadow_pen_fixed
embodichain run-task --gym_config extensions/embodichain_tasks/configs/tasks/manipulation/dexterous/shadow_pen_fixed/env.yaml
embodichain train-rl --config extensions/embodichain_tasks/configs/tasks/manipulation/dexterous/shadow_pen_fixed/agents/ppo.yaml
```

Continue training from a saved checkpoint with `embodichain train-rl --config CONFIG --resume latest --run RUN_DIRECTORY`, setting `trainer.iterations` to the cumulative update target. Evaluate a saved checkpoint via `embodichain eval-policy RUN_DIRECTORY --checkpoint latest --num-envs 64 --episodes 64 --renderer no-render`.

## Task Semantics and Evaluation

Fixed and continuous pen-spinning share the `ShadowHandPen` USD scene; continuous pen-spinning samples across 22 initial grasp poses. Cube face-turning (`LeapCubeFaceTurnRL-v1`) models the cube as a single-hinge articulation (`face_turn`) and applies passive detent torque before each physics substep. Cube reorientation (`LeapCubeReorientRL-v1`) and Z-axis rotation (`LeapCubeRotateZAxisRL-v1`) construct asymmetric actor and privileged critic observation groups from public `Robot` and `RigidObject` state, site transforms parsed from USD, and pre-integration PD actuator forces.

Evaluation reports `eval/success_rate` for `ShadowPenFixedRL-v1`, `ShadowPenContinuousRL-v1`, `LeapCubeFaceTurnRL-v1`, and `LeapCubeReorientRL-v1`: an episode counts as successful if at least one task success event occurs before termination or truncation (`info["success"]`), while per-step pulses remain in `info["metrics"]["success"]`. `LeapCubeRotateZAxisRL-v1` optimizes unbounded angular velocity without a binary success threshold; evaluate `rotation`, `angular_velocity_z`, `retained`, episode length, and return instead.

## USD Physical Fidelity and Solver Settings

All five environments import fixed tendons, joint limits, actuator `ctrlrange`, PD `target_ke`/`target_kd`, site local poses, and collision filter pairs directly from `.usdc`.

`LeapCubeFaceTurnRL-v1` reserves `njmax: 2048` constraint rows per world (`512` for the remaining tasks). Each `env.yaml` preserves authored solver settings (`20/50` Newton/line-search iterations on Shadow Hand tasks; `5/8` on LEAP Hand tasks) and sets `arena_space: 0.0` with `separate_worlds: true` so execution-model geometry stays spatially uniform across parallel worlds.

## DexUni Training and Evaluation Baseline

Sustained single-GPU PPO training, `--resume latest` continuation (`+10` updates), and `64`-episode `eval-policy` results under `env.yaml` (`physics: newton`, `solver_type: dexuni`, `4096` parallel environments, single NVIDIA RTX PRO 5000 Blackwell 72 GB GPU):

| Environment | Updates (`train + resume`) | Transitions | Steady SPS | Train `episode_reward_avg_100` (`start -> end`) | `eval-policy` (`64` episodes, `--checkpoint latest`) |
|---|---:|---:|---:|---|---|
| `ShadowPenFixedRL-v1` | `150 + 10` | `26,214,400` | `54,106` | `-50.49 -> +47.94` | `avg_reward=+74.26`, `avg_length=500.0`, `retained=100%`, `error=0.143 rad`, `rotation=2.40 rad` |
| `ShadowPenContinuousRL-v1` | `150 + 10` | `52,428,800` | `57,265` | `-39.46 -> +14.47` | `avg_reward=+9.64`, `avg_length=485.8`, `retained=96.9%`, `error=0.229 rad`, `rotation=0.82 rad` |
| `LeapCubeFaceTurnRL-v1` | `150 + 10` | `26,214,400` | `55,562` | `-0.47 -> +216.71` | `avg_reward=+255.53`, `avg_length=500.0`, `success_rate=100%`, `success_count=12.52`, `rotation=20.56 rad` |
| `LeapCubeReorientRL-v1` | `300 + 10` | `50,790,400` | `48,218` | `-4.58 -> +172.64` | `avg_reward=+180.98`, `avg_length=997.6`, `retained=98.4%`, `success_rate=3.1%`, `position_error=0.020 m` |
| `LeapCubeRotateZAxisRL-v1` | `300 + 10` | `50,790,400` | `94,216` | `-5.56 -> +27.40` | `avg_reward=+29.35`, `avg_length=497.5`, `retained=98.4%`, `rotation=29.43 rad`, `angular_velocity_z=1.39 rad/s` |
