# Strict native peg policy evaluation

Fresh strict evaluation of one nominal simulation checkpoint; no retroactive training-health or full paper-reproduction claim.

## Time & Memory

| implementation | elapsed_wall_seconds | cost_time_ms | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb |
|---|---:|---|---|---|---|
| native deterministic evaluation with health guard | 146.7834343179129 | not measured | not measured | not measured | not measured |

## Success & Other Metrics

| success_rate | scored_episodes | CI95_low | CI95_high | capacity_loss | convergence_budget | native_state_all_samples_finite |
|---:|---:|---:|---:|---|---|---|
| 0.977294921875 | 8192.0 | 0.9738384900930109 | 0.9803039294853718 | False | False | True |

## Leaderboard

| rank | algorithm | overall_success_rate | qualification |
|---:|---|---:|---|
| 1 | native proprioceptive PPO checkpoint | 0.977294921875 | fresh nominal evaluation; training coverage and paper equivalence remain separate |

Status: completed. Standard evaluation report: /home/dex/.codex/worktrees/d58c/EmbodiChain/outputs/assembly_rl_attempt_20261026/final_eval/20261025T203144.902398Z-policy/evaluation.json. No task success report is produced when native health or evaluation fails.

Capacity/collision loss, unknown overflow bits or nonfinite complete native qpos/qvel/qacc matrices fail before auto-reset. States between control boundaries are not sampled. Sticky capacity flags cover all physical substeps. Prepared solver options are distinguished from the first effective timestep and first two observed control-clock intervals.

Only exact production-evaluator episode accounting is scored. The runtime factory's shape-probe reset and post-build health snapshot do not count as episodes. This fresh evaluation evidence does not retroactively qualify training. End-effector path length, action norms, actual joint torque and memory were not measured.
Scored episode records, per-arena counts and the JSONL path/SHA-256 are retained in health.json. The persisted rows are reread to verify the exact fixed quotas before success is reported.
Balanced allocation fixes each arena's scored episode quota before stepping; extra fast completions after a row's quota are unscored. Quotas and the finite control-step budget are retained in JSON. Training validation retains its original allocation protocol.
