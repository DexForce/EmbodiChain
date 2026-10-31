# Configured manipulation RL comparison

Run `compare_push_cube.py --gpus 3 4 --output output/configured_rl_comparison`
from the repository root, choosing available GPUs. It schedules six independent
PPO experiments with the fixed 64 × 1024 × 1000 budget, seeds 42/43/44 and
legacy/configured PushCube variants. Each worker reloads its final checkpoint
and compares it with the untrained policy on 100 identical initial cases.

For a smoke check, add `--iterations 2 --cases 3` and use a separate output
directory. Reduced runs never qualify as formal convergence evidence. Existing
results are reused only if their identity, budget, complete cases and
configuration/implementation fingerprints match. The runner never increases
training budgets automatically.

The full protocol and condition configuration are described in
`docs/source/guides/configured_manipulation_rl.md`.
