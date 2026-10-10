# G1 PPO CUDA benchmark

Measure EmbodiChain PPO on **Dexsim Default CUDA** and **Dexsim Newton/MJWarp CUDA**.
The official G1 task and PPO configuration supply the environment, policy, and optimizer.

Run each backend in a separate process with a new output directory:

```bash
for backend in default newton; do
  python -m scripts.benchmark.rl.locomotion_ppo \
    --backend "$backend" --output "outputs/g1-ppo/$backend"
done
```

The default workload uses 4096 environments, 24 steps per rollout, 3 warmup updates,
and 10 measured updates. Use `--envs 32 --warmup 1 --updates 3` for a smoke run.
`--task-dir` selects another G1 configuration directory; `--seed` and `--cpu-threads`
control the training seed and PyTorch CPU thread count.
Newton uses 20 solver iterations (`--solver-iterations`) and the task's 50 line-search
iterations. The selected environment configuration is saved as `env.yaml`.

`result.json` contains synchronized rollout/optimizer timings, losses, backend and
runtime details, environment/policy creation time, and memory samples. PPO SPS is
total measured transitions divided by total rollout plus optimizer time. Creation
is timed separately; warmup, validation, memory sampling, and checkpoint writes
are excluded from PPO timings. `report.md` summarizes the run;
`model.pt` contains the policy, observation normalizers, optimizer, and trainer counters.
The Markdown report includes creation/rollout/update times, throughput and its
per-update coefficient of variation (CV), process RAM/VRAM peaks, and GPU allocator
memory. CV is the population standard deviation divided by the mean of per-update
SPS. Process VRAM is sampled at phase boundaries; allocator peaks are measured
inside each timed phase.
Process VRAM matches one worker PID in NVIDIA's host PID namespace. A worker in
a nested PID namespace requires readable host procfs to resolve that PID;
private or inaccessible procfs keeps the sample unknown and reports `n/a`.

Run G1/Go2 reset and inference checks, then G1 training/checkpoint tests on both backends:

```bash
pytest tests/gym/envs/tasks/test_locomotion_cuda.py --run-gpu -v --junitxml=outputs/locomotion-tests.xml
```
