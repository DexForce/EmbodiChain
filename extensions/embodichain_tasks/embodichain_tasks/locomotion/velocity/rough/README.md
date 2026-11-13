# G1 rough-terrain velocity task

The task controls 37 joints and exposes 310 policy observations, including 187
terrain height samples. It uses the independent `UnitreeG1Rough` robot archive
and the complete `G1RoughTerrains` curriculum. The existing 29-joint
`UnitreeG1Locomotion` asset remains unchanged.

Two runnable deployments are packaged under
`extensions/embodichain_tasks/configs/tasks/locomotion/velocity/g1_rough/`:

- `env.newton.yaml` (`agents/ppo.newton.yaml`): Newton contact generation with
  the MuJoCo Warp solver.
- `env.dexuni.yaml` (`agents/ppo.dexuni.yaml`): Newton contact generation with
  the DexUni solver.

Both deployments use 0.005 s outer physics steps, two solver substeps, four
outer steps per control interval, 100 solver iterations, and 50 line-search
iterations.

Train through the unified CLI:

```bash
embodichain train-rl --config extensions/embodichain_tasks/configs/tasks/locomotion/velocity/g1_rough/agents/ppo.newton.yaml
embodichain train-rl --config extensions/embodichain_tasks/configs/tasks/locomotion/velocity/g1_rough/agents/ppo.dexuni.yaml
```

The default smoke configuration uses 64 environments and 20 updates. Increase
these fields for longer training. Archive availability and the matching DexSim
selected-link property APIs are required. `load_rough_env_cfg` supports explicit
G1 asset bundles for reproducible terrain and initial-state checks.

The training config resolves its task config relative to the config file, so
the installed configuration also works outside the repository directory.
Contact observations read samples owned by the task's physics-step callback;
reset clears selected rows without querying the invalidated solver snapshot.

Both deployments select Newton heightfield contacts because native MuJoCo Warp
heightfield collision produces nonfinite states on this terrain.
