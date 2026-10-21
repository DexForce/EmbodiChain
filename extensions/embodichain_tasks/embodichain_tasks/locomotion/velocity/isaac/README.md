# G1 rough-terrain velocity task

The task controls 37 joints and exposes 310 policy observations, including 187
terrain height samples. It uses the independent `UnitreeG1Rough` robot archive
and the complete `G1RoughTerrains` curriculum. The existing 29-joint
`UnitreeG1Locomotion` asset remains unchanged.

The runnable deployment is
`extensions/embodichain_tasks/configs/tasks/locomotion/velocity/g1_isaac_rough/env.newton.yaml`.
It explicitly selects Newton contact generation with the MuJoCo Warp solver,
0.005 s outer physics steps, two solver substeps and four outer steps per
control interval. It retains the reference task's 100 solver iterations and
50 line-search iterations.

Train through the unified CLI:

```bash
embodichain train-rl --config extensions/embodichain_tasks/configs/tasks/locomotion/velocity/g1_isaac_rough/agents/ppo.newton.yaml
```

The default smoke configuration uses 64 environments and 20 updates. Increase
these fields for longer training. Archive availability and the matching
DexSim selected-link property APIs are required. The package loader supports
explicit G1 reference bundles for reproducible terrain and initial-state checks.
`ReferenceRslRlEnv` retains the previously validated RSL-RL interface.

The native MuJoCo Warp heightfield collision path remains unsuitable for this
terrain because the retained official-runtime reproducer produces nonfinite
states. The default deployment therefore selects Newton contacts. DexUni's
G1 external-contact path still requires the separately retained dispatch
adapter; it is not enabled by this deployment.
