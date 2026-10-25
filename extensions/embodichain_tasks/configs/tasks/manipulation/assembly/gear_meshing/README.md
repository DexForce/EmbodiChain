# Blind gear meshing

Native simulator task with the EmbodiChain PPO pipeline. Formal study/fullrange
profiles fail closed while heterogeneous per-arena geometry and a qualified
Rizon4S/Robotiq model are unavailable. Explicit `.nominal` deployments permit
the public uncalibrated arm and explicit parallel-jaw approximation with source pad geometry and inertias for development.
They have one geometry per run and a local contact reset, not the paper's
workspace-wide and per-world geometry distribution. No success rate is implied. The original four-bar mimic preset remains separately selectable; nominal carriages omit closed-loop linkage dynamics.

```bash
embodichain train-rl --config extensions/embodichain_tasks/configs/tasks/manipulation/assembly/gear_meshing/agents/ppo.study.nominal.yaml
```

For a smoke run, override trainer `num_envs: 1`, `iterations: 1`, `buffer_size: 8`,
`enable_eval: false`. The full nominal template retains the intended batch budget
so capacity failures are visible. Do not label a reduced batch as paper reproduction.

The actor receives 33 encoder-derived values per frame, 30 historical frames
and the current frame. The critic sees actual mating pose, geometric progress
and one-hot contact classes; true contact classes are separate training fields.
Reward milestones are paid once per episode. Success requires retained contact
on both physical pads and bounded penetration. No fixed grasp constraint is used.

Sources: [paper and appendix](https://arxiv.org/html/2610.23849v1),
[mjfit](https://github.com/younghyopark/mjfit/tree/5833f465896e1e06bc4be4f003be847f5183a4db),
[public robot models](https://github.com/google-deepmind/mujoco_menagerie/tree/f3b6013629c7b94f8ee93fd977b161fa8350f9bc).
The contact class thresholds and undocumented milestone details are engineering
choices. Gravity feedforward uses the known URDF COM potential gradient from
encoders at 50 Hz; the native impedance drive runs at 1 kHz. Formal robot and heterogeneous geometry gates prevent treating this
development task as author-code equivalence or qualified sim-to-real training.
