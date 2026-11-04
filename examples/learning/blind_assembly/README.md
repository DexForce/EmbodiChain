# Blind assembly policy examples

The peg, gear and nut tasks use EmbodiChain's native managers, temporal policy
and PPO. Actor inputs contain encoder-derived features and action history;
privileged state and contact labels are training-only inputs.

## Training and evaluation

Use the registered default Rizon4S and Robotiq2F85 assets through the task
embodiment component:

```bash
embodichain train-rl --config extensions/embodichain_tasks/configs/tasks/manipulation/assembly/peg_insertion/agents/ppo.study.default_assets.dev.yaml
embodichain eval-policy RUN --checkpoint best --episodes 8192 --num-envs 1024 --episode-allocation balanced --renderer no-render
```

This development configuration uses one geometry per run and a local grasp
reset. It is not the paper's heterogeneous training distribution or identified
robot model. Formal profiles reject missing qualification prerequisites.
Historical `.nominal` profiles use an explicitly approximated gripper; select
those only when comparing that model. No success rate is implied by a config.
Choose a checkpoint on validation data before evaluating an independent seed.
See each task's configuration README for the observation and success contracts.

Generation requires the pinned mjfit provider described in
[the assembly examples](../../sim/assembly/README.md). Loading saved manifests
and bundled checkpoint resources does not require generation. Raw contact
capabilities require the SDK integration described in
[the dependency notes](../../../scripts/patches/README.md).

## Actor export and native preview

```bash
embodichain run-example learning/blind_assembly/export_actor CHECKPOINT --config RUN/configs/train.yaml --output RUN/exports/actor.pt
DISPLAY=:1 embodichain run-example learning/blind_assembly/preview RUN --checkpoint best --episodes 3 --seed 30000 --output outputs/assembly_preview
```

The actor export is checked against eager inference and excludes critic and
contact decoder. The preview uses the production checkpoint loader and native
DexSim rendering, captures all requested episodes, and writes recordings and
traces beneath the output directory. A display and FFmpeg are required.

A checkpoint's bundled embodiment and feature convention must match training.
Changing the hand, TCP, normalization or history layout requires separate
validation; matching tensor dimensions alone is insufficient. Generated runs,
reports, learning curves and videos belong outside the source tree.
