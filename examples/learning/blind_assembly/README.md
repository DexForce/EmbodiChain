# Blind assembly policy examples

The peg, gear and nut tasks use EmbodiChain's native managers and PPO.
`ActorCritic` composes a temporal actor with optional contact supervision;
the actor reads a proprioceptive history window and produces its own features.
Privileged state and contact labels remain training-only inputs. Current
training recipes select `actor_critic`, `actor.type: temporal` and optional named
contact supervision. Task deployments declare `rl_task`; physical components own
explicit preparation, export, startup/interval/reset events and action selection.
Entity bindings, sampling counts and measurement thresholds live in the validated
`env.extensions.assembly` namespace. Old import-registered assembly environment
IDs and flat extension fields are removed.

Assembly observations, conditions, rewards and reset behavior live in the task
extension's `assembly/functors.py`. Its packaged `assembly/defaults.yaml` owns
measurement, grasp and asset-generation defaults; deployments override them
through configuration. Shared polygon distances and URDF gravity computation
live in `embodichain.compute`, and resource snapshots in `embodichain.data`.

A pinned checkpoint, frozen physical resources and native replay evidence are
available through [the reproduction guide](REPRODUCING.md) and `model.json`.
The existing Python 3.12 / DexSim 0.6.0rc2 frozen peg model was replayed after
the core/task refactor and review fixes: **7910/8192 (96.56%)** in source and **7923/8192 (96.72%)**
with an isolated installed wheel. Both exceed the 95% development target, with
per-episode reports matching aggregate statistics. `model.json` pins the unchanged
model archive, original qualification and separate current-code replay evidence.

[Trained Peg inference video (1280×720, 25 fps)](https://raw.githubusercontent.com/DexForce/EmbodiChain/a9869001f1ad961f5496ca0c7f6aa4aeeeae6029/preview.mp4)
records the complete fixed seed sequence 30000–30002, with 3/3 successes.

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
parameters require a supporting SDK with the matching public descriptor fields;
unsupported builds fail explicitly.

## Actor export and native preview

```bash
embodichain run-example learning/blind_assembly/export_actor CHECKPOINT --config RUN/configs/train.yaml --output RUN/exports/actor.pt
embodichain run-example learning/blind_assembly/preview RUN --checkpoint best --headless --episodes 3 --seed 30000 --output outputs/assembly_preview
```

The actor export is checked against eager inference and excludes critic and
contact decoder. The preview uses the production checkpoint loader and native
DexSim rendering, captures all requested episodes, and writes recordings and
traces beneath the output directory. Native RGB capture requires the graphics
runtime and FFmpeg; unattended validation used Xvfb with `--headless`.

A checkpoint's bundled embodiment and feature convention must match training.
Changing the hand, TCP, normalization or history layout requires separate
validation; matching tensor dimensions alone is insufficient. Generated runs,
reports, learning curves and videos belong outside the source tree.
