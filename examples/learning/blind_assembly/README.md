# Native proprioceptive assembly

The three registered tasks use native Gym managers, the temporal actor-critic,
the native collector/rollout/PPO, and checkpoint evaluation. Actor input is
30 previous frames plus the current frame; contact labels and privileged state
are separate training inputs. The paper's unavailable temporal architecture
details use the documented engineering defaults.

Install the optional assembly generator and the SDK capabilities described in
`scripts/patches/README.md` in a source checkout. Wheels include the patch files
under the installation prefix's `share/embodichain/assembly_patches/`.
The public robot sources and both licenses are pinned by Git blob hashes.

```bash
embodichain run-example learning/blind_assembly/run_suite --phase prepare --suite-dir outputs/assembly_suite --nominal
embodichain run-example learning/blind_assembly/run_suite --phase train --suite-dir outputs/assembly_suite --nominal
embodichain run-example learning/blind_assembly/run_suite --phase evaluate --suite-dir outputs/assembly_suite --nominal
```

Preparation writes 54 trials: three tasks, study/full-range profiles, three
seeds and full/no-history/no-contact-supervision variants. Defaults preserve
12,288 environments and 5,000 updates, with validation every 100 updates.
Preparation does not run those experiments. Evaluation requires an actual
validation-selected best checkpoint; it cannot silently fall back to latest.
OOD axes remain pending until separately held-out geometry configurations are
provided. The suite never substitutes training geometry for an OOD evaluation.

Formal profiles fail before scene construction when arbitrary-topology
heterogeneous assets or qualified source robot dynamics are unavailable.
`--nominal` explicitly selects the development approximation: the Rizon4S arm,
parallel-jaw carriages carrying source finger mass and pad geometry, encoder
gravity feedforward, and one geometry per run. This retains physical gripping
without a weld, but does not reproduce the Robotiq four-bar/tendon dynamics,
the author's identified model, workspace-wide reset distribution or geometry
training distribution. Nut grasp stability is not fully qualified.

Native checkpoint snapshots freeze component paths. These tasks additionally
bundle the exact assembly manifest, prepared robot meshes/URDF, fidelity record
and source licenses. The relative resources can be replayed after relocating
the run, without the generator or original asset cache.

```bash
embodichain eval-policy RUN --checkpoint latest --episodes 2 --num-envs 2 --renderer no-render
embodichain run-example learning/blind_assembly/export_actor CHECKPOINT --config RUN/configs/train.yaml --output RUN/exports/actor.pt
```

The actor export is checked against eager inference for multiple batch sizes.
It contains normalization and deterministic action means; actuator scaling and
limits remain owned by the task/controller.

See [CAPACITY.md](CAPACITY.md) for measured synthetic 12,288-environment policy
capacity, and [the implementation report](../../sim/assembly/IMPLEMENTATION.md)
for actual task smoke checkpoints, evaluation results and outstanding gates.
Scientific precision/performance runners under `scripts/benchmark/` require a
source checkout.
