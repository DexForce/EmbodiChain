# Reproduce the configuration-driven peg policy

The current delivery uses `ActorCritic` with `TemporalActor` (`official_v1`) and
optional `ContactPrediction`. The task registers the common `EmbodiedEnv` through
its root `rl_task` declaration. Preparation, actions, observations, rewards,
startup/interval/reset events and resource export are explicitly configured.
Historical policy constructors, previous-history layouts and checkpoint migration
paths have been removed.

## Qualified runtime and selection

Python **3.12.14**, Torch **2.11.0+cu128**, DexSim **0.6.0rc2**, Newton **1.7.0**
and MuJoCo/MuJoCo-Warp **3.15.0**, on an RTX PRO5000 72GB Blackwell.
The evidence records exact package versions and installation RECORD hashes;
all 1547 checked dependency code/binary files matched their installation records.
No dependency modification or SDK binary is shipped in the model archive.

Training started from random weights with seed **0**, **4096 arenas** and
**32 control transitions per rollout**. The maximum budget was 1000 updates.
Update **300** was frozen for early stopping after balanced validation on seed
**11000** scored **1988/2048 (97.0703%)**, Wilson 95% **96.2472–97.7172%**.
The candidate checkpoint and its SHA-256 were recorded before test episodes ran;
later updates were discarded. The test set did not select the model.

The explicit file-backed appearance path preserves the physical collision pieces,
source mass/COM/inertia and contact parameters while reducing material-reset
work. MuJoCo-Warp startup gains use the shared batched physics view with unchanged
configured values. Complete current-schema optimizer/normalizer/RNG continuation
was verified; it does not provide historical checkpoint migration.

## Download and replay

The adjacent `model.json` pins the current archive to an immutable artifact commit
and SHA-256. It contains the selected complete checkpoint, portable configuration,
frozen robot and assembly resources, attribution and raw evidence.

```bash
python - <<'PYCODE'
import hashlib, json, pathlib, urllib.request
info = json.loads(pathlib.Path("examples/learning/blind_assembly/model.json").read_text())
archive = pathlib.Path("configured-peg.zip")
with urllib.request.urlopen(info["url"]) as response:
    archive.write_bytes(response.read())
assert hashlib.sha256(archive.read_bytes()).hexdigest() == info["sha256"]
PYCODE
unzip configured-peg.zip
export RUN="$PWD/configured-peg"
embodichain eval-policy "$RUN" --checkpoint best --episodes 8192 --num-envs 1024 --seed 20000 --episode-allocation balanced --renderer no-render
```

Replay needs the installed current code and declared SDK dependencies. It loads
physical resources from the archive and does not need mjfit or the original asset
cache. A source-free installed-wheel smoke with an empty asset cache and the
generator import blocked completed **62/64** episodes; that smoke is separate from
the full evaluations below. GPU physics is not bitwise deterministic.

## Independent evaluation

Both executions use the same frozen checkpoint, deterministic action means,
seed **20000**, **1024 arenas**, and **8 scored completions per arena**:

| Execution | Success | Wilson 95% interval |
|---|---:|---:|
| Source checkout | **7930/8192 (96.8018%)** | 96.3983–97.1614% |
| Isolated installed wheel | **7933/8192 (96.8384%)** | 96.4370–97.1959% |

Both exceed the **95% engineering target**. The installed wheel was checked
against the current production sources and imported its three main packages from
its own environment. The raw source and wheel reports are retained separately;
no preferred replay result was substituted for the other. Earlier executions of
the same frozen model, before the other profiles' sensor-capacity correction,
scored 7942/8192 in source and 7929/8192 in the installed wheel. Those reports
are also retained, with their separate code revisions.

## Fresh training, export and RGB preview

```bash
embodichain train-rl --config "$RUN/configs/train.yaml"
embodichain run-example learning/blind_assembly/export_actor "$RUN/checkpoints/configured_current_peg_insertion_seed0_best_step_39321600.pt" --config "$RUN/configs/train.yaml" --output "$RUN/exports/actor.pt"
embodichain run-example learning/blind_assembly/preview "$RUN" --checkpoint best --headless --episodes 3 --seed 30000 --output outputs/assembly_preview
```

The saved training config reuses the frozen geometry and tested optimizer settings
with new random weights. The task-supplied
`peg_insertion/agents/ppo.study.default_assets.dev.yaml` is also available as a
fresh-generation recipe; its evaluation cadence differs from the recorded run.

Fresh training and frozen-checkpoint replay are distinct experiments; a new run
is not guaranteed to reproduce the same checkpoint or success rate. Validation
must remain separate from the final test seed. Generation of new geometry needs
the pinned provider described in the simulation examples.

The exported actor excludes critic/contact heads and matched eager inference with
trace error **0**. The current native RGB preview records the complete predeclared
seed sequence **30000, 30001, 30002**, with **3/3** successes and the original grip,
penetration and stable-seat guards. Headless recording uses an actual native
camera and finishes encoding before native cleanup. Omit `--headless` for a window.
The three-episode preview is visual evidence, not a success-rate estimate.

## Qualification boundary

This is a single-geometry, single-training-seed development profile with the
public default arm/full mimic hand. It does not qualify the paper's heterogeneous
geometry distribution, identified hardware dynamics, other-task learning or real
device performance. Formal study/full-range deployments retain their explicit
qualification gates. Solver iteration-budget warnings remain in the evidence;
the success result is not a contact-precision certification. Gear and nut
configurations retain lifecycle coverage; their learning performance is outside
this Peg qualification.
