# Reproduce the default-asset peg policy

This delivery replays the unchanged validation-selected update-850 checkpoint
from the earlier 1000-update native PPO run. It uses the default Rizon4S and
full Robotiq2F85 assets with frozen geometry, materials and physical settings.
It does not claim a new full training run or the paper's heterogeneous task
and calibrated robot distribution.

## Runtime

Current main uses Python 3.12 and DexSim 0.6.0rc1. The synchronized tasks passed
two-environment reset, three control steps and selected-row reset checks on
that profile (default robot assets for peg, explicit nominal assets for gear
and nut). Full training and the 8192-episode success-rate evaluation have not
been repeated on the new profile.

The results below describe the pre-sync split at commit
`5401a00ce26beee81be2248629e3d993b27f618d`. Use that source revision and an
isolated Python 3.11 / Torch 2.7 environment to reconstruct its historical
measurements. The measured profile pins Torch 2.7.1+cu128, Newton 1.7.0,
MuJoCo 3.15.0, MuJoCo Warp 3.15.0 and Warp 1.17.0 on an RTX 5090.
Use an isolated environment for these pins; do not upgrade a shared training
runtime. FFmpeg and a native display are needed for preview recording.

The exact replay uses historical DexSim source revision
`528daf12fd44b9383b30165ac484cf79b574e873` with source-contact forwarding.
Prepare that SDK through its normal build/package process, using the
historical contact patch and binary hashes described in the bundle's evidence
and [archived runtime material](https://github.com/DexForce/EmbodiChain/tree/e15c945a8976b06b447f933400d12e5bd936dd3a/scripts).
The unextended `dexsim_engine==0.5.1rc1` wheel alone is insufficient for raw
contact parameters. SDK binaries are not redistributed with the model.
The earlier dev integration was tracked separately by
[Dexverse Draft MR !1562](http://192.168.3.16/Engine/dexverse/-/merge_requests/1562);
its status does not qualify a different native release. Current direct-compound
smoke checks use the installed 0.6.0rc1 package without the historical overlay.
A new SDK/solver profile requires a new evaluation and must not inherit the
historical result.

Current source and distributions do not provide dependency-overlay tooling.
Reconstruct historical measurements only with the external runtime recorded in
the bundle's provenance; its capacity modification remains part of that
historical profile rather than the current SDK installation.

## Download and replay

The adjacent `model.json` pins the downloadable archive by immutable Git commit
and SHA-256. Model binaries, frozen physical resources and raw evidence live
in a separate artifact branch; they are not merged into the feature source.
The archive retains robot/gripper attribution and contains no SDK binaries.

```bash
python - <<'PYCODE'
import hashlib, json, pathlib, urllib.request
info = json.loads(pathlib.Path("examples/learning/blind_assembly/model.json").read_text())
archive = pathlib.Path("default-assets-peg.zip")
with urllib.request.urlopen(info["url"]) as response:
    archive.write_bytes(response.read())
assert hashlib.sha256(archive.read_bytes()).hexdigest() == info["sha256"]
PYCODE
unzip default-assets-peg.zip
export RUN="$PWD/default-assets-peg"
embodichain eval-policy "$RUN" --checkpoint best --episodes 8192 --num-envs 1024 --seed 20000 --episode-allocation balanced --renderer no-render
DISPLAY=:1 embodichain run-example learning/blind_assembly/preview "$RUN" --checkpoint best --episodes 3 --seed 30000 --output outputs/assembly_preview
```

Replay loads all physical resources from the bundle and needs neither mjfit nor
the original source-model cache. Preserve its actor normalization, feature/history
convention and full hand model. Contact solves need not be bitwise identical;
compare balanced completed-episode statistics with the recorded confidence
intervals. Solver iteration-budget warnings and SDK teardown entity-accounting
warnings remain in the evidence and prevent claiming precision qualification.

## Training

To start a fresh random-initialization run against the frozen physical assets:

```bash
embodichain train-rl --config "$RUN/configs/train.yaml"
```

This configuration retains 4096 arenas, 32 controls per rollout and 1000 updates.
The new-run source recipe is
`extensions/embodichain_tasks/configs/tasks/manipulation/assembly/peg_insertion/agents/ppo.study.default_assets.dev.yaml`;
it composes the bare arm and hand in the application embodiment. Generation of
new manifests needs the pinned mjfit provider described by the simulation examples.

The split was checked with a separate 16-arena, 8-control, 2-update random-start
run, including checkpoint/resource saving. This checks the training path;
it is not a replacement for a full training experiment. Formal study/full-range
profiles retain their qualification gates. The 95% success target is unmet.

## Recorded results

The machine-readable descriptor and bundled `evidence/split_replay_20261105/`
contain source and installed-wheel results, native preview traces, logs,
checkpoint/resource identities and runtime provenance. The wheel check installs
the built distribution in a separate system-site-packages environment outside
the source checkout; it reuses Torch and the explicitly selected SDK profile.

| Replay | Episodes / arenas | Success | 95% Wilson interval |
|---|---:|---:|---:|
| Source checkout | 8192 / 1024 | 7275 (88.8062%) | 88.1051–89.4708% |
| Installed wheel | 8192 / 1024 | 7288 (88.9648%) | 88.2680–89.6252% |

Both use seed 20000 and eight scored completions per arena. The historical
selected checkpoint scored 7240/8192 (88.3789%) on its original implementation.
The small 64-arena smoke results (source 46/64, wheel 42/64) are retained in the
bundle; they do not replace either full-protocol result. The native preview is
3/3 on its predeclared seeds and is visual evidence, not a success-rate estimate.
