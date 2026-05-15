# DexSim 0.3.11 Agent Atomic Action Reproduction Guide

This guide reproduces the current Agent atomic-action migration validation on
the local DexSim 0.3.11 runtime. The branch is prepared for DexSim 0.4.0
compatibility, but the validation results below were produced with DexSim
0.3.11. Use this guide to verify the public atomic-action interfaces, Agent
adapter path, graph recovery runtime, videos, and output artifacts before
deciding which parts still need DexSim 0.4.0-specific changes.

## Scope

Validated branch:

```bash
git switch ljd/agentchoard_dexsim040_migration
```

Main changes covered by this reproduction:

- Public atomic-action functional API:
  `move`, `pick_up`, `place`, `gripper_open`, and `gripper_close`.
- Agent atomic-skill adapter path for public move, place, gripper, and grasp
  execution.
- Graph recovery runtime with monitor-triggered recovery branches.
- DexSim renderer compatibility layer that keeps DexSim 0.3.11 on raster
  rendering while preparing DexSim 0.4.0 `hybrid` configuration.
- Two gym validation demos:
  `grasp_cup.py` and `pour_water_recovery_compare.py`.

## Environment

Run all commands from the repository root:

```bash
cd "/home/dex/desktop/EmbodiChain/dexsim040/agentchoard"
```

If the local path contains Chinese characters, use the actual path:

```bash
cd "/home/dex/桌面/EmbodiChain/dexsim040/agentchoard"
```

Use the existing conda environment:

```bash
conda run --no-capture-output -n embodichain python - <<'PY'
import dexsim
print(dexsim.__version__)
PY
```

Expected output:

```text
0.3.11
```

Important notes:

- The project metadata currently targets `dexsim_engine==0.4.0` for the
  migration branch. Do not recreate the environment from scratch if the goal is
  to reproduce the current 0.3.11 validation exactly.
- Keep the default raster rendering path for 0.3.11. Do not pass `--enable_rt`
  unless you are explicitly testing renderer changes.
- Use `conda run --no-capture-output -n embodichain ...`; running bare
  `python` may fail with missing packages such as `gymnasium`.

## Static Checks and Unit Tests

Run:

```bash
git diff --check

conda run --no-capture-output -n embodichain pytest -q \
  tests/sim/agent \
  tests/sim/atomic_actions \
  tests/sim/utility/test_solver_utils.py
```

Expected result from the validated 0.3.11 run:

```text
103 passed
```

Warnings from third-party packages are acceptable if the tests pass.

## Required Cached Agent Artifacts

`scripts/tutorials/gym/pour_water_recovery_compare.py` is designed to run
offline by copying pre-generated graph artifacts from:

```text
outputs/agent_repro_compare
```

This avoids requiring reviewers to call an LLM during deterministic demo
validation.

Before running the pour-water matrix, check that these files exist:

```bash
test -f "outputs/agent_repro_compare/single_normal/artifacts/agent_task_graph.json"
test -f "outputs/agent_repro_compare/single_recovery/artifacts/agent_recovery_spec.json"
test -f "outputs/agent_repro_compare/dual_normal/artifacts/agent_task_graph.json"
test -f "outputs/agent_repro_compare/dual_recovery/artifacts/agent_recovery_spec.json"
```

If the cache is missing, copy the `outputs/agent_repro_compare` artifact bundle
from the validation machine, or regenerate artifacts with a configured LLM
environment. The default compare runner installs offline LLM stubs and expects
cached artifacts for normal reproduction.

## Demo 1: Agent Skill Pick-Place

Run:

```bash
TS="$(date +%Y%m%d_%H%M)"
OUT="outputs/grasp_cup_repro_${TS}"

conda run --no-capture-output -n embodichain python \
  "scripts/tutorials/gym/grasp_cup.py" \
  --output_root "${OUT}" \
  2>&1 | tee "${OUT}.runner.log"
```

Expected outputs:

```text
${OUT}/summary.tsv
${OUT}/outputs/videos/episode_0_cam1.mp4
${OUT}.runner.log
```

Expected behavior:

- The camera view should show the table, robot arm, and mug clearly.
- The Agent skill sequence should complete the pick, lift, move, place, and
  return steps.
- `summary.tsv` should mark the run as successful.

If the video is black, clipped, or from an unexpected angle, debug the DexSim
renderer/camera path before debugging the Agent action logic.

## Demo 2: Pour-Water Recovery Matrix

Run:

```bash
TS="$(date +%Y%m%d_%H%M)"
OUT="outputs/pour_water_recovery_compare_repro_${TS}"

conda run --no-capture-output -n embodichain python \
  "scripts/tutorials/gym/pour_water_recovery_compare.py" \
  --case all \
  --continue_on_case_failure \
  --output_root "${OUT}" \
  2>&1 | tee "${OUT}.runner.log"
```

Expected outputs:

```text
${OUT}/logs/summary.tsv
${OUT}/report.md
${OUT}/<case>/artifacts/case_result.json
${OUT}/<case>/outputs/videos/episode_0_cam1.mp4
${OUT}.runner.log
```

Expected cases:

```text
single_clean_no_recovery
single_error_no_recovery
single_error_blind_no_recovery
single_clean_with_recovery
single_error_with_recovery
dual_clean_no_recovery
dual_error_no_recovery
dual_error_blind_no_recovery
dual_clean_with_recovery
dual_error_with_recovery
```

Expected `summary.tsv` behavior from the validated 0.3.11 run:

- All cases should have `expectation_matched=True`.
- Clean cases should have `program_success=True` and `semantic_success=True`.
- Error cases without recovery should stop after monitor trigger, so
  `program_success=False` is expected.
- Error cases with recovery should have `program_success=True` and
  `semantic_success=True`.

The default deterministic error is a horizontal bottle translation:

```text
--error_injection_type misplaced_object
--error_injection_offset 0.12,0.0,0.0
```

It should not intentionally topple the bottle. Use `--error_injection_type
fallen_object` only for fallen-object stress tests.

## Optional Keyframe Check

Use this helper to sample representative video frames:

```bash
conda run --no-capture-output -n embodichain python - <<'PY'
import cv2
from pathlib import Path

root = Path("outputs/pour_water_recovery_compare_repro_YYYYMMDD_HHMM")
out_dir = root / "keyframes_check"
out_dir.mkdir(parents=True, exist_ok=True)

for case in [
    "single_error_no_recovery",
    "single_error_with_recovery",
    "dual_error_blind_no_recovery",
    "dual_error_with_recovery",
]:
    video = root / case / "outputs/videos/episode_0_cam1.mp4"
    cap = cv2.VideoCapture(str(video))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    for label, idx in [
        ("mid", frames // 2),
        ("late", int(frames * 0.8)),
        ("final", max(frames - 1, 0)),
    ]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            cv2.imwrite(str(out_dir / f"{case}_{label}_{idx:04d}.jpg"), frame)
    cap.release()
PY
```

Replace `pour_water_recovery_compare_repro_YYYYMMDD_HHMM` with the actual output
directory name.

Visual acceptance criteria:

- Bottle and cup remain visible in the camera frame.
- Default error injection translates the bottle rather than toppling it.
- Recovery cases should resume and finish the pour-water sequence.
- No-recovery error cases should record the failure state long enough for
  inspection.

## Common Failures

`ModuleNotFoundError: No module named 'gymnasium'`
: Use `conda run --no-capture-output -n embodichain ...` instead of bare
  `python`.

Missing cached artifacts
: Copy `outputs/agent_repro_compare` from the validation machine, or configure
  the LLM environment and regenerate artifacts explicitly.

Unexpected video angle or blank video
: First verify DexSim renderer/camera compatibility. On DexSim 0.3.11, keep the
  default raster renderer. Do not use the DexSim 0.4.0 hybrid renderer path for
  this reproduction.

Different physical outcome under DexSim 0.4.0
: Treat this as a DexSim 0.4.0 migration issue rather than a failure to
  reproduce the 0.3.11 baseline. Re-run the same commands after installing
  DexSim 0.4.0 and compare videos, `summary.tsv`, and `case_result.json`.

## PR Commit Message Format

Use one of these title prefixes:

```text
NEW:
ENH:
FIX:
OTH:
```

The commit body should follow this layout:

```text
WHY: <why this change is needed>

URL: <requirement or bug link, or N/A>

TEST: <what was tested; keep this specific and longer than a short placeholder>
```

For this migration, a suitable PR title is:

```text
ENH: Add dexsim040 agent atomic action migration
```

The PR description should explicitly say that the branch is prepared for
DexSim 0.4.0 compatibility, while the reproduction baseline in this guide uses
DexSim 0.3.11.
