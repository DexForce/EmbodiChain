# Rebuild and audit the retained results

Use the py311 environment from the demo README, from the repository root.
The original run root is `/tmp/ods-visual-budget-prefetch-20261022`.
The implementation is frozen at `915510b58`; later changes only add evidence
and documentation. Re-run collection/training into a new directory, never over
the measured corpus.

```bash
conda activate py311

python -m examples.ods.data_stream_demo.train --suite calibrate \
  --producer-envs 2 --evaluation-scenes 0 --gpu-ids 3 4 5 7 \
  --aux-graphics-gpu-id 7 --output outputs/ods-capacity-new/p2
python -m examples.ods.data_stream_demo.train --suite calibrate \
  --producer-envs 4 --evaluation-scenes 0 --gpu-ids 3 4 5 7 \
  --aux-graphics-gpu-id 7 --output outputs/ods-capacity-new/p4
python -m examples.ods.data_stream_demo.train --suite calibrate \
  --producer-envs 8 --evaluation-scenes 0 --gpu-ids 3 4 5 7 \
  --aux-graphics-gpu-id 7 --output outputs/ods-capacity-new/p8
python -m examples.ods.data_stream_demo.training_throughput \
  outputs/ods-capacity-new/p2 outputs/ods-capacity-new/p4 \
  outputs/ods-capacity-new/p8 --output outputs/ods-capacity-new/selection

python -m examples.ods.data_stream_demo.train --suite pilot \
  --producer-envs 4 --updates 128 --batch-size 32 --buffer-size 8 \
  --episode-budget 128 --evaluation-scenes 0 --gpu-ids 3 4 5 7 \
  --aux-graphics-gpu-id 7 --output outputs/ods-pilot-new
```

Physical IDs above describe the recorded host, not a device reservation.
Admit only currently idle devices with sufficient free memory. Two primary
GPUs can run the vision groups sequentially; the recorded cohort uses two
fixed pairs concurrently. Select P from the new measurements; P4 is the
recorded common setting, not a portable optimum. Clear `DISPLAY` and
`WAYLAND_DISPLAY` and use `SDL_VIDEODRIVER=dummy` for the qualified headless
runtime. This host additionally uses the existing `/tmp/ods-dexsim-runtime`
compatibility package in `PYTHONPATH`, after the repo and `extensions/`.
No packages or native binaries were replaced during this study.

To regenerate figures and the primary report without simulation, set
`ODS_BUNDLE` to this directory and choose fresh scratch outputs:

```bash
python "$ODS_BUNDLE/analyze.py" "$ODS_BUNDLE/inputs" \
  --output /tmp/ods-prefetch-comparison-new
python -m examples.ods.data_stream_demo.training_report \
  "$ODS_BUNDLE/inputs/pilot/main" --output /tmp/ods-prefetch-report-new
```

`inputs/pilot/main` contains only the six primary outcomes. Eager controls stay
under `inputs/pilot/controls/eager`. The compact `suite-overview.json` is
explicitly derived; the original large suite/report JSON lives in the archive.
`report-input-files.json` identifies the 194 byte-identical original report
inputs. `reproduction-check.json` records matching qualification, paired
results, aggregates and resource aggregates after relocation.

`raw-logs.tar.gz` contains all original non-corpus text logs for the three
calibrations and pilot, plus the four supervisor CLI logs. Every one of its
380 members was decompressed and SHA-checked against `raw-log-files.json`.
Extract into a separate scratch directory to recompute event-based coverage.
The corpus/checkpoint files are external; `external-artifacts.json` records
their exact original paths, sizes and hashes. Losing those paths prevents a
new checkpoint/video-file audit but does not replace or alter the retained
measurements. `independent-audit.json` records the completed audit.

The `.py.source` files under `qualification/` preserve the exact temporary
CPU drivers as evidence. Python can execute that suffix directly. They
reference the original worktree for this host's qualification. The v2
independent driver reuses the earlier study's trusted event reconstruction,
checkpoint hash and full-video evidence functions. It accepts `--root`,
`--pilot-root`, `--protocol`, `--execute` and a fresh `--output` outside the
input tree. It checks all actual suite source entries against the 32-file
protocol union; two target/observation helpers have snapshot/current-file
checks only. Original tool self-check failures and the rejected v1 receipt
remain labeled under `qualification/`.

No new production API or example test framework was introduced. The private
prefetch regression uses actual extracted methods and the real sampler;
capacity/report fixtures check independent known arithmetic, selection
boundaries, strict pilot isolation and unchanged historical 29/23 counts.
The real 14-run cohort owns model/data/closure integration validation. The
unchanged production/model contracts reuse their existing focused coverage.
