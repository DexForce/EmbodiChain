# Task Engine route benchmarks

These repository tools evaluate authoring and publication separately from scene
qualification, compilation and physical acceptance. Run as modules from the
repository root. Complete deployment examples live in
[examples/gensim/goal_authoring](../../../examples/gensim/goal_authoring/README.md).

## Same-budget CLI comparison

`CACHE` contains original task1171–1190 PNG files and `task_descriptions.json`
from dataset snapshot `569f1860815d256ad87f53f31e19a71363c8afb9`.
Set `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL`, or the project's model
environment configuration. `OUT` must be a new directory. Only explicit template
adds `--route`; the default branch receives just image, instruction and output
identity. The meter caps actual HTTP requests at five per case and route,
including repair and later native-scene matching, with no transport retry.

```bash
export EMBODICHAIN_MODEL_SERVICES_CONFIG="$PWD/examples/gensim/goal_authoring/services.json"
python -m scripts.benchmark.task_engine.default_cohort "$CACHE" "$OUT" --workers 10
python -m scripts.benchmark.task_engine.default_cohort "$CACHE" "$OUT" --assess --fruit-installation "$INSTALLATION"
```

`INSTALLATION` is a complete fruit_pilot output, including its scene, trusted
runtime/policy and prepared integration. It is used only after authoring and
must match the source image. Only task1186 has a matching installation in this
cohort. Other cases remain unverified downstream. `--assess` writes
`$OUT/results.json`; it may use remaining requests for native role binding and
requires fresh assessment output directories. Planning is never resampled by
assessment. For a template-only rerun, add `--template-only` to **both** commands
and compare against a previously saved default result after verifying input
hashes, instructions, model and budget.

## Other evaluations

```bash
# Template / goal / hybrid static comparisons; --all selects all 20 cases.
python -m scripts.benchmark.task_engine.hybrid_pilot --dataset "$CACHE" --output "$OUT" --all
python -m scripts.benchmark.task_engine.grade_cohort "$OUT" "$OUT" "$SCORES"
# Fixed regression, new-instruction holdout, or independent camera-image suite.
python -m scripts.benchmark.task_engine.source_gate_trial --dataset "$CACHE" --suite regression --output "$OUT"
python -m scripts.benchmark.task_engine.source_gate_trial --dataset "$CACHE" --suite holdout --output "$OUT"
python -m scripts.benchmark.task_engine.source_gate_trial --dataset "$CAMERA" --suite camera --output "$OUT"
# Legacy archived-template vs goal study and goal coverage summary.
python -m scripts.benchmark.task_engine.compare --dataset "$CACHE" --archives "$ARCHIVE" --live-template --output "$OUT"
python -m scripts.benchmark.task_engine.coverage_summary "$OUT"
```

Choose a new `OUT` for every independent run. The latter static tools retain
three-route/legacy analysis semantics; their service-call budget is distinct
from the CLI HTTP meter. Never interpret a covered graph as a physical verdict.
Frozen holdout inputs remain beside the runner. Camera source images and
historical model responses are local artifacts, not bundled data.

## Recorded comparison: 2026-11-15

The default baseline used commit `101f3b07e`; template after visual-format repair
used the source change published in `8c252079b`. Both used original Chinese
instructions and images, `mimo-v2.6-pro`, at most five requests per case and route,
one trial per case, and no manual boxes or candidate proposals. Template was
rerun; hybrid is the saved baseline, not a simultaneous new sample.

| 题号 | template 发布/视觉/资格/预检 | hybrid 发布/视觉/资格/预检 | 请求 T/H |
|---|---|---|---:|
| task1171 | 失败 / 失败 / — / — | 通过 / — / — / — | 3/2 |
| task1172 | 通过 / 部分 / — / — | 通过 / — / — / — | 5/2 |
| task1173 | 失败 / 完整 / — / — | 通过 / — / — / — | 5/2 |
| task1174 | 通过 / 部分 / — / — | 通过 / — / — / — | 3/2 |
| task1175 | 失败 / 失败 / — / — | 失败 / — / — / — | 3/2 |
| task1176 | 通过 / 部分 / — / — | 通过 / — / — / — | 3/2 |
| task1177 | 失败 / 失败 / — / — | 通过 / — / — / — | 3/2 |
| task1178 | 失败 / 失败 / — / — | 通过 / 部分 / — / — | 3/3 |
| task1179 | 失败 / 失败 / — / — | 通过 / — / — / — | 3/2 |
| task1180 | 失败 / 部分 / — / — | 通过 / — / — / — | 5/2 |
| task1181 | 通过 / 部分 / — / — | 通过 / 部分 / — / — | 3/3 |
| task1182 | 失败 / 失败 / — / — | 通过 / — / — / — | 3/2 |
| task1183 | 失败 / 失败 / — / — | 通过 / — / — / — | 3/2 |
| task1184 | 失败 / 失败 / — / — | 通过 / — / — / — | 3/2 |
| task1185 | 失败 / 失败 / — / — | 通过 / — / — / — | 3/2 |
| task1186 | 失败 / 失败 / — / — | 通过 / 完整 / 通过 / 通过 | 3/4 |
| task1187 | 失败 / 失败 / — / — | 通过 / — / — / — | 4/3 |
| task1188 | 失败 / 失败 / — / — | 通过 / — / — / — | 3/2 |
| task1189 | 失败 / 部分 / — / — | 通过 / — / — / — | 4/2 |
| task1190 | 失败 / 完整 / — / — | 通过 / — / — / — | 4/2 |

“—” means **not verified**, not failed. Visual complete/partial is contract
status, not comparison with human box labels. For graph failures it describes
validated grounding artifacts; published plans use the manifest's visual state.

- Template published 4/20 (previously 0/20), used 69 requests, and had 8 validated
  visual responses: 2 complete and 6 partial. Four of those later failed graph
  checks. All 20 downstream qualification/preflight stages remained unverified.
- Default hybrid published 19/20 (16 authoring-only, 2 partial visual bindings,
  1 complete), used 45 requests including one qualification request. Only
  task1186 passed fresh scene qualification and seven-segment compilation.
  Sixteen authoring-only plans and the failed task were not visually verified.
- Template failures: invalid boxes (1171, 1183); extra reference-map keys (1175,
  1178, 1185, 1186); malformed entity (1177); unresolved source clauses (1179,
  1188); nonempty misplaced milestones (1182); invented boxes for unobserved
  entities (1184, 1187). Graph-stage failures were an origin-support alias treated
  as an observed entity (1173), missing handle-parent identity (1180, 1190), and
  missing upright restoration (1189). Archived ambiguity and invalid-entity
  responses remain rejected by current validation.
- No physics was run in these cohort comparisons. Separate installation evidence
  is retained in the examples: fixed block execution (1028 steps), task1186
  (4096 steps), and later live-default block/task1186 acceptance (1027/4096).
  These are geometric checks, not per-hand contact-force certificates. This
  development cohort and single sampling do not establish unseen-task reliability.

## Local historical archive

The full previous report and all 22 pre-cleanup example files are preserved,
byte-for-byte, under `outputs/task-engine-review-history/` on the authoring host.
`sha256.json` records each original file's digest. Complete cohort record files
are `default_cohort_results.json`, `template_format_results.json` and
`review_results.json`; the full multi-round narrative is `REPORT.md`. Raw plans,
failures and scene/deployment artifacts remain in their original local `outputs`
run directories. Neither this archive nor demo media are included in a clean
checkout. Rerun the commands above to generate independent records; do not claim
local historical evidence is distributed with the source.

## Layout-change verification

After relocation, all ten benchmark/example CLI help entries loaded, the
historical 60-row cohort could be regraded, and the 20-case goal coverage summary
could be regenerated without new model calls. task1186 was freshly qualified and
compiled with all three arm/tray/order negative controls, and its accepted-run
viewer was regenerated. Existing production CLI and goal-planning tests passed
(87), including hinge-closing rejection, plus 42 context checks. This validation
did not resample model plans or repeat physical execution. Retained small
acceptance records and all 22 archived source files were checked against their
pre-cleanup SHA-256 values.
