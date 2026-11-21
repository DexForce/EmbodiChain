# Task Engine examples

Run these commands from the repository root with EmbodiChain, its task/robot
extensions, USD and the DexSim Default backend installed. Every output directory
must be new. These examples use procedural rigid proxies and task-owned robot
calibration; they do not reconstruct the input photograph into physical assets.

## Two-block Pick/Place

```bash
python -m examples.gensim.goal_authoring.stack_pilot --output outputs/block-example
```

The example generates its own overhead image, replays fixed E1 interpretation
and grounding, qualifies a UR5 scene, compiles and executes Pick/Place, and
measures the small block against the large block in the terminal frame.
Inspect `execution.json`, `terminal.json` and the separate certificate verdict.
The historical fixed-input run completed 1028 steps with a 2.0 mm terminal
support gap. Its result JSON is archived locally; inspect the new run's outputs
for its own verdict.

## task1186: handovers and coordinated transport

`IMAGE` is the original `task1186.png` from the pinned dataset snapshot
`569f1860815d256ad87f53f31e19a71363c8afb9` under
[gen_sim_assets](https://huggingface.co/datasets/DexForceAI/embodichain_data/tree/569f1860815d256ad87f53f31e19a71363c8afb9/gen_sim_assets).
The fixture checks its image digest. The program hands each fruit from right to
left, places it with the left arm, releases both hands, then moves the same tray
with both arms. It rejects wrong-arm, wrong-tray and missing-order witnesses
before qualification. Fruit geometry uses rigid proxies.

```bash
# Replay the fixed interpretation and reviewed image bindings; qualify and compile.
python -m examples.gensim.goal_authoring.fruit_pilot --image "$IMAGE" --output outputs/fruit-preflight
# Also execute and measure holders, tray lift and all three final support relations.
python -m examples.gensim.goal_authoring.fruit_pilot --image "$IMAGE" --output outputs/fruit-example --execute
```

The fixed `task1186_fixture.json` makes this an installation example independent
of live model sampling. An executed run writes `review.json` with negative
controls and geometric measurements. The historical run completed 4096 steps,
including a 66 mm tray lift and three satisfied terminal support relations.
These checks do not certify per-hand contact force.

## Live default image entry

Set `OPENAI_BASE_URL`, `OPENAI_API_KEY` and `OPENAI_MODEL` for your service, or use
the project's configured model environment. The sample service configuration
sets one transport attempt. The following planning command supplies only image
and instruction; it omits route, proposal and human grounding.

```bash
export EMBODICHAIN_MODEL_SERVICES_CONFIG="$PWD/examples/gensim/goal_authoring/services.json"
embodichain gensim task plan --image "$IMAGE" --instruction "$INSTRUCTION" --task-id "$TASK" --output "$OUTPUT/plan"
# INSTALLATION is a complete stack_pilot or fruit_pilot output, including prepared/.
python -m examples.gensim.goal_authoring.auto_acceptance "$OUTPUT" "$INSTALLATION"
```

`auto_acceptance` reuses only the matching installation and calibration,
requalifies the new plan, compiles it and performs independent terminal geometry
checks and writes `$OUTPUT/acceptance.json`. Do not reuse a scene from a different
source image. Historical live-default runs completed 1027 steps for blocks and
4096 for task1186; they cover these two installations, not general
image-to-execution success.

## Unsupported hinge closing

```bash
embodichain gensim task plan --route goal --task-id door --instruction 关上铰链门 --proposal examples/gensim/goal_authoring/close_hinge.json --deployment embodichain/extensions/tasks/configs/tasks/manipulation/tableware/stack_blocks_two/task.cobotmagic.yaml --output outputs/door-preflight
```

For this Pick/Place installation, the declared hinge-closing operation must
produce a rejected deployment preflight with
`closing_direction_unsupported`; publishing a graph does not authorize execution.
The existing production CLI regression retains this negative case.

## Local interactive evidence

```bash
mkdir -p outputs/task1186-review
python -m examples.gensim.goal_authoring.fruit_view "$RUN" outputs/task1186-review/index.html --replay
```

`RUN` must be a complete accepted fruit_pilot run containing `review.json`,
`execution.json`, its plan and prepared deployment. `--replay` expects a sibling
`execution.mp4`; omit the flag for a page without video. Existing media are kept
locally in `outputs/task1186-pr-preview/`, excluded from Git. Open its
`index.html` or extract `interactive.zip`. Node evidence comes from the original
4096-step accepted run; the video is a separate replay of the same deployment.

For same-budget route comparisons, scoring and the concise results table, see
[Task Engine benchmarks](../../../scripts/benchmark/task_engine/README.md).

## Historical result files

`automatic_results.json`, `stack_result.json` and `task1186_result.json` are
archived under `outputs/task-engine-review-history/` on the authoring host.
They are excluded from Git and are not required to run these examples. The
archive's `sha256.json` records their original digests. A clean checkout retains
`task1186_fixture.json`, `close_hinge.json` and `services.json` as example inputs;
use the commands above to generate new run-specific results under `outputs`.
