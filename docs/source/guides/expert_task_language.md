(expert-task-language)=

# Task and segment language for expert data

An expert episode has an **overall task instruction** and one or more
**segment instructions**. For example, the overall task can be "Stack block 2
on block 1", while its segments describe "Grasp and lift block 2" and "Place
block 2 on block 1 and release it". Every frame retains the overall goal and
the instruction for the segment currently executing.

These two annotation levels apply to handwritten experts, configured and
dynamic Task Programs, trajectory expansion, synchronous/asynchronous LeRobot
recording, and online generation. Repeats and parallel blocks describe workflow
structure; they do not create additional levels in the recorded language
schema. Instructions do not automatically infer segment boundaries or prove
physical success.

## Choose where to declare the language

| Expert source | Overall task instruction | Segment instruction |
| --- | --- | --- |
| Handwritten Python, MotionGenerator, or Atomic Skills | `env.task_instruction` in the Gym config, or `EmbodiedEnvCfg.task_instruction` in Python | `DemoSegment.instruction` from `create_demo_segments()` |
| Configured Task Program | Root `instruction` in `program.yaml` | `instruction` on each `kind: segment` node |
| Dynamic Python or validated MLLM Task Program | `TaskProgramCfg.instruction` on the program actually selected for the episode | `SegmentCfg.instruction` on that program's segments |
| Trajectory expansion | The source task/program instruction, unless explicitly overridden for an episode | The source semantic segments; augmentation does not infer new labels |
| Legacy action-list expert | Prefer `env.task_instruction`; the recorder's old `instruction` parameter remains a fallback | One `legacy` segment, using the overall instruction |

Use non-empty strings for new task and segment fields. For serialized Task
Programs and `env.task_instruction`, do not include leading or trailing
whitespace. The old `{"lang": "..."}` mapping is supported only at the legacy
recorder boundary. A recorder's `extra.task_description` is descriptive metadata,
not a source of task language.

## Configure a Task Program

The following is the complete program used by the three-cycle cube example at
`embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task_program/program.yaml`:

```yaml
program_id: repeated_cube_pick_place
instruction: Move the cube between the two target positions three times.
targets:
  drop_pose:
    kind: cyclic_pose
    values:
      - position: [-0.40, 0.48, 0.10]
        quaternion_xyzw: [0.0, 0.0, 0.0, 1.0]
      - position: [-0.42, -0.08, 0.10]
        quaternion_xyzw: [0.0, 0.0, 0.0, 1.0]
program:
  kind: repeat
  count: 3
  body:
    kind: sequence
    items:
      - kind: segment
        name: pick_cube
        instruction: Grasp and lift the cube.
        steps:
          kind: invoke
          call:
            kind: pick
            object: cube
      - kind: segment
        name: place_cube
        instruction: Place the cube at the next target position and release it.
        steps:
          kind: invoke
          call:
            kind: place
            object: cube
            at:
              kind: target_ref
              target: drop_pose
```

Compilation produces **six segment occurrences**: pick, place, pick, place,
pick, place. Their names and descriptions can repeat; each occurrence still has
its own episode-local identity and frame range. Equal descriptions share one
subtask vocabulary entry. A `subtask_index` therefore identifies text, not a
particular segment occurrence.

Keep the program's overall instruction in `program.yaml`; configure the
recorder separately in the selected physical environment. The deployment
supplies the trusted integration, robot embodiment, and execution policy. See
{doc}`/tutorial/task_program` when creating another deployment.

Run the shipped example from the repository root:

```bash
embodichain run-env \
    --gym_config embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.yaml \
    --headless \
    --max_episodes 1
```

Its Default environment already enables LeRobot recording under
`/tmp/repeated-pick-place/datasets`. Inspect the completed episode:

```bash
embodichain preview_lerobot_data \
    /tmp/repeated-pick-place/datasets \
    --latest \
    --episode 0 \
    --expect-segments 6
```

For a four-stage example, PourWater declares pick, move above the cup, pour,
and return segments. It also retains a measured bottle-position check before
pouring and final bottle/cup checks. Those validators are independent of the
language annotations; this example does not measure liquid transfer.

## Annotate a handwritten expert

Put the overall goal beside the ordinary environment settings. This YAML block
is merged into an existing handwritten task's Gym configuration; it does not
replace that task's scene, robot, or registration:

```yaml
env:
  task_instruction: Stack block 2 on block 1.
  dataset:
    lerobot:
      func: LeRobotRecorder
      mode: save
      params:
        save_path: outputs/lerobot/stack_blocks
        robot_meta:
          robot_type: CobotMagic
        use_videos: false
```

In a Python configuration, assign the same field on `EmbodiedEnvCfg`:

```python
from embodichain.lab.gym.envs import EmbodiedEnvCfg

cfg = EmbodiedEnvCfg(task_instruction="Stack block 2 on block 1.")
# Assemble the task's robot, scene, and other required settings before use.
```

Return explicit descriptions from the task's `create_demo_segments()` method.
The shipped stacking task implements the complete planner, action conversion,
and validators, so the following example stays synchronized with production:

```{literalinclude} ../../../embodichain_tasks/embodichain_tasks/manipulation/tableware/stack_blocks_two.py
:language: python
:start-at:     def create_demo_segments(
:end-before:     def _plan_stack(
```

The task yields two segments from the existing PickUp/Place trajectory split.
The grasp segment checks measured lift; the placement segment owns release
settling and final stack validation. For your own planner, retain its actual
action order and choose boundaries around describable subgoals. Use lazy
iterables when later planning depends on the state left by an earlier segment.
The planner yields actions; the common executor calls `env.step()`.

Legacy `create_demo_action_list()` tasks continue to work as one `legacy`
segment. A single action list does not acquire finer labels merely because an
overall instruction has been added. See {doc}`/tutorial/data_expansion` for
expert authoring and collection lifecycle details.

## Override one episode and understand fallback

The common executor resolves the overall instruction in this order:

| Priority | Source | Recorded `instruction_source` |
| --- | --- | --- |
| 1 | `execute_demo_episode(..., instruction="...")` | `episode` |
| 2 | Root instruction of the Task Program selected for this episode | `task_program` |
| 3 | `EmbodiedEnvCfg.task_instruction` | `task` |
| 4 | Legacy recorder instruction | `legacy_recorder` |
| 5 | No instruction at any of these sources: `unknown_task` | `unknown` |

For an already created and reset environment, an explicit override applies to
one execution:

```python
from embodichain.lab.gym.envs import execute_demo_episode

result = execute_demo_episode(
    env,
    instruction="Move the cube between the target positions three times.",
)
print(result.instruction)         # The resolved overall goal
print(result.instruction_source)  # "episode"

# A direct caller owns the transaction: commit success, discard failure.
env.reset(options={"save_data": result.success})
```

If planning or execution raises an exception, discard the attempt with
`env.reset(options={"save_data": False})` before retrying. The `run-env`
collector manages commit/discard and retries for CLI collection. Finalizing or
closing a recorder drains committed writes; it does not commit an unfinished
rollout.

The executor freezes the resolved overall text before planning. Changing
`env.cfg.task_instruction` or a recorder's configuration later cannot rename
that episode. Task Program compilation also freezes the program's root and
segment instructions. The recorder consumes these snapshots, including when
an async writer processes the episode after execution.

A dynamic program is supplied directly as a planning argument:

```python
# selected_program is a TaskProgramCfg or compiled program compatible with the
# environment's trusted Task Program integration.
result = execute_demo_episode(env, task_program=selected_program)
env.reset(options={"save_data": result.success})
```

Its root instruction replaces the static program as the program-level source.
If the selected program has no root instruction, resolution may still use the
task configuration or legacy recorder fallback; it never borrows a different
static program's goal. If all sources are absent, `unknown_task` remains frozen
through persistence. Validated MLLM programs follow the same rule and decoder.

For segments:

- An explicit segment instruction is preserved and marked
  `instruction_source="segment"`.
- An omitted instruction uses the resolved episode goal and is marked
  `instruction_source="task_fallback"`.
- An episode override changes these fallback labels, but does not rewrite
  explicitly authored segment descriptions.

Fallback labels are useful for compatibility but do not supply a finer subgoal.
If multiple legacy recorders specify different instructions and no higher
priority source resolves the task, execution raises `ValueError`. Configure
one authoritative task/program instruction instead.

## Read the two levels from LeRobot data

These mappings are independent of the action representation. Expert and policy
action contracts, full episodes, and independently saved segment fragments
retain both language levels:

| Stored field or metadata | Meaning |
| --- | --- |
| `task_index`, `meta/tasks.parquet` | Dataset vocabulary for the overall episode goal |
| `subtask_index`, `meta/subtasks.parquet` | Dataset vocabulary for the current segment instruction |
| `annotation.segment_id` | Episode-local segment occurrence; independent of the subtask vocabulary ID |
| `annotation.segment_start`, `annotation.segment_end` | First/last frame of the segment |
| `meta/embodichain_episodes.jsonl` | Overall `instruction`/source, per-segment instructions/sources, frame ranges, and execution outcomes |

LeRobot 0.4.4 resolves the vocabulary IDs into `sample["task"]` and
`sample["subtask"]`. After collection and recorder finalization, use the exact
auto-numbered dataset directory printed by the runner:

```python
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset_dir = Path("outputs/lerobot/stack_blocks/0000")  # Use your actual directory.
dataset = LeRobotDataset(repo_id=dataset_dir.name, root=dataset_dir)
sample = dataset[0]
print(sample["task"])           # Overall stacking goal
print(sample["subtask"])        # Grasp/lift instruction on the first segment
print(sample["task_index"])
print(sample["subtask_index"])
```

Inspect annotation provenance and ranges without decoding camera images:

```python
import json

sidecar = dataset_dir / "meta" / "embodichain_episodes.jsonl"
with sidecar.open(encoding="utf-8") as stream:
    episode = json.loads(next(stream))

print(episode["instruction"], episode["instruction_source"])
for segment in episode["segments"]:
    print(
        segment["name"],
        segment["instruction"],
        segment["instruction_source"],
        segment["start_step"],
        segment["end_step"],
    )
```

Segment ranges are half-open: `[start_step, end_step)`. In
`DemoExecutionCfg(mode="segment_fragments")`, an eligible natural segment is
saved as a separate LeRobot episode, but its `task` still describes the overall
source task and `subtask` describes the saved segment. Fragment sidecars retain
source identity/ranges. This mode does not add checkpoint restore or resume
after a failed segment. See {doc}`/overview/gym/dataset_functors` for recording,
fragment metadata, depth, and preview options.

## Resolve language in online training

Online batches contain numeric `task_index` and `subtask_index` tensors. Their
vocabularies are shared across producer/consumer processes and append new text
without reassigning existing IDs. They are **engine-local**, not indices into
an offline LeRobot dataset. Use the engine that produced the batch to resolve
them.

With an already started `OnlineDataEngine`, the consumer can retrieve both
levels without placing Python strings in shared TensorDict storage:

```python
from torch.utils.data import DataLoader

from embodichain.data_pipeline.datasets.online_data import OnlineDataset

dataset = OnlineDataset(
    engine,
    chunk_size=8,
    batch_size=4,
    sampling_mode="segment",
)
loader = DataLoader(
    dataset,
    batch_size=None,
    num_workers=0,
    collate_fn=OnlineDataset.passthrough_collate_fn,
)
batch = next(iter(loader))
language = dataset.resolve_language(batch)  # Also: engine.resolve_language(batch)

print(batch["task_index"].shape)  # torch.Size([4, 8])
print(language["task"][0][0])
print(language["subtask"][0][0])
# Tokenize language["task"] and/or language["subtask"] in your training code.
```

`sampling_mode="segment"` keeps each window inside one accepted segment;
the example requires a valid segment of at least eight frames. `"episode"`
may cross semantic segment boundaries, and `"boundary"` deliberately samples
across an accepted boundary. In those modes, use the per-frame subtask texts
instead of assuming the whole chunk has one subtask. All modes respect causal
continuity boundaries.

Text lookup preserves the sampled tensor dimensions. A copied batch continues
to resolve after trajectory slots are refilled, and after shutdown while its
engine/registry remains retained. Keep engine start/stop in the creator process.
For multiprocessing DataLoaders, use top-level, picklable transforms and a
guarded `if __name__ == "__main__":` entry point; resolve/tokenize sampled IDs
in the consumer rather than transferring simulator objects.

`OnlineDataEngineCfg.language_buffer_bytes` defaults to 1 MiB for the lifetime
append-only UTF-8 registry. Many unique episode goals can exhaust it. Overflow
fails generation before publishing the affected rollout; create a new engine
with a larger capacity. Missing IDs, unknown IDs, and padding ID `-1` are lookup
errors. See {doc}`/api_reference/embodichain/embodichain.data_pipeline.engine`
for the engine and registry APIs.

## Migrate existing configurations and consumers

1. Move the recorder's overall `instruction` text to `env.task_instruction` for
   a handwritten task, or to the program root for a Task Program. Replace the
   old language mapping with a string and remove duplicate recorder labels.
2. Keep recording options such as `save_path`, `robot_meta`, and `use_videos`
   under the recorder. Both `LeRobotRecorder` and `AsyncLeRobotRecorder` use
   the same language contract.
3. Add explicit segment instructions where a finer goal exists. If a recorded
   trajectory has no reliable semantic boundaries, keep one descriptive segment.
4. Read `sample["task"]` for overall-goal conditioning and `sample["subtask"]`
   for segment conditioning. Older action-contract consumers that treated
   `task` as the segment label must switch to `subtask` for newly recorded data.
5. Recheck hard-coded segment counts and IDs. The official examples now have
   six segments for RepeatedPickPlace/RubiksCubePickPlace, four for
   PourWater/BlocksRankingRGB, and two for StackBlocksTwo. Repeated labels can
   share a vocabulary ID while retaining distinct segment occurrences.

Existing dataset files are not rewritten. Check each old dataset's schema and
annotations before combining it with new recordings. This language contract
uses the existing LeRobot 0.4.x dependency and does not require LeRobot 0.6.1 or
a change to EmbodiChain's supported Python versions.
