# Directional carton stocking

This configured Task Program uses one real Panda + Robotiq 2F-85 demonstration
from DROID to pick, transport and place a rigid carton at a requested position
and yaw. Every collected candidate derives from the real motion prototypes.

```bash
embodichain run-task \
  --gym-config extensions/embodichain_tasks/configs/tasks/manipulation/carton_stocking/task.panda.yaml \
  --dataset-dir /tmp/carton-stocking
```

The default collects six baseline recipes: nominal, joint residual seeds 7/11,
a 5 mm task-space variant, and free-phase duration scales 1.15/1.30. Use
`--max_episodes 1` for a nominal smoke run or `--viser` for live browser viewing.
The standard runtime loads the pinned inputs automatically. Optional cache warmup:

```bash
embodichain data download --name Real2SimCarton
```

## Ownership and motion sources

| File | Responsibility |
| --- | --- |
| `env.yaml` | One dynamic carton, static worktop, default ground and physics |
| `task_program/program.yaml` | Pick → MoveHeldObject → Place and segment acceptance |
| `task_program/integration.yaml` | Semantic entities, calibrated affordances, three pinned source bindings and refinement |
| `real2sim/collection.yaml` | Layouts, recipes, source evidence, motion and measured acceptance limits |
| `task.panda.yaml` | Select components, Real2Sim recorder owner and collection target |

The deployment reuses the SDK's `franka_panda_robotiq85` embodiment and
`real2sim_verified` execution policy. No task environment subclass is needed.
Source bindings retain the existing program fingerprint and call occurrences
0/1/2. The ordinary Task Program bridge, Gym stepping and qualified synchronous
LeRobot sink own execution and submission.
The Place call explicitly requests 3 cm of vertical retreat, matching the
measured prototype's approximately 3.16 cm. The external skill validates that
clearance using FK; requesting more than the prototype provides is rejected.

Prototype and evidence references have the closed form
`{asset: Real2SimCarton, path: inputs/measured/0/prototype.json}`. Only registered
asset IDs and files within their verified bundle are accepted. The resolver
preserves ordinary owner-relative file references for rebuilt sources. The
prototype digest, normalized arrays, raw recording and typed evidence are
verified separately; a successful download does not establish task success.

The immutable `Real2SimCartonV1` distribution is pinned to Hugging Face revision
`3c340238f6acc29a50272c33f79264764d0704ff`. The selected episode is
`RAIL+d027f2ae+2023-11-20-11h-50m-49s`. Its raw recording and measured derivatives
retain CC BY 4.0 attribution. Historical independent synthetic files in that
published v1 archive are unused by this task. Published bytes are not replaced.

## Scope and acceptance

- B=1, `physics: default`, 30 Hz control and 240 Hz physics; no sensor recording.
- Original Panda and coupled Robotiq geometry; same cuboid object and support.
- Object/target XY and yaw transfer, affordance refinement, protected same-grid
  trajectory expansion and mapped time expansion.
- IK/FK position ≤2 mm and rotation ≤2°, measured lift ≥3 cm, held relation,
  placement center ≤5 mm, directed yaw ≤3°, full support footprint and stable
  drift ≤3 mm. Motion and sampled collision checks remain mandatory.
- Confirmed coverage requires actual submitted actions, T+1 observations,
  verified source/trace attachments and a matching confirmed receipt.

The complete benchmark contains 13 layouts × 6 recipes = 78 candidates.
Individual candidates may be rejected; the benchmark reports every result and
does not certify the whole rectangle as feasible. Ordinary collection stops on
failure. The full benchmark runs independent B=1 candidates and retains failures.

This validates public real motion in a declared estimated cuboid proxy. It does
not validate real-object dynamics or hardware deployment. RL, reconstruction,
new shapes, cross-robot transfer and B>1 collection are outside this task.

## Rebuild and inspect

The example reads these official templates to reconstruct the three prototypes
from the original recording, apply new layouts/recipes, and export replay/video:

```bash
embodichain run-example expansion/trajectory/real2sim_carton -- \
  --output /tmp/carton-example
```

See the packaged example's README for preparation-only, repeated collection
and the complete 78-candidate benchmark. Reports, datasets and media belong to
the selected output directory, not the task package or asset cache.
