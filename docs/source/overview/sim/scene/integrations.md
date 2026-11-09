# Where Scene Services Are Used

Scene services have several production consumers with different lifecycles.
Choose an integration based on who owns initialization, stepping, and result
acceptance. The paths below are repository-relative change sites.

## GenSim generation and editing

Scene Engine keeps model calls, asset preparation, its editable support graph,
hierarchy traversal, Y-up conversion, diagnostics, and export. Its adapters reuse
the scene geometry and layout strategies:

| GenSim caller under `embodichain/gen_sim/scene_engine/` | Shared operation |
| --- | --- |
| `pipeline/utils/table_support_surface.py` | Main horizontal support detection |
| `pipeline/utils/assets_group_table_aligner.py` | Shared group-height offset |
| `pipeline/utils/assets_group_support_clamp.py` | Support-valid group translation |
| `pipeline/utils/assets_group_layout_optimizer.py` | Greedy AABB separation |
| `pipeline/utils/table_surface_layout_optimizer.py`, `parent_surface_layout_optimizer.py` | `LayoutProblem` and `solve_layout` |
| `pipeline/utils/gravity_settler.py` | `SceneView` pose readback and `settle_scene`, within a temporary caller-owned simulation |
| `randomization/adapter.py` | Source-independent expansion preparation, generation, and persistence |

The image/edit pipelines do not need to construct a `SceneRecipe`. That API is
the explicit configuration-based route shown in the
[usage workflows](workflows.md). See [Scene Engine](../../../features/generative_sim/scene_engine.md)
and the [adapter reference](../../../api_reference/embodichain/embodichain.gen_sim.scene_adapters.rst)
for generation commands and adapter APIs.

## Gym initialization for expert and RL tasks

`embodichain/lab/gym/envs/managers/randomization/_scene_layout.py` implements
`ExpandSceneLayout`. It uses the shared expansion APIs as an ordinary reset
event; handwritten expert and simulator RL environments can use it without a
Task Program.

For an environment that already declares `bench`, `tray`, and `mug`, the event
configuration is:

```yaml
env:
  events:
    scene_layout:
      func: ExpandSceneLayout
      mode: reset
      params:
        entity_cfg: {uid: bench}
        entity_cfgs:
          - {uid: tray}
          - {uid: mug}
        parent_by_uid: {tray: bench, mug: tray}
        source: {kind: online}
        policy:
          backend: batched
          device: cpu
          group_targets: [tray]
        candidate_count: 64
        max_attempts: 10000
```

This is an event fragment for an existing runnable environment, whose physical
environment config still owns its `physics: default|newton` declaration. The
event's `entity_cfgs` covers every non-support entity in the explicit tree.
Schedule it after other pose initialization and before state-registration or
observation events. Overlapping built-in pose/scale randomizers are rejected.

Ordinary selective resets retain prepared geometry and unused candidates.
Explicit reseeding rewinds the candidate stream. A saved source uses
`source: {kind: saved, layout_dir: /absolute/run/path}` to cycle validated v4
layouts without resampling. The event does not step physics. EventManager owns
scheduling and seeds; `episode_context` owns per-row provenance and recording
eligibility. See [manager functors](../../../guides/custom_functors.md) for the
surrounding environment mechanism.

## Task Program scene expansion

Two layers connect proposals to actual expert execution:

| Owner | Responsibility |
| --- | --- |
| `embodichain/lab/task_program/integrations/scene_expansion.py` | `execute_scene_variant`: preparation callback, canonical Task Program execution, and measured result collection |
| `embodichain/lab/task_program/integrations/simulation/scene_expansion.py` | `SimulationSceneExpansionHost`: fixed B=1 physical preparation and restoration, initial checks, sensor/history refresh, and fresh bridge creation |
| `embodichain/lab/gym/envs/base_env.py`, `embodied_env.py`, `demo.py` | Reset ordering, environment stepping, observation/recording initialization, and persistence eligibility |

The simulation host delegates numeric capture/restore, rigid-pose application,
state comparison, and settling to `sim.scene`. It also validates physical
parameter and control-period invariants around reset. It recreates the Task
Program bridge so providers and planners observe the changed scene even if the
simulation clock is unchanged.

Proposal checks, measured initial-state checks, program completion, and measured
task acceptance are separate gates. Rejected recording buffers must be discarded
through reset before retrying. See the
[Task Program scene expansion reference](../../../api_reference/embodichain/embodichain.lab.task_program.integrations.scene_expansion.rst)
and [simulation host reference](../../../api_reference/embodichain/embodichain.lab.task_program.integrations.simulation.scene_expansion.rst)
for the callback and restoration contracts.

## Expansion Engine and scene/robot feasibility

`embodichain/lab/expansion/` adapts these services to augmentation and offline
collection. In particular, `scene.py` applies joint scene/robot candidates;
`_gym_prepare.py` captures physical and parameter baselines; `visual.py` publishes
saved measured states through `restore_render_state` with physics disabled.
The host owns execution, reset, recording, and persistence.

`scene.feasibility` searches declared scene-group and robot-installation
variables through explicit evaluation ports. Its Task Program adapters live in
`embodichain/lab/task_program/integrations/feasibility.py` and
`integrations/simulation/feasibility.py`. They retain the canonical program and
physical acceptance path. The
[feasibility reference](../../../api_reference/embodichain/embodichain.lab.sim.scene.feasibility.rst)
documents the runnable UR5 search, budgets, query geometry, and independent
validation. A previously successful proposal still requires fresh execution
evidence when reused.

## Atomic Skills, observations, and planning

`embodichain/lab/sim/atomic_actions/runtime.py` and `source_native_contact.py`
use `scene.native_hand` for strict native hand/rigid geometry and physical
offset capture. Task Program simulation observations use `scene.rigid_mesh`;
the stability adapter uses `NativePlanarPlacement` for measured footprints.
These acquisition routes preserve explicit geometry provenance and unavailable
evidence. Their callers choose grasp policy, thresholds, observation timing,
and semantic acceptance.

The cuRobo adapter in
`embodichain/lab/sim/motion/planners/curobo/curobo_planner.py` uses
`snapshot_collision_sources` to compose explicit shape descriptors and
same-observation poses into owned planner providers. Input envelopes, native
cooked shapes, and contact observations have different contracts; select the
acquisition API required by the consumer. See the
[scene geometry API](../../../api_reference/embodichain/embodichain.lab.sim.scene.rst)
and [cuRobo planner overview](../motion/planners/curobo_planner.md).
