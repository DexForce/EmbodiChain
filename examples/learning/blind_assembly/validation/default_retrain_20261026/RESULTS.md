# Default-asset blind peg insertion: measured result

The validation-selected default-asset policy achieves **7240/8192 = 88.3789%**, Wilson95% interval **87.6669–89.0550%**, on8192 independent balanced episodes. **The95% engineering target is not met.** This report records failure evidence rather than declaring a completed paper reproduction.

The native PPO lineage starts from random weights, preserves the registered Rizon4S and Robotiq2F85 source assets, and reaches1000updates/131,072,000transitions. The original process disappears without finalization after logging958updates; its last saved checkpoint is950. Native resume restores policy, optimizer, normalizer and scheduler at950, then completes the remaining50updates. The unsaved8-update branch is discarded. The parent health file remains unchanged with its last recorded `running` status; the external interruption record explains that status. This is **not an uninterrupted run**.

## Time & Memory

| Measurement | Workload/scope | Wall seconds | Sampled process VRAM maximum(MiB) | Outcome |
|---|---|---:|---:|---|
| Parent retained checkpoint snapshot |4096arenas,950updates; setup/collection/PPO/validation/strict readbacks |26897.339|10500|Last retained snapshot; no normal process finalization |
| Parent observed process span |Health creation→last1Hz liveness sample, through logged958 |27235.353|10500|Approximate observed span, not exact exit time |
| Continuation |Native restore950→1000; process wall includes imports/setup/teardown |2094.259|10474|Completed, exit0 |
| Sum of observed segment spans |Parent last liveness span+continuation process wall |29329.612|Not summed|About8.147h; excludes interruption gap |
| Calendar first launch→continuation exit |Includes interruption/recovery gap |30247.335|Unmeasured|About8.402h |
| Balanced checkpoint selection |20saved updates×512episodes,64arenas,4processes |907.834|Unmeasured|Every candidate completed; final test excluded |
| Independent assessment |8192episodes,1024arenas,8each |247.645|Unmeasured|Completed with fresh strict health |

Hardware: RTX5090; Ryzen99950X. Parent/continuation setup until trainer-ready is222.588s/293.609s. The native continuation wrapper reports2072.963s; its owning process reports2094.259s, including imports and process cleanup. Sampling is PID-filtered1Hz, not transient allocation peaks or total-machine resources. Own development probes overlapped parts of the parent run. These are observed run costs, **not an isolated single-run throughput benchmark or a12288-arena/5000-update measurement**.

## Success & Other Metrics

| Cohort/measure | Count | Value | Qualification |
|---|---:|---|---|
| Selected development checkpoint |512|Update850:462/512=90.2344%|Balanced64×8,seed10000; selects weights |
| Independent success |8192|7240/8192=88.3789%[87.6669%,89.0550%]|Seed20000,1024×8; not used for selection |
| Failed endpoints |952|899grasp=false(94.43%);279captured;53length250|Overlapping endpoint observations; no causal timing classification |
| Episode duration |8192|Mean0.901187s; median0.760s; p951.720s|Length×0.02s; not measured native time per episode |
| Terminal penetration |8192|Mean3.670120µm; maximum175.631809µm|Final-substep task metric, not peak intersection or precision qualification |
| Fresh native health |799control boundaries×1024worlds|qpos/qvel/qacc finite; no capacity/unclassified flags; ITERATIONS observed|Convergence/accuracy not certified; no intermediate-substep state sampling |
| Native window preview |3predeclared seeds|3/3successes:39/39/43controls|0.78/0.78/0.86control-grid seconds; separate small visual sample |
| Actor-only export |3fixed random inputs|Maximum eager/export difference0.0|Input1023/output7; excludes critic/decoder |

## Leaderboard

| Candidate | Initialization/physical model | Assessment | Success | Engineering target |
|---|---|---|---|---|
| EmbodiChain native PPO,default assets,selected850 |Random native training; real source mimic tree with approximate equality compliance; interrupted lineage later reaches1000 |8192independent balanced episodes |**88.3789%[87.6669%,89.0550%]**|95%not met |

One geometry and one training seed are measured. The historical nominal97.7295%result uses another hand model and another runtime; it is not inherited by these assets. Three successful preview episodes do not replace the8192-episode assessment.

![Learning evidence](learning_curves.png)

## Geometry, physics, algorithm and learning conclusions

- Geometry/resource conclusion: default source URDFs, visuals, collision meshes, inertials and mimic metadata are frozen in the run. The stadium fixture is physically unchanged from nominal within1.11e-16rounding. No new micrometre-metrology claim is made here.
- Physics conclusion: the retained parent950updates and continuation50updates have finite sampled native states and no recorded capacity/unclassified loss. Fresh assessment independently passes those checks. Native iteration-budget flags occur; the soft linkage/contact model remains uncalibrated.
- Algorithm conclusion: native managers/policy/collector/buffer/PPO/trainer are used. Actor inputs exclude privileged pose/contact labels. The public actor numerical reference and exported-inference parity do not establish author-training-code equivalence. Native critic has235features versus the paper's240with an incomplete public feature definition.
- Learning conclusion: the independently measured88.38%does not meet the declared95%target. Multi-seed, broad geometry/contact/workspace distributions, OOD, ablations, other tasks and hardware deployment remain outside this experiment.

## Selection and execution provenance

All20saved lineage updates50…1000 are evaluated on the same balanced512-episode development protocol. Ties prefer earlier updates. The selected checkpoint is850/111,411,200transitions, SHA256`68c9625efbb7f85c07e22b1b2b0111e1a5e83d4ac386121927c57e0a898da116`. The frozen selected copy, native final evaluator, actor export and preview bind to this hash. Parent event files are preserved; continuation events replace duplicate discarded951…958steps by newer wall time. The learning report sees1000unique update events, while its native health/time input covers the50-update continuation only; this report preserves the separate parent health and full cost.

Training launch code is179d0e15with its captured working-tree patch. Continuation, checkpoint ranking, assessment and preview execute at40e8ed41. The immutable SDK snapshot records revision528daf plus the source-contact patch and unchanged mapped native-library hashes. MuJoCo and MuJoCo Warp are3.15.0,Newton1.7.0,Warp1.17.0,PyTorch2.7.1+cu128. The isolated EPA128 allocation overlay changes capacity only. No shared SDK checkout or installed package is changed by this run.

Requested physics/control cadence is1ms/20ms. The sampled first control advances21.000002ms and the second19.999985ms. Individual substep clocks are not measured. Float32 MuJoCo Warp clamps requested tolerance1e-8 to a prepared value near1e-6; no post-step tolerance certification is claimed. These details are shared qualification limits, not demonstrated causes of the success gap.

## Analysis and reuse

The [experiment review](EXPERIMENT_REVIEW.md) distinguishes confirmed setting changes, repeatability evidence and unproven causal hypotheses. [Result summary](summary.json),[balanced selection summary](selection-summary.json) and[all scored final episodes](episodes.jsonl.gz) retain the primary evidence in the current tree. Full logs, state snapshots and duplicate media are available through the [evidence archive](../../../../sim/assembly/validation/ARCHIVE.md); links below to those artifacts point to the immutable pre-cleanup commit. The [full original result](https://github.com/DexForce/EmbodiChain/blob/f8f5dfcc81b03e0fe4a9698d46bfcf406d9a89a0/examples/learning/blind_assembly/validation/default_retrain_20261026/result.json),[balanced selection](https://github.com/DexForce/EmbodiChain/blob/f8f5dfcc81b03e0fe4a9698d46bfcf406d9a89a0/examples/learning/blind_assembly/validation/default_retrain_20261026/selection.json),[parent health](https://github.com/DexForce/EmbodiChain/blob/f8f5dfcc81b03e0fe4a9698d46bfcf406d9a89a0/examples/learning/blind_assembly/validation/default_retrain_20261026/parent_training_health.json),[continuation health](https://github.com/DexForce/EmbodiChain/blob/f8f5dfcc81b03e0fe4a9698d46bfcf406d9a89a0/examples/learning/blind_assembly/validation/default_retrain_20261026/training_health.json),[final health](https://github.com/DexForce/EmbodiChain/blob/f8f5dfcc81b03e0fe4a9698d46bfcf406d9a89a0/examples/learning/blind_assembly/validation/default_retrain_20261026/evaluation_health.json),[interruption](https://github.com/DexForce/EmbodiChain/blob/f8f5dfcc81b03e0fe4a9698d46bfcf406d9a89a0/examples/learning/blind_assembly/validation/default_retrain_20261026/external_interruption.json) and[resource samples](https://github.com/DexForce/EmbodiChain/blob/f8f5dfcc81b03e0fe4a9698d46bfcf406d9a89a0/examples/learning/blind_assembly/validation/default_retrain_20261026/resource_samples.jsonl.gz) preserve both positive and negative evidence.

![Initial and terminal frames](https://raw.githubusercontent.com/DexForce/EmbodiChain/f8f5dfcc81b03e0fe4a9698d46bfcf406d9a89a0/examples/learning/blind_assembly/validation/default_retrain_20261026/contact_sheet.png)

[Complete native MP4](preview.mp4),[trace](https://github.com/DexForce/EmbodiChain/blob/f8f5dfcc81b03e0fe4a9698d46bfcf406d9a89a0/examples/learning/blind_assembly/validation/default_retrain_20261026/preview_trace.json),and[actor export record](actor_export.json) retain all predeclared preview seeds30000–30002. MP4 is960×540,25fps,95frames,3.8s; no episode is omitted. The18mm-deep plate is seated when the peg's bottom reaches its seat plane; most of the90mm peg remains above the plate. Camera reads do not advance physics. The preview has no separate strict native-state instrumentation.

The local model archive ships the selected native checkpoint, frozen relative configs/resources, actor-only export and evidence. Its external `model_descriptor.json` records archive CRC/SHA verification and is intentionally external to the archive to avoid a self-checksum cycle. No author policy weights or proprietary SDK binaries are redistributed. Preserve the available source-asset attribution and license records; this delivery does not assign a new license to the source meshes.

## Publication validation

Main81116eb3 is merged as79e63c8a after the scientific assessment. The affected configuration/catalog/Spawn/task/RL suite passes857unique CPU cases; API coverage is3306/3306 and Black26.10.1 passes. The relocated downloaded model completes8fresh integration-smoke episodes(seed40000,8arenas), with6successes, finite sampled native state and no capacity/unclassified loss. This small check does not replace the8192primary assessment. Its teardown emits an SDK entity-accounting warning(`1024!=1023,diff1`) while the process exits0, so leak-free teardown is not claimed. See[publication validation](https://github.com/DexForce/EmbodiChain/blob/f8f5dfcc81b03e0fe4a9698d46bfcf406d9a89a0/examples/learning/blind_assembly/validation/default_retrain_20261026/publication_validation.json) and[relocated replay health](https://github.com/DexForce/EmbodiChain/blob/f8f5dfcc81b03e0fe4a9698d46bfcf406d9a89a0/examples/learning/blind_assembly/validation/default_retrain_20261026/publication_replay/health.json). The full Sphinx site and clean isolated dependency installation were not rebuilt.
