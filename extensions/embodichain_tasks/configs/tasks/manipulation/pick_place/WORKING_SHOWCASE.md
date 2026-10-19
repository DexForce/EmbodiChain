# Working PickPlace policy validation

This record establishes physical feasibility of the current ChunkBC/DLS recipe
and preserves its later refresh regression. The executed implementation is
retained at `b683c4c699a71562341d4b213d06857f0fbdf744`.
[Structured evidence](WORKING_SHOWCASE_RESULTS.json) keeps the qualification,
checkpoint-selection basis and every outcome in both 100-scene evaluations.
For the current acquisition-inclusive comparison, use the
[four-arm results](OFFLINE_PILOT_RESULTS.md).

## Fixed recipe and acceptance

- Franka single PickPlace, real contact, unchanged cube reset distribution,
  240/30 Hz physics/control, and a 600-step horizon.
- The cube must lift at least 5 cm, settle within 3 cm of the goal, and remain
  stable for 0.3 seconds after finger opening and TCP withdrawal.
- Two measured 51-feature observations feed a two-layer MLP-512 with 643,712
  trainable parameters. It predicts 64 absolute TCP/gripper targets and executes
  16 before replanning. No expert phase, clock or success label enters inference.
- Observation and target normalization are frozen from the initial 32 episodes.
  AdamW uses learning rate 3e-4, weight decay 1e-6, gradient norm limit 1,
  batch size 256 and training-only normalized observation noise 0.01.
- A 32-episode CPU ODS cache supplies two generations over two 5,000-update
  rounds. Immutable checkpoint selection uses development performance.
- The expert uses a cube-symmetric top-down grasp close to the initial wrist
  orientation. The policy and expert use the same bounded DLS action processing.

## Physical results

| Check | Complete successes | Interpretation |
| --- | --- | --- |
| Expert qualification | 40/40 | Meets the physical expert requirement. |
| Online step 5,000, development | 20/20 | Selected before reserved-test outcomes were available. |
| Online step 10,000, development | 18/20 | Consumed the refreshed generation; regression retained. |
| Selected step 5,000, reserved scenes | 99/100 | Meets the original 80/100 learning target. |
| Later step 10,000, same reserved scenes | 65/100 | Fails the target; continuous refresh is not qualified. |
| One-/four-demo diagnostics | 6/10 and 8/20 | Fail their 9/10 and 18/20 repeatability thresholds. |

The selected policy lifted 100/100 cubes. Its one failure, seed 50060, ended
3.37 cm from the goal and was rejected by the unchanged 3 cm criterion. Mean
final goal error across all 100 scenes was 11.03 mm. Student evaluation used
only model inference and environment action handling.

## Retention finding

A frozen-model audit over the archived training anchors gives:

| Source | Arm MSE at 5,000 / 10,000 | Gripper MSE at 5,000 / 10,000 |
| --- | --- | --- |
| G1 | 0.000526 / 0.000918 | 0.000847 / 0.001416 |
| G2 | 0.001100 / 0.000407 | 0.002763 / 0.000462 |

Fit improves on G2 and worsens on G1. This is consistent with forgetting after
replacement-only training at a fixed learning rate; it does not isolate every
cause of rollout failure. The run generated 64 accepted episodes and actually
consumed both generations. Development selection protects the delivered
checkpoint but does not establish stable continual learning. This finding
motivates bounded reuse and the separate [quota validation](ODS_QUOTA_TESTS.md).

## Limits

These are historical single-training-seed results. Reserved scenes 50000–50099
were exposed by these evaluations and cannot serve as fresh test data for a
later study. The initial integrated runs disabled their in-run final evaluation;
the separate frozen-checkpoint verification is the source of the 99/100 and
65/100 counts, not their original summary target flags.

The final verification distributed independent B=1 evaluators across three GPUs.
It establishes learning feasibility, not single-GPU timing or an ODS efficiency
advantage. Low-data repeatability remains insufficient. Earlier parameterizations
and obsolete prototype traces are omitted from the current record; their Git
history is available at the execution revision. No success predicate was relaxed.
