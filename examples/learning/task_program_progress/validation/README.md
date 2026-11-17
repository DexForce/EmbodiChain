# Native Task Program reward acceptance

The control targets were collected from the calibrated planning Task Program and replayed through ordinary `JointPositionAction` terms in a configuration-owned Task Program reward environment. No policy model or optimizer was constructed. The `execution: policy` setting declares external action ownership.

| Case | Control steps | Weighted stage events (row 0) | Outcome sum (row 0) | Maximum scalar error |
|---|---:|---|---:|---:|
| success | 892 | [0.5, 0.5] | 10 | 3.2782555e-07 |
| hold | 64 | [0.0, 0.0] | 0 | 1.70985e-12 |
| truncated | 600 | [0.5, 0.0] | 0 | 7.2196126e-08 |
| failure | 1 | [0.0, 0.0] | -10 | 0 |

All four cases passed across 1,558 native env.step calls and 3,116 environment reward samples, including one post-reset transition. Repeated-read error was zero; maximum discounted-shaping identity error was 1.4607122e-08.

Each milestone pays 0.5 once; accepted task success pays 10 once. Failure pays -10. A time limit retains the successor potential and pays no success/failure bonus. True termination applies the absorbing-potential correction, so the final total reward includes both event bonuses and a negative shaping term.

The second environment holds its initial joint targets. It receives no milestone or success events; the failure fixture deliberately fails both rows. Resetting only row 0 preserves row 1’s next reward, with absolute error 9.8225428e-13.

The scalar oracle independently evaluates reward composition using measured skill potentials. It is not an independent geometry or skill-potential-formula oracle. All actual, expected and repeated values are checked for finiteness.

[Annotated native reward preview](task_program_rewards.mp4) displays the actual reward ledger. The last frame is held for approximately one second. [Machine-readable results](results.json) retain controller/configuration identities and numerical results. Full per-step artifacts are generated locally rather than committed.

Reproduce with the commands in the [example guide](../README.md#task-program-reward-acceptance-without-training). CPU physics and the project simulator dependencies are required; use a native display or Xvfb for the preview.
