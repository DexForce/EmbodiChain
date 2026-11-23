# Cube validation

## Complete example runs

Both runs used released DexSim Engine 0.6.0 on Linux x86-64 with RTX PRO
5000 Blackwell GPUs and the CUDA 12.8 Torch wheels. Starting from cached,
verified official inputs, each run prepared new assets and queries, captured
272 paired episodes, trained a new visual checkpoint, evaluated it, and
encoded the recorded native rollout.

| Runtime | Asset compiler | Held-out visual MSE | Official 50 queries |
| --- | --- | ---: | ---: |
| Python 3.11, Torch 2.7.0+cu128, Transformers 4.53.3 | MuJoCo 3.11.0 | 0.009205 | **35/50** |
| Python 3.12, Torch 2.11.0+cu128, Transformers 5.5.4 | MuJoCo 3.15.0 | 0.009267 | **33/50** |
| Archived upstream MuJoCo evaluation | Published Cube assets | — | **34/50** |

Both runtime environments used MuJoCo 3.15.0 internally for the DexUni CPU
solver. Follow the [deployment guide](DEPLOYMENT.md) for the pinned
dependencies and separate asset compiler. The Python 3.11 asset export
reproduced the archived robot URDF digest
`25088d77d3880ca6a244e162eefd68e9ad0ef1beac3d8e2bfbde6a32edc27647`.

The native adapter produces two CONNECT rows and one joint equality, with all
22 joints assigned to the rigid-body solve. A 45-control source-expert replay
on the released SDK first reached the target at control 40. The wheel was
installed in a separate directory and used from outside the source checkout
to collect the ten-episode smoke set and encode a 448×224 H.264 preview.

## Fixed-query protocol

Both new checkpoints use 8,960 training and 1,920 validation native renders
of archived poses. The published encoder supplies paired visual supervision.
The trained export uses the official dynamics and action statistics. The
checkpoint is selected by held-out visual error.

Eval uses the official seed-42 query selection, 50 controls per query,
upstream CEM and a 4 cm success threshold. EmbodiChain supplies initial,
goal and live replanning frames. In both runs the official query HDF5
SHA-256 was
`7d2d66317ab10644173308fb9a3a8eaf9a9edc32889047c89d22e54a1f9f8b8f`.
The totals were recomputed from the emitted position and action traces.
Query 12 succeeded at control 22 in the Python 3.11 reproduction video.

## Expert collection and earlier controls

The native expert pilot retained 384 training and 64 validation episodes,
totaling 90,048 RGB frames. Warm-start training with prediction and SIGReg
reached 31/50 in that pilot. The released-SDK expert smoke run collected
two successful training episodes and one successful validation episode and
completed training and export. The prior native run with the unadapted
published Cube checkpoint reached 25/50. Evaluating the earlier visual
adaptation checkpoint on the released SDK reached 35/50.

Reported counts apply to the tested runtime, weights and fixed queries.
The example starts from the published weights and uses its
[reproduction preset](configs/reproduce.json). The original corpus contains
10,000 episodes.
