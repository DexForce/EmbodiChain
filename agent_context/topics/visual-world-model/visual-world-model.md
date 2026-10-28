# Visual world model

`embodichain_models.dynamics.lewm` owns the image encoder, action-conditioned
latent predictor, SIGReg objective, HDF5 window source, weights-only export and
bounded CEM planning. It has no simulator dependency. The installable
`build_lewm_trainer` factory uses common `learning.sl` orchestration and
rejects multi-process training: rank-local SIGReg statistics do not implement a
global batch objective. `examples/learning/lewm/` owns pinned reference
comparison, UR push-cube collection and paired simulator evaluation.

Episodes contain T+1 RGB frames and T raw actions; each action leaves the
corresponding frame. Windows stay within episodes. The source identity binds
the file content, split and window shape. Training/validation seeds must be
disjoint. Action statistics are computed only from training episodes and saved
with model weights; exported metadata binds the camera, control period and
action contract. Rollout receives H observed latents, H-1 executed actions and
strictly future candidate actions. The model does not produce decoded pixels
or physical state.

The example keeps reference weights and simulator datasets outside the wheel.
Upstream numerical equivalence, held-out latent error and paired physical
success are different validation claims; see its README for commands and
resource revisions. Focused model tests live in `tests/models/dynamics/lewm/`.
