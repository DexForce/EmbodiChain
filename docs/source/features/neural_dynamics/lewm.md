# LeWM visual world model

**LeWorldModel (LeWM)** learns action-conditioned dynamics in a visual latent
space. An encoder maps RGB observations to embeddings, and a causal predictor
rolls those embeddings forward under proposed actions. Goal-conditioned
planning ranks action sequences by the distance between their final predicted
embeddings and the encoded goal image. [LeWorldModel paper](https://arxiv.org/abs/2603.19312)

## Model interface

`preprocess_rgb` accepts channels-last uint8 RGB or RGBA images.
`LeWM.encode` takes preprocessed RGB histories and returns latents of shape
`(B, H, D)`. Here `B` is the batch size, `H` is the number of observed
frames and `D` is the latent width.

`LeWM.rollout` takes those latents, the `H - 1` executed model actions
between observations, and candidate future actions of shape `(B, S, T, A)`.
It returns predicted future latents of shape `(B, S, T, D)` for `S` proposals
and `T` future model steps. Each model action can contain a block of raw
controller actions. `LeWMCEMPlanner` uses the cross-entropy method (CEM) to
select action sequences without gradients. [LeWM API](../../api_reference/lewm.rst)

## Training and Cube workflow

The shared SL recipe combines next-embedding prediction with SIGReg
regularization. It reads HDF5 episodes containing `T + 1` RGB frames and
`T` actions, and samples windows within each episode. The exported weights
carry the model configuration and action statistics.

The Cube example constructs a DexUni environment and renders the initial,
goal and replanning images through EmbodiChain's camera. It prepares pinned
inputs, captures paired images, fits a visual checkpoint, evaluates the fixed
50 OGBench queries and exports a rollout video. The paired recipe adapts
the visual representation using the published encoder's features at the
same poses. The predictor and action statistics come from the published Cube
weights. Setup and complete commands are in the
[Cube reproduction guide](https://github.com/DexForce/EmbodiChain/blob/main/examples/learning/lewm/cube/README.md).

```bash
embodichain run-example learning/lewm/cube --help
```

## Recorded Cube rollout

```{raw} html
<figure>
  <video controls playsinline preload="metadata" style="width: 100%; border-radius: 12px;"
         aria-label="LeWM Cube rollout, with the live camera at left and goal image at right">
    <source src="../../_static/lewm/cube-query-12.mp4" type="video/mp4">
    <a href="../../_static/lewm/cube-query-12.mp4">Download the LeWM Cube rollout MP4.</a>
  </video>
  <figcaption>Query 12 reaches the goal at control step 22. Live native RGB is on the left, and the native goal image is on the right.</figcaption>
</figure>
```

The recorded Python 3.11 reproduction run succeeded on **35/50** queries.
The separately tested Python 3.12 profile reached **33/50**, and the archived
upstream MuJoCo reference reached **34/50**. The
[validation record](https://github.com/DexForce/EmbodiChain/blob/main/examples/learning/lewm/cube/VALIDATION.md)
lists the runtime profiles and success criteria.
