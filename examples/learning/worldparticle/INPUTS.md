# WorldParticle input and coordinate contract

The public loader accepts an `options` mapping. It loads fixed inputs once;
the callable receives only the current world-space state and total acceleration
at each step. See [README.md](README.md) for complete examples and
[the runtime requirements](README.md#runtime-requirements) for the CUDA environment.

## Loader options

| Key | Required | Value |
| --- | --- | --- |
| `checkpoint` | Yes | Existing local Lightning checkpoint with `model.*` tensors |
| `config` | Yes | Matching local YAML with a supported `data` task, finite positive `timestep`, finite three-component `gravity` and `verbose: false` |
| `inputs` | Yes | Local NPZ with the fixed fields below; original authored particle order |
| `model_to_world` | No | Proper orthonormal 3×3 rotation R, determinant +1; identity by default |

Unknown option keys are rejected. Paths are resolved by the caller, not relative
to the YAML. Config task IDs differ from some demo names:

| Demo | Config `data` | Checkpoint directory | Source sample |
| --- | --- | --- | --- |
| Fancy Fluid | `fancy_fluid` | `checkpoints/fancy_fluid` | `data/fancy_fluid/66/66.npz` |
| Cloth on Ball | `cloth_on_ball` | `checkpoints/cloth_on_ball` | `data/cloth_on_ball/10/10.npz` |
| Duck forward deformation (`inverse_design`) | `duck_drop` | `checkpoints/inverse_design` | `data/inverse_design/0/0.npz` |
| Deep Fluid | `fluid` | `checkpoints/deep_lagrange_dataset` | `data/deep_lagrange_demo/initial.npz` |

## Fixed NPZ fields

Let N be the particle count, M the boundary-point count and K the total number
of authored topology-neighbor entries. Use real finite numeric arrays for
physical fields, retaining the sample's original feature precision and order.
The loader converts normalized features and boundary arrays to float32.

| Field | Shape/type | Meaning |
| --- | --- | --- |
| `feats` | `(N, C)`, N > 0 | Raw task-specific features; not pre-normalized |
| `box` | `(M, 3)`, M > 0 | Fixed boundary sample positions in the model frame |
| `box_feats` | `(M, 3)` | Matching authored boundary normal/features, in the model frame |
| `neighbors_index` | `(K,)`, integer | Zero-based particle indices in `[0, N)` |
| `neighbors_row_splits` | `(N+1,)`, integer | CSR offsets: first 0, nondecreasing, last K |

The last two fields are required when the matching config enables topology
convolution; the published Cloth and Duck profiles use them. They are not
recomputed from current positions. Keep the original graph and particle order.
The loader checks CSR shape/range/monotonicity, including unsigned inputs;
it cannot prove geometry identity or detect a valid but wrong permutation.

Raw feature handling follows the original task loaders:

| Config task | Raw C | Transform used for inference |
| --- | ---: | --- |
| `fancy_fluid` | 3 | Normalize column 2 by its sample mean/std; the demo uses columns 0/1 as mass/density for the native carrier |
| `cloth_on_ball` | 7 | Select mass in column 0, Young modulus in column 5 and density in column 6; sample-standardize mass, log-scale Young modulus between 5e3 and 3e5, linearly scale density between 50 and 500 |
| `duck_drop` | 1 | Preserve the authored scalar feature |
| `fluid` | 2 | Preserve both authored feature columns |

Cloth's other four raw columns are not network conditioning in this adapter.
Mean/std normalization is over the N supplied particles, using population std
plus 1e-8. The Young-modulus/density transforms do not clamp to their reference
intervals. These preprocessing rules do not qualify arbitrary material values
or new geometry for a checkpoint.
Do not normalize features twice, reinterpret Duck's feature as native SPH mass,
or infer new material semantics from feature width alone.

`load_model` does not read `position` or `velocity` from the NPZ. The demo/public
`data.load_sample` additionally expects both fields as `(T, N, 3)` arrays.
They provide the initial frame and optional reference-only metrics. Task stride
is applied first, followed by the configured frame slice. Cloth starts at frame
10; Duck uses stride 5. Subsequent predictions are free rollout, not replay of
the stored trajectory. Deep Fluid has generated initial conditions and no
official reference trajectory.

## Runtime tensors and frames

Call `next_q, next_qd = model(q, qd, acceleration, dt)` with float32 CUDA tensors
of shape `(N, 3)` on the loaded device and in the same particle order:

- `q`: world position in metres; `qd`: world velocity in metres/second.
- `acceleration`: total world acceleration in metres/second², including gravity.
  Direct callers supply acceleration, not force. The native SDK adapter computes
  `particle_force * particle_inv_mass + gravity`; do not add gravity twice.
- `dt`: seconds, equal to the checkpoint timestep. It is not an arbitrary
  integrator setting or a resampling request.
- Outputs are float32 world position/velocity arrays with the same `(N, 3)` layout.

For row-vector coordinates, `q_world = q_model @ R.T`; the model internally
uses `q_model = q_world @ R`. Boundary `box`/`box_feats` stay in the model frame.
Native gravity must be `R @ config_gravity`. Fancy Fluid, Cloth and Deep Fluid
use the demo's y-up to z-up rotation:

```python
R = [[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]
```

Duck uses identity. `load_sample` returns model-frame initial state; its caller
performs this world conversion before invoking the model or native builder.
World/frame transport remains FP32 under ambient autocast; the frozen network
uses BF16. The caller owns creation/use on its execution stream. Real-network
bitwise determinism and native/standalone numerical parity are not established.

## Native scene admission and lifecycle

The Provider accepts one nonempty CUDA particle world, with no native rigid
bodies, joints or collision shapes, `requires_grad=False`, matching gravity/dt,
positive initial inverse masses and initially active particles. Cloth uses the
cloth carrier; the other demos use an unconnected fluid-particle carrier. Static
network boundaries are not native collision shapes or dynamic rigid coupling.
The Provider checks particle count; identity/order remain caller obligations.

Use `num_substeps=1`, disabled CUDA Graph capture and no native collision pipeline
as in the quickstart. Native reset restores the scene's physical baseline; it
does not replace features/topology, boundaries, weights or rotation. Close and
recreate the model/scene when those fixed inputs change. The standalone model is
stateless: the caller owns current q/qd and resets by supplying the initial
tensors again; repeated BF16 evaluation need not be bitwise identical.
