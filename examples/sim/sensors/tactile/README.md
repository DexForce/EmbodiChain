# Geometric tactile indentation

Run from the root of an installed EmbodiChain checkout with Torch, Warp, and
Matplotlib. The tested path uses the Default physics backend on CPU with DexSim
offscreen rendering. Install FFmpeg with an H.264 encoder for the optional
video.

The scene panel is a DexSim Hybrid camera capture of the sampled state.
It requires an NVIDIA GPU with native camera rendering support, even when
physics runs on CPU. The demo does not open a simulation window.

```bash
python -m examples.sim.sensors.tactile.demo --pad curved --output-dir /tmp/tactile --video
python -m examples.sim.sensors.tactile.demo --pad planar --output-dir /tmp/tactile-planar
```

The example constructs the registered `TactileIndentation` sensor through
`SensorCfg.from_dict`. The kinematic sphere loads the pad, slides in both
directions, then releases. Probe origins and normals describe the undeformed
pad. The ray target is a closed mesh with consistently outward winding. Probe
depth is the outward exit distance from inside that mesh, clipped to 0.012 m.

- `preview.png` shows the simulated tool pose and the same completed sample's depth
  in metres. Gray cells indicate no contact.
- `preview.mp4` is produced with `--video` and follows the sequence from first
  miss to release. Playback time is independent of physics time.
- `samples.npz` stores depth, hit validity, shape IDs, the tool position and
  sample metadata. The first dimension is the completed sample index. The
  depth grid is ordered by local Y then X.

## Tactile RGB

```bash
python -m examples.sim.sensors.tactile.demo --rgb --output-dir /tmp/tactile-rgb --video
```

The scene, depth, and RGB panels share one sample ID and simulation timestamp.
RGB is a normalized, first-order optical model on the same pad grid. The
triples supplied by `--background`, `--x-response`, and `--y-response` set the
no-contact color and the three channel gains for depth slopes along increasing
probe columns and rows. The defaults are illustrative settings. Calibrate all
three triples for a physical pad before comparing its images. The sphere
releases at the end and every pixel returns to the same background used by a
reset row. The archive includes the normalized `rgb` array in Y/X/channel order.

## Shear and marker trajectories

```bash
python -m examples.sim.sensors.tactile.demo --pad planar --rgb --shear --markers --output-dir /tmp/tactile-shear --video
```

`--shear` enables the geometric elastic-penalty model and overlays the
sampled displacement on the RGB or depth image. Add `--markers` to use the
public sparse marker displacement instead of subsampling the dense field.
Arrows represent elastic displacement in metres at 3× display scale. They
use the local tangent along mount +X projected onto the probe plane and the
second tangent from normal × first tangent. These are pad X/Y on planar pads.

The bottom panel plots sums over probes in newtons during loading, forward
and reverse sliding, and release. The signed traces are sums of local
components. On curved pads, different probes have different tangent frames.
The dashed line identifies the time of the shared scene/depth/RGB sample.
Shear stiffness, normal stiffness, and Coulomb friction are illustrative
geometric-model parameters. For physical use, calibrate them for the pad.

## Newton contact surface

Use a CUDA-enabled DexSim SDK that exposes
`Scene.get_hydroelastic_contact_capabilities()` and
`Scene.get_hydroelastic_contact_surface()`. Use the Newton MuJoCo Warp solver
with `use_mujoco_contacts=False`, collision `update_interval=1`, and
`output_contact_surface=True`. Install `trimesh`
alongside the plotting dependencies above.

```bash
python -m examples.sim.sensors.tactile.demo --surface --device cuda:0 \
  --frames 33 --video --output-dir /tmp/tactile-surface
```

The tool loads, slides across, and releases a planar SDF pad. Each frame shows
the tool position, geometric depth (m), face count per probe cell, and signed
separation (m) from the same completed sample. The footer and `samples.npz`
record `surface_complete`, collision generation and substep. An empty
complete snapshot is labeled “Complete · no faces”. Missing or truncated
snapshots mask the surface panels. The other demo modes use only geometric
queries and run without the completed-surface SDK.

The reference preview was recorded with the DexSim 0.6.0 completed-surface
candidate on Python 3.11.
