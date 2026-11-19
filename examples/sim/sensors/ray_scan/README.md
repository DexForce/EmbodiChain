# Geometric ray scans

Run from the root of an installed EmbodiChain checkout with DexSim 0.6.0rc2
(or a compatible SDK), Torch, Warp, and Matplotlib. This example uses the
Default physics backend on CPU with DexSim offscreen rendering. Install FFmpeg
with an H.264 encoder for the optional video.

The scene panel is a DexSim Hybrid camera capture of the sampled state.
It requires an NVIDIA GPU with native camera rendering support, even when
physics runs on CPU. The demo does not open a simulation window.

```bash
python -m examples.sim.sensors.ray_scan.demo --output-dir /tmp/terrain --video
```

The public `SensorCfg.from_dict` configuration constructs a registered
`TerrainScan`. A tilted scanner moves above a ground plane and three static
boxes. It casts a regular pad of rays along its local −Z axis at every
completed physics step. The tilted grid and 1.48 m range limit leave a visible
band of no-hit cells on the terrain image.

- `preview.png` shows the simulator geometry with orange hit points and the
  range map in metres. Gray cells have no hit.
- `preview.mp4` is produced with `--video` and follows the sequence. Playback
  time is independent of the simulation-time annotation.
- `samples.npz` stores the sample IDs, timestamps, masks, ranges, arena-frame
  hit points, shape identities and scanner positions. Ray order is X then Y,
  with Y varying fastest.

## Moving target

```bash
python -m examples.sim.sensors.ray_scan.demo --mode dynamic --output-dir /tmp/ray-motion --video
python -m examples.sim.sensors.ray_scan.demo --mode dynamic --geometry visual --output-dir /tmp/ray-visual
```

The registered `RayLidar` and `RayDepthCamera` observe the same kinematic
box at each completed step. The scene panel overlays the LiDAR's arena-frame
points. The images show axial depth along the camera's optical +Z and radial
range along each camera ray. Gray is no hit. The archive prefixes the two
sensors' outputs with `lidar_` and `depth_`.

Collision mode uses cooked collision geometry. Visual mode uses the larger
authored render box and works with no native renderer. It supports procedural
and array-backed geometry on rigid and visual objects. Moving rigid targets
can change pose between samples. Replace an asset to change cached mesh
geometry or scale.

## VLP16 preset

Install this checkout or its wheel so the `embodichain.sensors` entry point
is visible. For an editable checkout, use `python -m pip install -e .` in the
same environment as the following command.

```bash
python -m examples.sim.sensors.ray_scan.demo --mode vlp16 --output-dir /tmp/vlp16 --video
```

This mode resolves `sensor_type: VLP16` with `SensorCfg.from_dict`. It scans
360° at 1,800 azimuth samples with 16 nominal channels and a 100 m limit.
Each scan is a simultaneous geometric first-return snapshot. It has no
rolling acquisition, intensity or dual returns. The point cloud is colored by
firing-order channel and displays every fourth azimuth for clarity. The range
map and archive retain all rays. The archive also stores the channel elevations
in degrees.
