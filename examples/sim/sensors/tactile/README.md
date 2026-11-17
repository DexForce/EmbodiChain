# Geometric tactile indentation

Run from the root of an installed EmbodiChain checkout with DexSim 0.6.0rc1
(or a compatible SDK), Torch, Warp, and Matplotlib. The tested path is the
Default physics backend on CPU with DexSim offscreen rendering. Install FFmpeg
with an H.264 encoder for the optional video.

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
