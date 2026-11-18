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
