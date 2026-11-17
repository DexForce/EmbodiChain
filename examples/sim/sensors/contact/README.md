# Contact sensors

From the repository root, install EmbodiChain and its declared DexSim SDK.
Use `python -m pip install matplotlib` and install `ffmpeg` for MP4; use
`--no-video` for PNG only.

The scene panel is a DexSim Hybrid camera capture of the sampled state.
It requires an NVIDIA GPU with native camera rendering support, even when
physics runs on CPU. The demo does not open a simulation window.

```bash
python -m examples.sim.sensors.contact.run --backend newton --device cuda:0 --output /tmp/contact-forces
python -m examples.sim.sensors.contact.run --backend default --device cpu --output /tmp/contact-default
```

The first command requires a CUDA device and the SDK's Newton MuJoCo Warp
solver with contact reports. The second uses Default CPU and no GPU or
native renderer. Both run headlessly and write `preview.png` and `preview.mp4`.

A moving cube falls onto a pad. The XZ projection is drawn from the simulated
collision box pose. Beside it, the registered `ContactResultant` shows the
contact flag, component validity and measured normal and friction force
magnitudes in N. The sensor already converts native impulse using the
solver's reported integration duration. Missing measurements leave gaps in the
traces. Default CPU exposes a total impulse scalar rather than a separate
normal component, so the normal trace is marked unavailable on that backend.

## Contact grid

```bash
python -m examples.sim.sensors.contact.run --grid --backend default --device cpu --output /tmp/contact-grid
python -m examples.sim.sensors.contact.run --grid --backend newton --device cuda:0 --output /tmp/contact-grid-newton
```

The registered `ContactGrid` partitions the pad's XY footprint into 12 × 12 cells.
Its origin is attached to the top surface. The side view shows that origin and
the X and Z axes. The two heatmaps show counts and the measured scalar impulse
in N·s. White cells in the impulse heatmap indicate missing data. Default CPU
reports total impulse magnitude, while the qualified Newton provider reports
normal impulse magnitude. The report panel shows completeness, validity of
the three impulse quantities and the native impulse duration in ms.
