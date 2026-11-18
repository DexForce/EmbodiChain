# State and frame sensors

From the repository root, with EmbodiChain and its declared DexSim SDK installed:

The scene panel is a DexSim Hybrid camera capture of the sampled state.
It requires an NVIDIA GPU with native camera rendering support, even when
physics runs on CPU. The demo does not open a simulation window.

```bash
python -m pip install matplotlib
python -m examples.sim.sensors.state.run --output /tmp/sensor-state
```

The example uses the Default physics backend on CPU and needs no window or native
renderer. Install `ffmpeg` for MP4 export or pass `--no-video` for PNG only.

The built-in `JointEncoder` and `FrameTransform` are constructed through
`SensorCfg.from_dict` and read after each completed physics step. The same revolute joint
spins in opposite directions in the two arenas, then reverses halfway through. The preview shows both
encoder traces and the pose of the tip frame relative to the base in arena 0.
The dashed RGB axes mark the base, and the solid axes mark the tip, with
red, green and blue denoting X, Y and Z. A vertical cursor shows which simulated
sample supplies the displayed frames.

The command writes `preview.png` and `preview.mp4` to the selected directory.
