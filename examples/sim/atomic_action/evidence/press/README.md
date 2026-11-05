# Press rendered evidence

Recorded on 2026-11-05 from example revision `dfff4d0af` and production revision
`1c596e01a`. Native DexSim 0.5.1rc1, Default physics, FastRT rendering, UR5/PGI,
RTX 4090, driver 580.126.20, PyTorch 2.10.0+cu128, Python 3.11.
Videos are native simulation-time H.264 recordings at 640×480 and 20 fps.
No physical state is teleported during playback.

| Run | Video | Observations | Planned dwell |
| --- | --- | --- | --- |
| Zero dwell | [MP4](dwell-0s.mp4) | [JSON](dwell-0s.json) | absent |
| One-second dwell | [MP4](dwell-1s.mp4) | [JSON](dwell-1s.json) | 100 × 10 ms |

The positive run has `press` ending at sample 541, `dwell` at indices
542–641, and `retract` starting at 642. Measured EEF position drift from the
first dwell observation is at most 0.653 mm. The measured button joint
is at 0.006 m during the dwell. This demonstrates a pose hold, not force regulation.
The independent zero/positive plans total 7.84 s / 8.86 s before final settling;
separate planning runs differ by two contact-path samples. The dwell itself is
exactly 1.00 s. The existing CPU contract test checks exact insertion against
the same zero-dwell motion path.

From a checkout of this branch, use the `embodichain311` environment (or an
equivalent DexSim environment) and the matching bundled extensions:

```bash
export PYTHONPATH="$PWD/extensions:$PWD/extensions/embodichain_tasks"
python -m examples.sim.atomic_action.press --headless --auto_play --renderer fast-rt --no_vis_eef_axis --dwell_duration 0 --auto_play_video_path /tmp/press-zero.mp4
python -m examples.sim.atomic_action.press --headless --auto_play --renderer fast-rt --no_vis_eef_axis --dwell_duration 1 --auto_play_video_path /tmp/press-one-second.mp4
```

`--headless` disables the interactive window; FastRT still renders all recorded
frames. Both runs report planning success and complete physical playback.
The JSON files contain per-phase measured EEF, gripper and articulation states.
The video shows the complete approach, press, dwell (when requested), retract,
and final settling; no phase is cut out.
