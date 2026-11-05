# OpenDoor rendered evidence

Recorded on 2026-11-05 from example revision `a309511f6` and production revision
`0c0840fe3`. Native DexSim 0.5.1rc1, Default physics, FastRT, UR5/PGI,
RTX 4090, driver 580.126.20, PyTorch 2.10.0+cu128, Python 3.11.
Complete simulation-time H.264 recordings, 640×480 at 20 fps.

| Run | Video | Measured hinge data | Initial / target / settled final |
| --- | --- | --- | --- |
| Opening | [MP4](open-0-to-30.mp4) | [JSON Lines](open-0-to-30.jsonl) | 0° / 30° / 31.9006° |
| Closing | [MP4](close-30-to-0.mp4) | [JSON Lines](close-30-to-0.jsonl) | 30° / 0° / 0.001515° |

Both plans succeed and complete physical grasp, signed hinge motion, release
and retreat. The hinge is unactuated. Opening overshoots its target by about
1.90° after release and final settling; this open-loop primitive does not
verify tight physical target-angle satisfaction. Closing reaches approximately
0.0015°. See [summary](summary.json) and every measured replay sample.

The initial angle is staged once as fixture initialization, before the planning
observation. Neither planning nor playback directly writes the door state.
The affordance's calibrated `opening_direction` is unchanged for both runs.

From a checkout of this branch, use `embodichain311` or an equivalent DexSim
environment and matching bundled extensions:

```bash
export PYTHONPATH="$PWD/extensions:$PWD/extensions/embodichain_tasks"
python -m examples.sim.atomic_action.open_door --headless --auto_play --renderer fast-rt --no_vis_eef_axis --initial_angle 0 --open_angle 0.5235987756 --auto_play_video_path /tmp/door-open.mp4
python -m examples.sim.atomic_action.open_door --headless --auto_play --renderer fast-rt --no_vis_eef_axis --initial_angle 0.5235987756 --open_angle 0 --sample_count 1200 --auto_play_video_path /tmp/door-close.mp4
```

`--headless` disables the interactive window, not native rendering. Both videos
include the full trajectory and the existing 4.8 s final settling window.
The new example flags expose fixture initialization and a larger sample budget,
and its sidecar records the actual hinge angle after each command.

Limitations observed during qualification: the 60° initial-open fixture did
not resolve a feasible grasp. The 30° closing case with the original 300-sample
budget raised the motion generator's `cartesian_path_sampler must preserve the
requested path endpoints` validation; using 1200 minimum samples completed it.
These failures were not counted as successful playback. The evidence does not
qualify all hinge angles, initial robot postures, fixtures or physics backends.
