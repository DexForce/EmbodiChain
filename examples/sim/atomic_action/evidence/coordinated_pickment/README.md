# CoordinatedPickment rendered evidence

Recorded on 2026-11-05 from example revision `0889b576a` and production revision
`b08f4206d`. Native DexSim 0.5.1rc1, Default physics, FastRT rendering, dual UR5
with DH PGI hands, analytic UR IK, RTX 4090, driver 580.126.20, PyTorch
2.10.0+cu128, Python 3.11. Native simulation-time H.264, 640×480 at 20 fps.

| Run | Video | Observed data | Final behavior |
| --- | --- | --- | --- |
| Continue holding | [MP4](resume-hold.mp4) | [JSON Lines](resume-hold.jsonl) | `move`, `hold`; no reopening |
| Move and release | [MP4](resume-release.mp4) | [JSON Lines](resume-release.jsonl) | `move`, `hold`, `release`, `retreat` |

Each recording includes physical pickup, a one-second settling hold, and a
separately planned continuation through two object waypoints. The tray bottom
rises from about 0.558 m to 0.654 m. At acquisition, the sensor reports contacts
with both `left_gripper_finger2_link_1` and `right_gripper_finger1_link_1`.
The example establishes its initial observed hold once from measured object
pose and robot FK after checking elevation and recent bilateral contacts.
It then passes both attachment transforms unchanged into `resume_held=True`.
There is no grasp resampling, opening, approach or lift phase in that continuation.
No joint/object teleport or attachment constraint is used during playback.

The hold run ends 0.735 mm from the requested object position; maximum measured
EEF/object positional drift relative to the original attachment is below
0.063 mm for either hand. In the release run the maximum such drift before
opening is below 0.117 mm. Both hands open and retreat; the settled tray has
1.291 mm XY error and 5.460 mm 3D error from the pre-release target, mainly its
intended clearance settling onto the support. These are observations of this
fixture, not a qualification across robots, objects or physics backends.
See [computed summary](summary.json) and the complete per-sample data.

From a checkout of this branch, use `embodichain311` or an equivalent DexSim
environment and the matching bundled extensions:

```bash
export PYTHONPATH="$PWD/extensions:$PWD/extensions/embodichain_tasks"
python -m examples.sim.atomic_action.coordinated_pickment --headless --auto_play --renderer fast-rt --no_vis_eef_axis --resume_held --object plastic_tray --solver ur --sample_count 600 --hand_interp_steps 60 --auto_play_video_path /tmp/resume-hold.mp4
python -m examples.sim.atomic_action.coordinated_pickment --headless --auto_play --renderer fast-rt --no_vis_eef_axis --resume_held --release --object plastic_tray --solver ur --sample_count 600 --hand_interp_steps 60 --auto_play_video_path /tmp/resume-release.mp4
```

`--headless` suppresses the interactive window, while FastRT renders the video.
The example retains the original pickup defaults unless the new flags are
selected. The recording uses a slower pickup budget and the existing analytic
UR backend. The default 96-sample pickup attempt did not lift the tray and was
rejected before continuation. A slower PyTorch-IK run did lift it; its initial
contact check misread the root `path` as the link name. That example-only
reader was corrected to use `ContactActorInfo.link_name`. A subsequent
PyTorch-IK grasp search was interrupted after several minutes without a plan;
the final recordings use analytic UR IK. Failed attempts were not counted as
successful continuation evidence.

Contact acquisition now resolves actor paths from each configured hand's joint
group, including mimic fingers and fixed pads. Native topology and actor lookup
checks passed for UR5/PGI and UR10/Robotiq. The videos above remain the original
UR5 recordings; this check does not qualify UR10 physical pickup or transport.
