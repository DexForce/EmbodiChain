(task-program-validation)=

# Quickstart validation record

This records the scope of the first-user walkthrough, rather than qualifying
all Task Program tasks, robots, seeds or physics backends.

## Environment and scope

| Item | Validated value |
|---|---|
| Date | 2026-10-17 |
| SDK baseline | `a8226e143343ed13fae0cdc4e71bf754ce364178` (`0.3.1` metadata), installed from a built wheel |
| CLI | `embodichain run-env`; `run-task` exists as an alias at this baseline |
| Runtime | Python 3.11.14, Linux, DexSim `0.5.1rc1`, Warp `1.17.0`, Newton dependency `1.6.0` |
| GPU | NVIDIA RTX PRO 5000 72 GB Blackwell; driver `580.126.20`; PyTorch `2.7.0+cu128`; one GPU selected with `CUDA_VISIBLE_DEVICES=1` |
| Deployment | `embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.franka.yaml` |
| Physics / execution | Default; `trajectory_open_loop.yaml`; `effect_assurance: projected` |
| Selection | Seed 7, one environment, one episode, no expansion |
| User directory | Empty directory outside the repository; installed package deployment plus local program overrides |

The isolated Python environment inherited other SDK dependencies from the host;
this was an installed-wheel, outside-checkout usage check, not a clean-room
installation audit. Required robot assets were already cached on the host. The existing development DexSim installation failed during
import. Installing the pinned wheel and matching Warp/Newton dependencies in an
isolated environment resolved that failure without modifying the shared install.

## Launcher migration

The current walkthrough uses `embodichain run-task`, the sole execution entry
point after main commit `db821682` (#793). The historical physical runs above
used `run-env` at baseline `a8226e14`; their measurements and screenshots are
retained unchanged. Synchronizing the documentation changes the command spelling
and launcher-guide links, not the tested program, deployment or physics policy.

## Observed results

| Case | Result |
|---|---|
| Discover/inspect | `list-task` and `show-task` resolve the installed official task. |
| Original | Six segments, 240 steps, A → B → A; command execution completed with saving disabled. |
| Two moves | Four segments, 160 steps, B → A; command execution completed with saving disabled. |
| Checked without saving | Four segments completed, including two successful settling/position checks. |
| Checked/save | Four segments, 188 frames, 7.52 s; two settling checks and two 3 cm position checks passed; one episode saved. |
| Repeat checked/save with the ordinary CLI | One episode, 188 frames; both position/stability checks passed again without the capture harness. |
| Dataset reader | `preview_lerobot_data` reads the episode, task/subtask language and four segment ranges; consistency check passes. |
| Unknown object | `reference_validation_failed` at `$.program.body.items[0].steps.call.object`; exit 1 before environment construction. |
| Unsupported field | `unknown_field` at `$.program.speed`; exit 1 before environment construction. |

In the first checked run, the Place segments each took 54 steps including
settling. Measured position errors were `0.022820491 m` at B and
`0.016308872 m` at A; the threshold remained `0.03 m`. Saved sidecars contain
`completed: true`, `success: true`, the per-segment validation results and
settling diagnostics. These observations establish the stated position and
stability conditions. Projected Pick/Place effects remain unverified by contact
or release monitors.

The original program was also rerun with a temporary capture harness calling
`SimulationManager.start_window_record()` and `stop_window_record()`, using
simulation-time capture at 20 FPS and one fixed viewpoint. The documentation's
three JPEGs are compressed frames 1, 31 and 191 from that original-program video.
No scene objects, targets, policy or robot configuration were changed for capture.
Raw logs, episode metadata, videos and the capture harness are retained with the
implementation's validation artifacts; the documentation only ships the three
compressed frames and the two program downloads.

## Maintainer checks

These commands require a **source checkout** and are separate from the installed
SDK walkthrough. The existing read-only inspector uses production component
resolution, loader, catalog preflight and lowering, without constructing a
simulator:

```bash
python .agents/skills/add-task-program/scripts/inspect_deployment.py \
  extensions/embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.franka.yaml \
  --json
```

For each downloadable variant, validation used a temporary copy of that deployment
with absolute component paths and only `task_program.program` replaced. This
exercises the same integration, physical UID and contract checks. It is not an
additional task configuration that users need to maintain.

The documentation build and public API/context checks are recorded in the PR.
No new runtime API, configuration field, tutorial-specific test or production
behavior is introduced by these pages. Future changes to the example, robot,
policy, physics dependencies or validator semantics require repeating the
physical checks, not just rebuilding HTML.
