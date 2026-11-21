# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Regenerate successful, fixed-target CubePlanOracle episodes in DexUni."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from multiprocessing.util import Finalize
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any

import h5py
import hdf5plugin
import mujoco
import numpy as np
import torch

from dexsim.engine.newton_physics.backend_registry import get_newton_backend
from embodichain.lab.sim import SimulationManager
from .._official_eval import _native_state
from .._official_native import _make_cube, _pixels, _NATIVE_RENDER

from ._capture import _episode_splits
from ._env import _source_state_observation
from ._resources import _CubeResources

_env = _source = _output = _solver = _oracle = None
_pad = _driver = None


def _close():
    _env.close()
    SimulationManager.flush_cleanup_queue()
    _source.close()


def _setup(reference, source, output, ogbench, devices, worker_id):
    global _env, _source, _output, _solver, _oracle, _pad, _driver
    sys.path.insert(0, str(ogbench))
    from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle

    with worker_id.get_lock():
        index = worker_id.value
        worker_id.value += 1
    _env = _make_cube(reference, 1, devices[index % len(devices)])
    _source, _output = h5py.File(source, "r"), output
    _solver = get_newton_backend(_env.sim._world).mujoco_solver
    _pad = _solver.mj_model.body("arena_0_ur5e_robotiq_right_pad").id
    _driver = _env._names.index("ur5e/robotiq/right_driver_joint")
    scale = np.asarray(_env.contract["action_range"])
    view = SimpleNamespace(
        _arm_sampling_bounds=np.array([[0.25, -0.35, 0.20], [0.6, 0.35, 0.35]]),
        _control_timestep=0.05,
        normalize_action=lambda action: np.clip(action / scale, -1, 1),
    )
    _oracle = CubePlanOracle(
        env=SimpleNamespace(unwrapped=view), noise=0.1, noise_smoothing=0.5
    )
    Finalize(None, _close, exitpriority=10)


def _measure(step, goal, goal_yaw):
    qpos, qvel = _native_state(_env)
    pinch = _env._frame_pose("ur5e/robotiq/pinch").cpu()
    cube = _env.cube.get_local_pose().cpu()
    w, x, y, z = qpos[0, 17:]
    cube_yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    pin = pinch[0].numpy()
    status = {
        "proprio/effector_pos": pin[:3, 3],
        "proprio/effector_yaw": [np.arctan2(pin[1, 0], pin[0, 0])],
        "proprio/gripper_opening": [np.clip(qpos[0, _driver] / 0.8, 0, 1)],
        "privileged/target_block": 0,
        "privileged/block_0_pos": qpos[0, 14:17],
        "privileged/block_0_yaw": [cube_yaw],
        "privileged/target_block_pos": goal,
        "privileged/target_block_yaw": [goal_yaw],
        "time": [step * 0.05],
    }
    wrenches = np.zeros((1, 6)) if step == 0 else _pad_wrench()
    state = _source_state_observation(
        torch.from_numpy(qpos[:, :14]),
        torch.from_numpy(qvel[:, :14]),
        pinch,
        cube,
        torch.from_numpy(wrenches),
    ).numpy()[0]
    return status, qpos[0], qvel[0], state


def _pad_wrench():
    mujoco.mj_rnePostConstraint(_solver.mj_model, _solver.mj_data)
    return _solver.mj_data.cfrc_ext[_pad : _pad + 1].copy()


def _capture_episode(eid: int) -> tuple[int, Path | None, int | None]:
    start = int(_source["ep_offset"][eid])
    goal, yaw = _source["privileged_target_block_pos"][start], float(
        _source["privileged_target_block_yaw"][start, 0]
    )
    np.random.seed(np.random.SeedSequence([3072, eid]).generate_state(1)[0])
    reset = {
        "qpos": _source["qpos"][start],
        "qvel": _source["qvel"][start],
        "goal_pos": goal,
    }
    obs, _ = _env.reset(seed=42, options=reset)
    rgb = np.empty((201, 224, 224, 3), dtype=np.uint8)
    qpos = np.empty((201, 21), dtype=np.float64)
    qvel = np.empty((201, 20), dtype=np.float64)
    obs_state = np.empty((201, 28), dtype=np.float64)
    actions = np.empty((200, 5), dtype=np.float32)
    success = np.empty(201, dtype=bool)
    for i in range(201):
        rgb[i] = _pixels(obs)[0]
        status, qpos[i], qvel[i], obs_state[i] = _measure(i, goal, yaw)
        success[i] = np.linalg.norm(qpos[i, 14:17] - goal) <= 0.04
        if i == 200:
            break
        if i == 0 or _oracle.done:
            _oracle.reset(None, status)
        actions[i] = _oracle.select_action(None, status)
        obs, _, _, _, _ = _env.step(torch.from_numpy(actions[i : i + 1]))
    first = np.flatnonzero(success[1:])
    if not len(first) or not success[-1]:
        return eid, None, None
    path = _output / f"{eid}.h5"
    partial = path.with_suffix(".partial")
    with h5py.File(partial, "x") as file:
        ep = file.create_group(str(eid))
        ep.attrs["source_episode_id"] = ep.attrs["seed"] = eid
        ep.attrs["success"] = True
        ep.attrs["first_success_step"] = int(first[0] + 1)
        ep.attrs["target_yaw"] = yaw
        ep["target"] = goal
        ep.create_dataset(
            "pixels",
            data=rgb,
            chunks=(5, 224, 224, 3),
            **hdf5plugin.Blosc(
                cname="zstd", clevel=2, shuffle=hdf5plugin.Blosc.SHUFFLE
            ),
        )
        for name, values in (
            ("actions", actions),
            ("qpos", qpos),
            ("qvel", qvel),
            ("observation", obs_state),
            ("step_success", success),
        ):
            ep[name] = values
    partial.rename(path)
    return eid, path, int(first[0] + 1)


def collect_expert(
    resources: _CubeResources,
    output: Path,
    *,
    train_episodes: int,
    validation_episodes: int,
    gpu_ids: list[int],
    workers: int,
) -> dict[str, Any]:
    """Re-run the closed-loop expert on native physics and retain successful episodes."""
    if min(train_episodes, validation_episodes, workers) < 1 or not gpu_ids:
        raise ValueError(
            "Cube collection requires positive episodes, workers and render GPUs"
        )
    output.mkdir(parents=True)
    spool = output / "spool"
    spool.mkdir()
    split = _episode_splits(resources)
    meta = {
        "environment": "OGBenchCubeDexUni-v0",
        "render": _NATIVE_RENDER,
        "physics": {
            "device": "cpu",
            "solver": "dexuni",
            "physics_dt": 0.002,
            "substeps": 25,
        },
        "action_dim": 5,
        "step_dt": 0.05,
        "camera": "front_pixels",
        "action_semantics": "relative end-effector XYZ, yaw and gripper",
        "collector": "OGBench CubePlanOracle, fixed source target across replans",
        "noise": 0.1,
        "noise_smoothing": 0.5,
        "episode_controls": 200,
        "eval_episodes_excluded": split["excluded"],
    }
    report = {"contract": meta, "splits": {}}
    context = mp.get_context("spawn")
    start = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_setup,
        initargs=(
            resources.reference,
            resources.data,
            spool,
            resources.ogbench,
            gpu_ids,
            context.Value("i", 0),
        ),
    ) as pool:
        for name, wanted in (
            ("train", train_episodes),
            ("validation", validation_episodes),
        ):
            candidates = split[f"expert_{name}"]
            accepted, attempts = {}, []
            temporary = output / f"{name}.partial.h5"
            with h5py.File(temporary, "x") as target:
                target.attrs["format"] = "embodichain.lewm.trajectory.v1"
                target.attrs["split"] = name
                target.attrs["metadata"] = json.dumps(meta)
                group = target.create_group("episodes")
                for offset in range(0, len(candidates), 2 * workers):
                    batch = candidates[offset : offset + 2 * workers]
                    for episode, path, first in pool.map(_capture_episode, batch):
                        attempts.append(episode)
                        if path is None:
                            continue
                        if len(accepted) < wanted:
                            with h5py.File(path, "r") as source:
                                source.copy(str(episode), group)
                            accepted[episode] = first
                        path.unlink()
                    print(
                        f"{name}: retained {len(accepted)}/{wanted} from {len(attempts)} attempts",
                        flush=True,
                    )
                    if len(accepted) == wanted:
                        break
                if len(accepted) != wanted:
                    raise RuntimeError(
                        f"Only {len(accepted)} successful {name} episodes were available"
                    )
            temporary.rename(output / f"{name}.h5")
            report["splits"][name] = {"accepted": accepted, "attempts": attempts}
    spool.rmdir()
    report["seconds"] = time.monotonic() - start
    (output / "dataset.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
