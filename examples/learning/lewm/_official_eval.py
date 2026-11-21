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

"""The pinned upstream Cube evaluator and shared planner and state helpers."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any

import h5py
import numpy as np
import torch

from ._official_cube import CUBE_SWM_REV, OGBENCH_REV, _revision
from ._official_data import HDF5_SHA256
from ._reference import _load_reference_model, _sha


def evaluate_reference(
    *,
    swm_root: Path,
    ogbench_root: Path,
    data: Path,
    queries: Path,
    weights: Path,
    output: Path,
    device: str,
) -> dict[str, Any]:
    """Use upstream World, policy, CEM and MuJoCo without changing their loop."""
    _revision(swm_root, CUBE_SWM_REV)
    _revision(ogbench_root, OGBENCH_REV)
    if (
        _sha(weights)
        != "2839a907362f403f9136383016e91774373a295d958ae75121791f22a9fddf89"
    ):
        raise ValueError("official Cube checkpoint checksum mismatch")
    metadata = json.loads(queries.with_suffix(".json").read_text())
    if _sha(queries) != metadata["queries_sha256"] or metadata["num_eval"] != 50:
        raise ValueError("official 50-query checksum/count mismatch")
    if metadata["source_sha256"] != HDF5_SHA256 or _sha(data) != HDF5_SHA256:
        raise ValueError("official Cube corpus checksum mismatch")
    sys.path[:0] = [str(swm_root), str(ogbench_root)]
    import stable_worldmodel as swm
    from sklearn.preprocessing import StandardScaler
    from torchvision.transforms import v2

    output.mkdir(parents=True, exist_ok=True)
    model_dir = swm_root / "stable_worldmodel/wm/lewm"
    model = (
        _load_reference_model(
            model_dir / "lewm.py", model_dir / "module.py", weights, 25
        )
        .to(device)
        .requires_grad_(False)
    )
    moments = StandardScaler()
    moments.mean_ = np.asarray(metadata["action_mean"], dtype=np.float64)
    moments.scale_ = np.asarray(metadata["action_std"], dtype=np.float64)
    moments.var_ = moments.scale_**2
    moments.n_features_in_ = 5
    moments.n_samples_seen_ = metadata["scaler_rows"]
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            v2.Resize((224, 224)),
        ]
    )
    # Load real 26-frame source chunks through the original dataset class.
    dataset = swm.data.HDF5Dataset(path=data, keys_to_cache=["action"])
    world = swm.World(
        env_name="swm/OGBCube-v0",
        num_envs=50,
        image_shape=(224, 224),
        max_episode_steps=100,
        env_type="single",
        ob_type="states",
        multiview=False,
        width=224,
        height=224,
        visualize_info=False,
        terminate_at_goal=True,
    )
    states, velocities, actions, plans = [], [], [], []
    started = time.monotonic()

    def record_state() -> None:
        states.append(np.stack([e.unwrapped.data.qpos.copy() for e in world.envs.envs]))
        velocities.append(
            np.stack([e.unwrapped.data.qvel.copy() for e in world.envs.envs])
        )

    class RecordingCEM(swm.solver.CEMSolver):
        def solve(self, info_dict: dict, init_action=None) -> dict:
            result = super().solve(info_dict, init_action)
            plans.append(result["actions"].detach().cpu().numpy().copy())
            return result

    class RecordingPolicy(swm.policy.WorldModelPolicy):
        def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
            record_state()
            action = super().get_action(info_dict, **kwargs)
            actions.append(action.copy())
            progress = {
                "completed_control_steps": len(actions) - 1,
                "planning_calls": len(plans),
                "elapsed_seconds": time.monotonic() - started,
            }
            (output / "progress.json").write_text(json.dumps(progress) + "\n")
            print(progress, flush=True)
            return action

    solver = RecordingCEM(
        model=model,
        batch_size=1,
        num_samples=300,
        var_scale=1,
        n_steps=30,
        topk=30,
        device=device,
        seed=42,
    )
    policy = RecordingPolicy(
        solver=solver,
        config=swm.PlanConfig(horizon=5, receding_horizon=5, action_block=5),
        process={"action": moments},
        transform={"pixels": transform, "goal": transform},
    )
    world.set_policy(policy)
    callables = [
        {
            "method": "set_state",
            "args": {"qpos": {"value": "qpos"}, "qvel": {"value": "qvel"}},
        },
        {
            "method": "set_target_pos",
            "args": {
                "cube_id": {"value": 0, "in_dataset": False},
                "target_pos": {"value": "goal_privileged_block_0_pos"},
                "target_quat": {"value": "goal_privileged_block_0_quat"},
            },
        },
    ]
    try:
        result = world.evaluate(
            dataset=dataset,
            episodes_idx=metadata["episode_ids"],
            start_steps=metadata["start_steps"],
            goal_offset=25,
            eval_budget=50,
            callables=callables,
        )
        record_state()
        with h5py.File(queries, "r") as file:
            goals = file["goal_privileged_block_0_pos"][:]
            np.testing.assert_allclose(states[0], file["qpos"][:], atol=0, rtol=0)
            np.testing.assert_allclose(velocities[0], file["qvel"][:], atol=0, rtol=0)
        report = {
            "kind": "official_cube_50_query_mujoco_reference",
            "upstream_loop": "World.evaluate / WorldModelPolicy / CEMSolver",
            "queries_sha256": metadata["queries_sha256"],
            "stable_worldmodel_revision": CUBE_SWM_REV,
            "ogbench_revision": OGBENCH_REV,
            "num_eval": 50,
            "eval_budget": 50,
            "horizon": 5,
            "history_len": 1,
            "receding_horizon": 5,
            "action_block": 5,
            "solver_batch_size": 1,
            "std_correction": 1,
            "control_steps": len(actions),
            "planning_calls": len(plans),
            "success_count": int(result["episode_successes"].sum()),
            "success_rate": result["success_rate"],
            "episode_successes": result["episode_successes"].tolist(),
            "final_goal_distances": np.linalg.norm(
                states[-1][:, 14:17] - goals, axis=1
            ).tolist(),
            "initially_within_goal_threshold": metadata[
                "already_within_success_threshold"
            ],
            "elapsed_seconds": time.monotonic() - started,
            "dexsim_physics_alignment_verified": False,
        }
        np.savez_compressed(
            output / "trace.npz",
            qpos=np.asarray(states),
            qvel=np.asarray(velocities),
            actions=np.asarray(actions),
            goals=goals,
        )
        for index, plan in enumerate(plans):
            np.save(output / f"plan-{index}.npy", plan, allow_pickle=False)
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    finally:
        for wrapped in world.envs.envs:
            env = wrapped.unwrapped
            if env._renderer is not None:
                env._renderer.close()
                env._renderer = None
        world.close()
        if dataset.h5_file is not None:
            dataset.h5_file.close()


def _original_cube_planner(
    swm_root: Path, weights: Path, metadata: dict, device: str
) -> Any:
    """Reuse the original preprocessing, model and solver for controlled comparisons."""
    _revision(swm_root, CUBE_SWM_REV)
    sys.path.insert(0, str(swm_root))
    import gymnasium as gym
    import stable_worldmodel as swm
    from sklearn.preprocessing import StandardScaler
    from torchvision.transforms import v2

    source = swm_root / "stable_worldmodel/wm/lewm"
    model = (
        _load_reference_model(source / "lewm.py", source / "module.py", weights, 25)
        .to(device)
        .requires_grad_(False)
    )
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(metadata["action_mean"])
    scaler.scale_ = np.asarray(metadata["action_std"])
    scaler.n_features_in_ = 5
    scaler.n_samples_seen_ = metadata["scaler_rows"]
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            v2.Resize((224, 224)),
        ]
    )
    solver = swm.solver.CEMSolver(
        model=model,
        batch_size=1,
        num_samples=300,
        var_scale=1,
        n_steps=30,
        topk=30,
        device=device,
        seed=42,
    )
    config = swm.PlanConfig(horizon=5, receding_horizon=5, action_block=5)
    solver.configure(
        action_space=gym.spaces.Box(-1, 1, (50, 5)), n_envs=50, config=config
    )
    processor = swm.policy.WorldModelPolicy(
        solver=solver,
        config=config,
        transform={"pixels": transform, "goal": transform},
    )

    def plan(pixels: np.ndarray, goals: np.ndarray, indices: np.ndarray) -> np.ndarray:
        prepared = processor._prepare_info(
            {
                "pixels": pixels[indices, None],
                "goal": goals[indices, None],
                "action": np.zeros((len(indices), 1, 5), dtype=np.float32),
            }
        )
        normalized = solver.solve(prepared)["actions"]
        return scaler.inverse_transform(
            normalized.cpu().numpy().reshape(-1, 5)
        ).reshape(-1, 5, 25)

    return plan


def _native_state(env: Any) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial.transform import Rotation

    cube = env.cube.get_local_pose().cpu().numpy()
    position = env.robot.get_qpos()[:, env._joint_ids].cpu().numpy()
    velocity = env.robot.get_qvel()[:, env._joint_ids].cpu().numpy()
    linear = env.cube.body_data.lin_vel.cpu().numpy()
    angular = env.cube.body_data.ang_vel.cpu().numpy()
    angular_local = Rotation.from_quat(cube[:, 3:]).inv().apply(angular)
    return (
        np.c_[position, cube[:, :3], np.roll(cube[:, 3:], 1, axis=1)],
        np.c_[velocity, linear, angular_local],
    )
