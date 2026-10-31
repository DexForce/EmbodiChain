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

"""Run the original Cube evaluator against the literal 50 official queries."""

from __future__ import annotations

import json
import importlib.util
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


def _native_runtime_identity(reference: Path) -> dict[str, Any]:
    """Bind measurements to the staged SDK, task code and compiled asset files."""
    spec = importlib.util.find_spec("dexsim")
    if spec is None or spec.origin is None:
        raise ImportError("native evaluation requires the staged DexSim SDK")
    package = Path(spec.origin).parent
    libraries = [
        package / "lib" / name
        for name in (
            "libengine.so",
            "libcore.so",
            "libDexSim.so",
            "libPhysXGpu_64.so",
        )
    ]
    libraries += list((package / "cuda").glob("pybind.*.so"))
    source_files = [
        package / "spawn/adapters/dexsim_adapter.py",
        package / "spawn/adapters/_urdf_contacts.py",
        package / "spawn/adapters/common.py",
        package / "scene/objects/articulation.py",
        package / "scene/batches/articulation.py",
        package / "scene/batches/_articulation_binding.py",
        package / "spawn/descs.py",
        Path(__file__),
        Path(__file__).with_name("configs") / "ogbench_cube.json",
        reference / "physical-contract.json",
        reference / "render-contract.json",
        reference / "robot/robot.urdf",
    ]
    for module_name in (
        "embodichain_tasks.manipulation.ogbench_cube",
        "embodichain.lab.sim.spawn.source",
        "embodichain.lab.sim.spawn.descriptors",
        "embodichain.lab.sim.physics.default",
        "embodichain.lab.sim.cfg.physics",
        "embodichain.lab.sim.objects.articulation",
        "embodichain.lab.sim.objects.backends.scene",
        "embodichain_models.dynamics.lewm.model",
        "embodichain_models.dynamics.lewm.planner",
    ):
        module_spec = importlib.util.find_spec(module_name)
        if module_spec is not None and module_spec.origin is not None:
            source_files.append(Path(module_spec.origin))
    return {
        "python": sys.executable,
        "torch": torch.__version__,
        "sdk_package": str(package),
        "sha256": {str(path): _sha(path) for path in libraries + source_files},
    }


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
                "pixels": pixels[:, None],
                "goal": goals[:, None],
                "action": np.zeros((50, 1, 5), dtype=np.float32),
            }
        )
        normalized = solver.solve(
            {key: value[indices] for key, value in prepared.items()}
        )["actions"]
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


def _native_worker(
    pipe: Any,
    reference: str,
    initial: dict[str, np.ndarray],
    device: str,
    images: bool,
) -> None:
    """Keep each native World alive across both MPC planning barriers."""
    import traceback
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import DefaultPhysicsCfg, RenderCfg
    from embodichain_tasks.manipulation.ogbench_cube import (
        OGBenchCubeEnv,
        OGBenchCubeCfg,
    )

    env = None
    try:
        env = OGBenchCubeEnv(
            OGBenchCubeCfg(
                reference_root=reference,
                num_envs=len(initial["qpos"]),
                enable_images=images,
                sim_cfg=SimulationManagerCfg(
                    physics_cfg=DefaultPhysicsCfg(
                        physics_dt=0.002,
                        device=device,
                        cache_material=True,
                        enable_tgs=False,
                    ),
                    device=device,
                    gpu_id=torch.device(device).index,
                    headless=True,
                    physics_dt=0.002,
                    enable_default_plane=False,
                    render_cfg=RenderCfg(renderer="auto" if images else "no-render"),
                    enable_entity_gizmo=False,
                    robot_ik_gizmo=None,
                    startup_summary="off",
                ),
            )
        )
        obs, _ = env.reset(seed=42, options=initial)
        qpos, qvel = _native_state(env)
        np.testing.assert_allclose(
            qpos[:, :17], initial["qpos"][:, :17], atol=2e-6, rtol=0
        )
        np.testing.assert_allclose(qvel, initial["qvel"], atol=2e-5, rtol=0)
        if images:
            for _ in range(32):
                with env.sim.render_frame():
                    obs = env.get_obs()

        def pixels() -> np.ndarray | None:
            if not images:
                return None
            return obs["sensor"]["main_cam"]["color"][..., :3].cpu().numpy().copy()

        pipe.send(
            {
                "qpos": qpos,
                "qvel": qvel,
                "pixels": pixels(),
                "scene_config": json.loads(
                    env.sim.physics.get_scene().dump_scene_config()
                ),
            }
        )
        frozen_qpos, frozen_qvel = qpos.copy(), qvel.copy()
        alive = np.ones(env.num_envs, dtype=bool)
        while True:
            request = pipe.recv()
            if request is None:
                break
            positions, velocities, used_actions, successes = [], [], [], []
            for step, supplied in enumerate(request["actions"]):
                if not images:
                    alive = np.isfinite(supplied).all(axis=1)
                action = np.where(alive[:, None], supplied, 0).astype(np.float32)
                used_actions.append(np.where(alive[:, None], action, np.nan))
                obs, _, _, _, info = env.step(torch.as_tensor(action, device=device))
                qpos, qvel = _native_state(env)
                frozen_qpos[alive], frozen_qvel[alive] = qpos[alive], qvel[alive]
                positions.append(frozen_qpos.copy())
                velocities.append(frozen_qvel.copy())
                success = info["success"].cpu().numpy() & alive
                successes.append(success)
                if images:
                    alive &= ~success
            pipe.send(
                {
                    "qpos": np.asarray(positions),
                    "qvel": np.asarray(velocities),
                    "actions": np.asarray(used_actions),
                    "successes": np.asarray(successes),
                    "pixels": pixels(),
                }
            )
    except BaseException:
        pipe.send({"error": traceback.format_exc()})
    finally:
        if env is not None:
            env.close()
            SimulationManager.flush_cleanup_queue()
        pipe.close()


@torch.inference_mode()
def evaluate_dexsim(
    *,
    reference: Path,
    queries: Path,
    weights: Path,
    output: Path,
    device: str,
    replay: Path | None = None,
    worker_devices: list[str] | None = None,
    reference_planner_root: Path | None = None,
) -> dict[str, Any]:
    """Use one ordered planner and persistent native Worlds of at most 25 views.

    Like the original evaluator, MPC starts from stored dataset pixels and
    uses live engine pixels at step 25. Native arenas for completed rows may
    continue stepping; their recorded results freeze at the success step.
    A supplied reference-action replay is only a physics diagnostic.
    """
    import multiprocessing
    from PIL import Image
    from embodichain_models.dynamics.lewm import (
        LeWM,
        LeWMCfg,
        LeWMCEMCfg,
        LeWMCEMPlanner,
        preprocess_rgb,
    )

    metadata = json.loads(queries.with_suffix(".json").read_text())
    if _sha(queries) != metadata["queries_sha256"] or metadata["num_eval"] != 50:
        raise ValueError("official 50-query checksum/count mismatch")
    if (
        _sha(weights)
        != "2839a907362f403f9136383016e91774373a295d958ae75121791f22a9fddf89"
    ):
        raise ValueError("official Cube checkpoint checksum mismatch")
    output.mkdir(parents=True, exist_ok=True)
    with h5py.File(queries, "r") as file:
        qpos0, qvel0 = file["qpos"][:], file["qvel"][:]
        goals, goal_pixels = (
            file["goal_privileged_block_0_pos"][:],
            file["goal_pixels"][:],
        )
        initial_pixels = file["pixels"][:]
    if replay is not None:
        with np.load(replay, allow_pickle=False) as file:
            replay_actions = file["actions"].copy()
        if replay_actions.shape != (50, 50, 5):
            raise ValueError("reference replay must contain 50 steps and 50 queries")
    devices = worker_devices or [device, device]
    if len(devices) != 2:
        raise ValueError(
            "two worker devices are required for 25+25 native environments"
        )
    context = multiprocessing.get_context("spawn")
    workers = []
    started = time.monotonic()
    positions, velocities, actions_trace, plan_sizes = [], [], [], []
    successes = np.zeros(50, dtype=bool)
    first_success = np.full(50, -1, dtype=np.int32)

    def receive(process: Any, connection: Any) -> dict:
        deadline = time.monotonic() + 300
        while not connection.poll(1):
            if not process.is_alive():
                raise RuntimeError(f"native worker exited with code {process.exitcode}")
            if time.monotonic() >= deadline:
                raise TimeoutError("native worker did not finish within 300 seconds")
        result = connection.recv()
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    try:
        for index, worker_device in enumerate(devices):
            rows = slice(index * 25, (index + 1) * 25)
            parent, child = context.Pipe()
            process = context.Process(
                target=_native_worker,
                args=(
                    child,
                    str(reference),
                    {"qpos": qpos0[rows], "qvel": qvel0[rows], "goal_pos": goals[rows]},
                    worker_device,
                    replay is None,
                ),
            )
            process.start()
            child.close()
            workers.append((process, parent))
        initial = [receive(*worker) for worker in workers]
        positions.append(np.concatenate([r["qpos"] for r in initial])[None])
        velocities.append(np.concatenate([r["qvel"] for r in initial])[None])
        if replay is None and reference_planner_root is not None:
            original_plan = _original_cube_planner(
                reference_planner_root, weights, metadata, device
            )
        elif replay is None:
            model = LeWM(LeWMCfg(action_dim=25, action_block=5)).eval().to(device)
            model.load_reference_weights(
                torch.load(weights, map_location="cpu", weights_only=True)
            )
            model.set_action_statistics(
                torch.tensor(metadata["action_mean"], device=device).repeat(5),
                torch.tensor(metadata["action_std"], device=device).repeat(5),
            )
            planner_cfg = json.loads(
                Path(__file__)
                .with_name("configs")
                .joinpath("ogbench_cube.json")
                .read_text()
            )["planner"]
            planner = LeWMCEMPlanner(
                model,
                -torch.ones(25, device=device),
                torch.ones(25, device=device),
                LeWMCEMCfg(**planner_cfg),
                seed=42,
            )
            goal = model.encode(
                preprocess_rgb(torch.as_tensor(goal_pixels, device=device)[:, None])
            )[:, 0]
        if replay is None:
            for offset, result in enumerate(initial):
                for index in (0, 10, 20):
                    Image.fromarray(result["pixels"][index]).save(
                        output / f"initial-{offset*25+index}.png"
                    )
        observed = initial_pixels
        for phase in range(2):
            if replay is None:
                pending = np.zeros((50, 25, 5), dtype=np.float32)
                active = np.flatnonzero(~successes)
                if reference_planner_root is not None:
                    plans = original_plan(observed, goal_pixels, active)
                    pending[active] = plans.reshape(-1, 25, 5)
                else:
                    history = model.encode(
                        preprocess_rgb(
                            torch.as_tensor(observed, device=device)[:, None]
                        )
                    )
                    plans = []
                    for index in active:
                        sequence, _ = planner.plan_sequence(
                            history[index : index + 1],
                            torch.empty(1, 0, 25, device=device),
                            goal[index : index + 1],
                        )
                        plans.append(sequence[0].cpu().numpy())
                        pending[index] = plans[-1].reshape(25, 5)
                np.save(
                    output / f"plan-{phase}.npy", np.asarray(plans), allow_pickle=False
                )
                plan_sizes.append(len(plans))
                supplied = pending.transpose(1, 0, 2)
            else:
                supplied = replay_actions[phase * 25 : (phase + 1) * 25]
            # Send both chunks before receiving either, retaining every World
            # and its contact/solver state across the planning barrier.
            for index, (_, connection) in enumerate(workers):
                connection.send({"actions": supplied[:, index * 25 : (index + 1) * 25]})
            results = [receive(*worker) for worker in workers]
            positions.append(np.concatenate([r["qpos"] for r in results], axis=1))
            velocities.append(np.concatenate([r["qvel"] for r in results], axis=1))
            actions_trace.append(
                np.concatenate([r["actions"] for r in results], axis=1)
            )
            phase_success = np.concatenate([r["successes"] for r in results], axis=1)
            for step, mask in enumerate(phase_success):
                first_success[mask & ~successes] = phase * 25 + step + 1
                successes |= mask
            if replay is None:
                observed = np.concatenate([r["pixels"] for r in results])
                for index in (0, 10, 20, 30, 40):
                    Image.fromarray(observed[index]).save(
                        output / f"step-{(phase+1)*25}-{index}.png"
                    )
            progress = {
                "control_steps": (phase + 1) * 25,
                "success_count": int(successes.sum()),
                "elapsed_seconds": time.monotonic() - started,
            }
            (output / "progress.json").write_text(json.dumps(progress) + "\n")
            print(progress, flush=True)
        positions, velocities = np.concatenate(positions), np.concatenate(velocities)
        np.savez_compressed(
            output / "trace.npz",
            qpos=positions,
            qvel=velocities,
            actions=np.concatenate(actions_trace),
            goals=goals,
        )
        report = {
            "kind": (
                "paired_source_action_replay"
                if replay is not None
                else "official_protocol_prototype_dexsim_50_query"
            ),
            "queries_sha256": metadata["queries_sha256"],
            "num_eval": 50,
            "eval_budget": 50,
            "success_count": int(successes.sum()),
            "success_rate": float(successes.mean() * 100),
            "episode_successes": successes.tolist(),
            "first_success_steps": first_success.tolist(),
            "final_goal_distances": np.linalg.norm(
                positions[-1, :, 14:17] - goals, axis=1
            ).tolist(),
            "initially_within_goal_threshold": metadata[
                "already_within_success_threshold"
            ],
            "initial_state_max_position_error": float(
                np.max(np.abs(positions[0, :, :17] - qpos0[:, :17]))
            ),
            "initial_state_max_velocity_error": float(
                np.max(np.abs(velocities[0] - qvel0))
            ),
            "history_len": 1,
            "action_block": 5,
            "horizon": 5,
            "receding_horizon": 5,
            "solver_batch_size": 1,
            "std_correction": 1,
            "plan_sizes": plan_sizes,
            "persistent_native_workers": 2,
            "environments_per_worker": 25,
            "worker_devices": devices,
            "planner_device": device,
            "native_material_cache": True,
            "native_scene_descriptors": [row["scene_config"] for row in initial],
            "planner_implementation": (
                "upstream" if reference_planner_root is not None else "embodichain"
            ),
            "elapsed_seconds": time.monotonic() - started,
            "remaining_differences": [
                "MuJoCo material priority for unlike pad pairs",
                "contact softness",
                "equality impedance",
                "convex cooking",
                "renderer lighting",
            ],
            "runtime_identity": _native_runtime_identity(reference),
            "dexsim_physics_alignment_verified": False,
        }
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    finally:
        for process, connection in workers:
            if process.is_alive():
                try:
                    connection.send(None)
                except (BrokenPipeError, EOFError):
                    pass
        for process, connection in workers:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
            connection.close()
