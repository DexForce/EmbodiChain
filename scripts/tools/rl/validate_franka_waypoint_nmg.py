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

"""Validate Franka waypoint environment and APG parity against NMG PR #6."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
for _local_path in (_REPOSITORY_ROOT, _REPOSITORY_ROOT / "embodichain_tasks"):
    if str(_local_path) not in sys.path:
        sys.path.insert(0, str(_local_path))

from embodichain.learning.rl.gradients import clip_batched_gradient_norm
from embodichain.learning.rl.models import WaypointTransformerActor
from embodichain.learning.rl.normalization import RunningObservationNormalizer
from embodichain_tasks.manipulation.franka_waypoint import FrankaWaypointNMGEnv

_DEFAULT_REFERENCE_ROOT = _REPOSITORY_ROOT.parent / "neural_motion_generator"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the local Franka waypoint port with neural_motion_generator "
            "PR #6 on generated tasks, one-step gradients, and one full APG update."
        )
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=_DEFAULT_REFERENCE_ROOT,
        help="Checkout containing neural_motion_generator PR #6.",
    )
    parser.add_argument("--device", default="cpu", help="Torch/Newton device.")
    parser.add_argument(
        "--atol",
        type=float,
        default=1.0e-7,
        help="Maximum accepted absolute tensor error.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report path.",
    )
    return parser.parse_args()


def _load_reference_modules(reference_root: Path) -> dict[str, Any]:
    if not (reference_root / "envs/franka_waypoint_env.py").is_file():
        raise FileNotFoundError(
            f"Reference checkout does not contain PR #6 sources: {reference_root}"
        )
    sys.path.insert(0, str(reference_root))
    from algo.agent import RunningObsNormalizer, WaypointTransformerActor
    from algo.apg import _clip_batched_grad_norm
    from envs.franka_waypoint_env import FrankaWaypointReachAPGEnv

    return {
        "actor": WaypointTransformerActor,
        "clip": _clip_batched_grad_norm,
        "env": FrankaWaypointReachAPGEnv,
        "normalizer": RunningObsNormalizer,
    }


def _max_abs_error(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    if reference.dtype == torch.bool or not (
        reference.is_floating_point() or reference.is_complex()
    ):
        return 0.0 if torch.equal(reference, candidate) else float("inf")
    return float((reference - candidate).abs().max())


def _record_tensor_error(
    report: dict[str, float],
    name: str,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    atol: float,
) -> None:
    error = _max_abs_error(reference, candidate)
    report[name] = error
    if error > atol:
        raise AssertionError(f"{name} max absolute error {error} exceeds {atol}")


def _environment_kwargs(
    *,
    device: str,
    waypoint_space: str,
    num_waypoints: int = 2,
) -> dict[str, Any]:
    return {
        "num_envs": 2,
        "device": device,
        "headless": True,
        "num_waypoints": num_waypoints,
        "waypoint_min_num_waypoints": 1,
        "waypoint_fixed_num_waypoints": num_waypoints,
        "waypoint_steps_per_waypoint": 30,
        "max_episode_steps": num_waypoints * 30,
        "waypoint_pos_threshold": 0.01,
        "waypoint_pos_weight": 0.0,
        "waypoint_rot_threshold": 0.1,
        "waypoint_rot_weight": 0.1,
        "waypoint_rot_precision_weight": 0.02,
        "waypoint_pose_constraint_weight": 0.002,
        "waypoint_pose_constraint_aggregation": "max",
        "waypoint_pose_feasibility_weight": 0.0,
        "waypoint_pose_feasibility_beta": 4.0,
        "waypoint_pose_violation_weight": 0.0,
        "waypoint_pose_violation_beta": 4.0,
        "waypoint_space": waypoint_space,
        "waypoint_joint_weight": 0.1,
        "waypoint_joint_threshold": 0.02,
        "waypoint_joint_exp_scale": 0.7,
        "waypoint_joint_dense_peak": 0.2,
        "waypoint_joint_precision_exp_scale": 0.01,
        "waypoint_joint_precision_dense_peak": 0.1,
        "waypoint_joint_fraction": 0.5,
        "waypoint_distance_bucket_lowers": (0.25, 1.0, 2.0, 4.0, 8.0),
        "waypoint_joint_limit_margin": 0.05,
        "waypoint_sampling_max_retries": 64,
        "waypoint_se3_translation_range": (0.03, 0.20),
        "waypoint_se3_rotation_range": (0.15, 1.50),
        "waypoint_se3_ik_iterations": 24,
        "waypoint_se3_ik_max_retries": 10,
        "waypoint_intermediate_orientation": True,
        "waypoint_bonus": 1.0,
        "waypoint_use_relative_obs": True,
        "canonicalize_quat_obs": True,
    }


def _compare_environment(
    reference_env_type: type,
    *,
    device: str,
    waypoint_space: str,
    atol: float,
) -> dict[str, float]:
    kwargs = _environment_kwargs(device=device, waypoint_space=waypoint_space)
    reference_env = reference_env_type(**kwargs)
    candidate_env = FrankaWaypointNMGEnv(**kwargs)
    report: dict[str, float] = {}
    try:
        reference_observation, _ = reference_env.reset(seed=2026)
        candidate_observation, _ = candidate_env.reset(seed=2026)
        _record_tensor_error(
            report,
            "generated_observation",
            reference_observation,
            candidate_observation,
            atol=atol,
        )

        reference_tasks = reference_env.export_task_batch()
        candidate_tasks = candidate_env.export_task_batch()
        for name in sorted(reference_tasks):
            _record_tensor_error(
                report,
                f"generated_task/{name}",
                reference_tasks[name],
                candidate_tasks[name],
                atol=atol,
            )

        reference_env.set_fixed_eval_tasks(reference_tasks, waypoint_count=2)
        candidate_env.set_fixed_eval_tasks(reference_tasks, waypoint_count=2)
        reference_observation, _ = reference_env.reset(seed=919)
        candidate_observation, _ = candidate_env.reset(seed=919)
        _record_tensor_error(
            report,
            "fixed_observation",
            reference_observation,
            candidate_observation,
            atol=atol,
        )

        base_action = torch.linspace(
            -0.25,
            0.25,
            14,
            device=torch.device(device),
        ).reshape(2, 7)
        reference_action = base_action.clone().requires_grad_(True)
        candidate_action = base_action.clone().requires_grad_(True)
        reference_step = reference_env.step(reference_action)
        candidate_step = candidate_env.step(candidate_action)
        for name, reference_value, candidate_value in zip(
            ("next_observation", "reward", "terminated", "truncated"),
            reference_step[:4],
            candidate_step[:4],
        ):
            _record_tensor_error(
                report,
                name,
                reference_value,
                candidate_value,
                atol=atol,
            )
        reference_gradient = torch.autograd.grad(
            reference_step[1].sum(),
            reference_action,
        )[0]
        candidate_gradient = torch.autograd.grad(
            candidate_step[1].sum(),
            candidate_action,
        )[0]
        _record_tensor_error(
            report,
            "action_gradient",
            reference_gradient,
            candidate_gradient,
            atol=atol,
        )
        for name in (
            "success",
            "waypoints_reached",
            "final_distance",
            "final_rot_distance",
            "final_joint_distance",
        ):
            _record_tensor_error(
                report,
                f"info/{name}",
                reference_step[4][name],
                candidate_step[4][name],
                atol=atol,
            )
        return report
    finally:
        reference_env.detach_state()
        candidate_env.detach_state()
        reference_env.close()
        candidate_env.close()


def _run_full_update(
    *,
    env: Any,
    actor: torch.nn.Module,
    normalizer: Any,
    fixed_tasks: dict[str, torch.Tensor],
    clip_action_gradient: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    env.set_fixed_eval_tasks(fixed_tasks, waypoint_count=1)
    observation, _ = env.reset(seed=919)
    optimizer = torch.optim.Adam(actor.parameters(), lr=2.5e-4, eps=1.0e-5)
    alive = torch.ones(env.num_envs, dtype=torch.bool, device=device)
    rewards: list[torch.Tensor] = []
    observations: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    for _ in range(30):
        observations.append(observation[alive].detach())
        action = actor(normalizer.normalize(observation)).clamp(-1.0, 1.0)
        action.register_hook(clip_action_gradient)
        actions.append(action.detach())
        observation, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward * alive.to(reward.dtype))
        alive &= ~(terminated | truncated)

    loss = -torch.stack(rewards).sum(dim=0).mean()
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
    optimizer.step()
    nonempty_observations = [value for value in observations if value.numel() > 0]
    if nonempty_observations:
        normalizer.update(torch.cat(nonempty_observations))
    env.detach_state()
    return {
        "actions": torch.stack(actions),
        "gradient_norm": gradient_norm.detach(),
        "loss": loss.detach(),
        "normalizer": normalizer,
        "parameters": actor.state_dict(),
    }


def _compare_full_apg_update(
    reference: dict[str, Any],
    *,
    device: str,
    atol: float,
) -> dict[str, float]:
    kwargs = _environment_kwargs(
        device=device,
        waypoint_space="cartesian",
        num_waypoints=1,
    )
    reference_env = reference["env"](**kwargs)
    candidate_env = FrankaWaypointNMGEnv(**kwargs)
    torch_device = torch.device(device)
    try:
        reference_env.reset(seed=73)
        tasks = reference_env.export_task_batch()
        observation_dim = int(reference_env.single_observation_space.shape[0])

        torch.manual_seed(101)
        reference_actor = reference["actor"](
            observation_dim,
            7,
            1,
            True,
            128,
            4,
            2,
            None,
        ).to(torch_device)
        torch.manual_seed(101)
        candidate_actor = WaypointTransformerActor(
            observation_dim,
            7,
            1,
            joint_dim=7,
            use_relative_observations=True,
            hidden_dim=128,
            num_attention_heads=4,
            num_layers=2,
        ).to(torch_device)
        if list(reference_actor.state_dict()) != list(candidate_actor.state_dict()):
            raise AssertionError("Actor state-dict layouts differ")

        reference_normalizer = reference["normalizer"](
            observation_dim,
            torch_device,
            reference_env.obs_normalize_mask,
        )
        candidate_normalizer = RunningObservationNormalizer(
            observation_dim,
            torch_device,
            candidate_env.observation_normalize_mask,
        )
        reference_result = _run_full_update(
            env=reference_env,
            actor=reference_actor,
            normalizer=reference_normalizer,
            fixed_tasks=tasks,
            clip_action_gradient=lambda gradient: reference["clip"](
                gradient,
                1.0,
            ),
            device=torch_device,
        )
        candidate_result = _run_full_update(
            env=candidate_env,
            actor=candidate_actor,
            normalizer=candidate_normalizer,
            fixed_tasks=tasks,
            clip_action_gradient=lambda gradient: clip_batched_gradient_norm(
                gradient,
                1.0,
            ),
            device=torch_device,
        )

        report: dict[str, float] = {}
        for name in ("actions", "loss", "gradient_norm"):
            _record_tensor_error(
                report,
                name,
                reference_result[name],
                candidate_result[name],
                atol=atol,
            )
        for name in reference_result["parameters"]:
            _record_tensor_error(
                report,
                f"parameter/{name}",
                reference_result["parameters"][name],
                candidate_result["parameters"][name],
                atol=atol,
            )
        _record_tensor_error(
            report,
            "normalizer/mean",
            reference_normalizer.mean,
            candidate_normalizer.mean,
            atol=atol,
        )
        _record_tensor_error(
            report,
            "normalizer/var",
            reference_normalizer.var,
            candidate_normalizer.var,
            atol=atol,
        )
        count_error = abs(
            float(reference_normalizer.count) - float(candidate_normalizer.count)
        )
        report["normalizer/count"] = count_error
        if count_error > atol:
            raise AssertionError(
                f"normalizer/count absolute error {count_error} exceeds {atol}"
            )
        report["reference_loss"] = float(reference_result["loss"])
        report["reference_gradient_norm"] = float(reference_result["gradient_norm"])
        return report
    finally:
        reference_env.close()
        candidate_env.close()


def main() -> None:
    args = _parse_args()
    if args.atol < 0.0:
        raise ValueError("--atol must be non-negative")
    reference = _load_reference_modules(args.reference_root.resolve())
    report: dict[str, Any] = {
        "reference_root": str(args.reference_root.resolve()),
        "device": args.device,
        "atol": args.atol,
        "environment": {},
    }
    for waypoint_space in ("joint", "cartesian", "mixed"):
        report["environment"][waypoint_space] = _compare_environment(
            reference["env"],
            device=args.device,
            waypoint_space=waypoint_space,
            atol=args.atol,
        )
    report["full_apg_update"] = _compare_full_apg_update(
        reference,
        device=args.device,
        atol=args.atol,
    )
    report["status"] = "pass"
    serialized = json.dumps(report, indent=2, sort_keys=True)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
