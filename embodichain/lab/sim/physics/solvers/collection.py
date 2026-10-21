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

"""Scene-owned analytical data collection and matched NeRD rollout evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from collections import OrderedDict
import hashlib
import math
from importlib import metadata
import json
import platform
from pathlib import Path
import time
from typing import Any

import h5py
import numpy as np
import torch

from embodichain.compute.dynamics import quat_rotate
from embodichain.compute.dynamics._evaluation import (
    _error_values,
    _safe_number,
    _teacher_row_sha256,
)
from embodichain.compute.dynamics.reference import reference_task_values as _task_values
from embodichain.data_pipeline.dynamics.shards import (
    DynamicsShardWriter,
    load_manifest,
    dataset_fingerprint,
)
from embodichain.utils import configclass
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import NewtonPhysicsCfg, RenderCfg

from ._nerd_binding import NeRDBinding
from ._reference_assets import make_reference_articulation, reference_profile

__all__ = [
    "DynamicsGenerationCfg",
    "DynamicsSimulation",
    "generate_dataset",
    "evaluate_rollouts",
]


def _runtime_identity(device: torch.device) -> dict[str, Any]:
    """Capture the actual neural executor separately from stored teacher metadata."""
    return {
        "device": str(device),
        "engine": "current_newton",
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.machine()
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "state_dtype": "float32",
        "cuda_runtime": torch.version.cuda,
        "versions": {
            name: metadata.version(name)
            for name in ("embodichain", "dexsim-engine", "newton", "warp-lang", "torch")
        },
        "source_sha256": {
            name: _sha256(Path(__file__).with_name(name))
            for name in (
                "collection.py",
                "nerd.py",
                "_nerd_binding.py",
                "_reference_assets.py",
            )
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class DynamicsSimulation:
    """Own one reference scene through SimulationManager and public Scene APIs.

    Args:
        robot: Reference identity, ``cartpole`` or ``ant``.
        device: Physics/tensor device.
        num_envs: Number of independent replicated worlds.
        solver: Named Newton solver configuration.
        num_substeps: Teacher solver substeps per 1/60-second sample.
        requires_grad: Allocate functional dynamics when supported by the provider.
    """

    def __init__(
        self,
        robot: str,
        *,
        device: str = "cpu",
        num_envs: int = 1,
        solver: Mapping[str, Any] | None = None,
        num_substeps: int = 10,
        requires_grad: bool = False,
        physics_profile: str = "native",
    ) -> None:
        for name, value in (("num_envs", num_envs), ("num_substeps", num_substeps)):
            _positive_integer(name, value)
        if type(requires_grad) is not bool:
            raise TypeError("requires_grad must be boolean")
        if physics_profile not in {"native", "warp_reference"}:
            raise ValueError(
                f"Unsupported reference physics profile: {physics_profile}"
            )
        self.profile = reference_profile(robot)
        self.profile["physics_profile"] = physics_profile
        self.robot, self.device, self.num_envs = robot, torch.device(device), num_envs
        self.dt = float(self.profile["dt"])
        solver_cfg = dict(
            solver
            or (
                {
                    "solver_type": "featherstone",
                    "update_mass_matrix_interval": num_substeps,
                }
                if physics_profile == "warp_reference"
                else {
                    "solver_type": "mujoco_warp",
                    "use_mujoco_cpu": self.device.type == "cpu",
                    "separate_worlds": True,
                }
            )
        )
        neural = solver_cfg.get("solver_type") == "nerd"
        self._featherstone_teacher = solver_cfg.get("solver_type") == "featherstone"
        if self._featherstone_teacher:
            interval = solver_cfg.get("update_mass_matrix_interval", 1)
            if type(interval) is not int or interval < 1 or num_substeps % interval:
                raise ValueError(
                    "Featherstone teacher mass-matrix interval must divide num_substeps "
                    "so every reset starts with a fresh mass matrix."
                )
        if neural and num_substeps != 1:
            raise ValueError("The NeRD reference solver requires one macro substep.")
        cfg = SimulationManagerCfg(
            headless=True,
            num_envs=num_envs,
            arena_space=0.0,
            enable_ground_plane=robot == "ant" and physics_profile == "native",
            enable_entity_gizmo=False,
            render_cfg=RenderCfg(renderer="no-render"),
            startup_summary="off",
            physics_cfg=NewtonPhysicsCfg(
                device=device,
                physics_dt=self.dt,
                num_substeps=num_substeps,
                requires_grad=requires_grad,
                use_cuda_graph=False,
                solver_cfg=solver_cfg,
                collision_cfg=(
                    None
                    if neural
                    else (
                        {"update_interval": None}
                        if physics_profile == "warp_reference"
                        else {}
                    )
                ),
            ),
        )
        self.sim = SimulationManager(cfg, requires_native_renderer=False)
        self.articulation = None
        try:
            if robot == "ant" and physics_profile == "warp_reference":
                # The old ModelBuilder's shared plane has different material
                # defaults from SimulationManager's ordinary default plane.
                from dexsim.spawn import (
                    CollisionDesc,
                    GeometryDesc,
                    NewtonCollisionDesc,
                    ObjectDesc,
                    RigidBodyPhysicsDesc,
                )

                collision = CollisionDesc.from_geometry(GeometryDesc.plane(1000.0))
                collision.newton = NewtonCollisionDesc(
                    ke=1e5,
                    kd=1e3,
                    kf=1e3,
                    mu=0.5,
                    margin=1e-5,
                    gap=0.05,
                    mu_torsional=0.0,
                    mu_rolling=0.0,
                )
                self.sim.add_rigid_object_descriptor(
                    ObjectDesc(
                        name="reference_ground",
                        collisions=[collision],
                        physics=RigidBodyPhysicsDesc.static(),
                        per_env=False,
                    )
                )
            self.articulation = self.sim.add_articulation_descriptor(
                make_reference_articulation(robot, physics_profile)
            )
            self.sim.prepare()
            self.binding = NeRDBinding.from_metadata(
                dict(self.sim.spawn_result.dynamics_metadata()), robot
            )
            self.joint_ids = [
                self.articulation.joint_names.index(name)
                for name in self.binding.joint_names
            ]
            self.manifest = {**self.profile, **self.binding.metadata(self.dt)}
            self.manifest["teacher"] = {
                "solver": solver_cfg,
                "num_substeps": num_substeps,
                "physics_profile": physics_profile,
                "versions": {
                    name: metadata.version(name)
                    for name in ("newton", "warp-lang", "torch", "dexsim-engine")
                },
            }
            self.manifest["teacher_origin"] = {
                "kind": "current_newton",
                "device": str(self.device),
                "solver": solver_cfg,
                "num_substeps": num_substeps,
                "versions": self.manifest["teacher"]["versions"],
                "dt": self.dt,
                "physics_signature": self.binding.physics_signature,
                "physics_signature_schema": self.binding.metadata(self.dt)[
                    "physics_signature_schema"
                ],
            }
            self.reset()
        except Exception:
            self.close()
            raise

    def snapshot(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Read independent physical state and aligned contact features."""
        physical = self.sim.spawn_result.dynamics_snapshot().physical
        q, qd = self.binding.read(physical)
        states = {
            "q": q,
            "qd": qd,
            "root_body_q": physical["body_q"][self.binding.root_bodies],
        }
        contacts = self.binding.contacts(physical["body_q"])
        contacts.pop("contact_masks", None)  # Recompute through the shared codec.
        return states, contacts

    def reset(
        self, q: torch.Tensor | None = None, qd: torch.Tensor | None = None
    ) -> None:
        """Set a complete reference batch through public articulation state writes."""
        # Featherstone has no Scene.reset capability in the named SDK. Complete
        # articulation state writes are supported and clear both native buffers
        # and inputs below. Calls occur at macro boundaries; the mass-matrix
        # interval divides each macro step for the reference teacher.
        if not self._featherstone_teacher:
            self.sim.spawn_result.reset()
        if q is None:
            q = (
                torch.tensor(self.profile["default_q"], device=self.device)
                .expand(self.num_envs, -1)
                .clone()
            )
        if qd is None:
            qd = (
                torch.tensor(self.profile["default_qd"], device=self.device)
                .expand(self.num_envs, -1)
                .clone()
            )
        if q.shape != (self.num_envs, self.profile["q_dim"]) or qd.shape != (
            self.num_envs,
            self.profile["qd_dim"],
        ):
            raise ValueError("Reset tensors must match the complete reference batch.")
        kwargs = {"joint_ids": self.joint_ids, "clear_dynamics": True}
        if self.robot == "ant":
            kwargs.update(
                root_pose=q[:, :7],
                root_velocity=qd[:, :6],
                qpos=q[:, 7:],
                qvel=qd[:, 6:],
            )
        else:
            kwargs.update(qpos=q, qvel=qd)
        self.articulation.set_state(**kwargs)

    def step(self, effort: torch.Tensor) -> None:
        """Apply recorded effort for one complete sampling interval."""
        if effort.shape != (self.num_envs, self.profile["qd_dim"]):
            raise ValueError(
                "Effort must use full reference generalized velocity layout."
            )
        if self.robot == "ant":
            if bool(torch.any(effort[:, :6] != 0)):
                raise ValueError("Ant sampling cannot apply free-base forces.")
            effort = effort[:, 6:]
        self.articulation.set_qf(effort, joint_ids=self.joint_ids)
        self.sim.update(step=1, render_final_step=False)

    def close(self) -> None:
        """Release public child views before destroying the native World owner."""
        self.articulation = None
        sim = getattr(self, "sim", None)
        if sim is not None:
            self.sim = None
            sim.destroy(exit_process=False)
            SimulationManager.flush_cleanup_queue()

    def __enter__(self) -> DynamicsSimulation:
        """Return this scene owner for bounded context-managed execution."""
        return self

    def __exit__(self, *_args: Any) -> None:
        """Release the owned simulation when leaving the context."""
        self.close()


def _sample(
    robot: str,
    seed: int,
    horizon: int,
    motion_mode: str,
    initial_states: str,
    effort_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample one stable trajectory ID independently of collector batch size."""
    profile = reference_profile(robot)
    rng = torch.Generator(device="cpu").manual_seed(seed)
    q, qd = torch.tensor(profile["default_q"]), torch.tensor(profile["default_qd"])
    if initial_states == "sample":
        if robot == "cartpole":
            q = (2 * torch.rand(2, generator=rng) - 1) * torch.tensor([3.5, torch.pi])
            qd = (2 * torch.rand(2, generator=rng) - 1) * 10
        else:
            q[:3] = torch.tensor([-10.0, -10.0, 1.0]) + torch.rand(
                3, generator=rng
            ) * torch.tensor([20.0, 20.0, 0.3])
            angle = (2 * torch.rand((), generator=rng) - 1) * (torch.pi / 2)
            axis = 2 * torch.rand(3, generator=rng) - 1
            axis = axis / axis.norm().clamp_min(1e-12)
            q[3:7] = torch.cat(
                (axis * torch.sin(angle / 2), torch.cos(angle / 2).reshape(1))
            )
            low = torch.tensor(
                [-40.0, 30.0, -40.0, -100.0, -40.0, -100.0, -40.0, 30.0]
            ) * (torch.pi / 180)
            high = torch.tensor(
                [40.0, 100.0, 40.0, -30.0, 40.0, -30.0, 40.0, 100.0]
            ) * (torch.pi / 180)
            q[7:] = low + torch.rand(8, generator=rng) * (high - low)
            qd[:3] = quat_rotate(q[3:7], 2 * torch.rand(3, generator=rng) - 1)
            qd[3:6] = quat_rotate(
                q[3:7], (2 * torch.rand(3, generator=rng) - 1) * torch.pi
            )
            qd[6:] = (2 * torch.rand(8, generator=rng) - 1) * (2 * torch.pi)
    elif initial_states != "default":
        raise ValueError("initial_states must be sample or default.")
    efforts = torch.zeros((horizon, profile["qd_dim"]))
    if motion_mode == "random_effort":
        limits = torch.tensor(profile["effort_limits"]) * effort_scale
        sampled = (2 * torch.rand((horizon, len(limits)), generator=rng) - 1) * limits
        efforts[:, 6:] = sampled if robot == "ant" else efforts[:, 6:]
        if robot == "cartpole":
            efforts = sampled
    elif motion_mode != "passive":
        raise ValueError("motion_mode must be passive or random_effort.")
    return q, qd, efforts


def _positive_integer(name: str, value: Any, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, not {value!r}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@configclass
class DynamicsGenerationCfg:
    """Validated analytical sampling settings; paths belong to the caller."""

    robot: str = "cartpole"
    output_dir: str = "outputs/dynamics/data"
    device: str = "cpu"
    num_envs: int = 64
    trajectories: int = 10000
    horizon: int = 100
    seed: int = 0
    teacher_solver: dict[str, Any] | None = None
    num_substeps: int = 10
    trajectories_per_shard: int = 64
    resume: bool = False
    motion_mode: str = "random_effort"
    split_seed: int = 0
    initial_states: str = "sample"
    max_attempts: int | None = None
    effort_scale: float = 1.0
    physics_profile: str = "native"

    def __post_init__(self) -> None:
        for name in (
            "num_envs",
            "trajectories",
            "horizon",
            "num_substeps",
            "trajectories_per_shard",
        ):
            _positive_integer(name, getattr(self, name))
        for name in ("seed", "split_seed"):
            _positive_integer(name, getattr(self, name), 0)
        if self.max_attempts is not None:
            _positive_integer("max_attempts", self.max_attempts, self.trajectories)
        if type(self.resume) is not bool:
            raise TypeError("resume must be boolean")
        if (
            isinstance(self.effort_scale, bool)
            or not isinstance(self.effort_scale, (float, int))
            or not math.isfinite(self.effort_scale)
            or self.effort_scale < 0
        ):
            raise ValueError("effort_scale must be finite and nonnegative")
        if (
            self.robot not in {"cartpole", "ant"}
            or self.motion_mode not in {"passive", "random_effort"}
            or self.initial_states not in {"sample", "default"}
        ):
            raise ValueError("Unsupported robot, motion_mode or initial_states")
        if self.physics_profile not in {"native", "warp_reference"}:
            raise ValueError("Unsupported reference physics profile")
        if self.teacher_solver is not None and not isinstance(
            self.teacher_solver, Mapping
        ):
            raise TypeError("teacher_solver must be a mapping")
        if not isinstance(self.output_dir, (str, Path)) or not str(self.output_dir):
            raise ValueError("output_dir must be a nonempty path")


def generate_dataset(config: Mapping[str, Any]) -> dict[str, Any]:
    """Generate committed finite trajectories with bounded attempts and staging.

    Config validation happens before simulator creation. Each candidate has an
    independent seed. Resume advances only past examined candidates, including
    a final partial batch; unused padded worlds never acquire trajectory IDs.

    Args:
        config: Settings accepted by :class:`DynamicsGenerationCfg`.

    Returns:
        Actual committed trajectory/transition counts, rejected candidate count,
        rejection evidence path and elapsed wall time.
    """
    cfg = DynamicsGenerationCfg(**dict(config))
    cfg.validate()
    output = Path(cfg.output_dir)
    progress_path = output / "generation-progress.json"
    progress = (
        json.loads(progress_path.read_text())
        if cfg.resume and progress_path.exists()
        else {"next_candidate": 0, "rejected": 0, "rejections": []}
    )
    progress.setdefault("rejections", [])
    started = time.perf_counter()
    with DynamicsSimulation(
        cfg.robot,
        device=cfg.device,
        num_envs=cfg.num_envs,
        solver=cfg.teacher_solver,
        num_substeps=cfg.num_substeps,
        physics_profile=cfg.physics_profile,
    ) as host:
        manifest = dict(host.manifest)
        manifest["generation"] = {
            "horizon": cfg.horizon,
            "seed": cfg.seed,
            "motion_mode": cfg.motion_mode,
            "initial_states": cfg.initial_states,
            "effort_scale": cfg.effort_scale,
            "num_envs": cfg.num_envs,
        }
        with DynamicsShardWriter(
            output,
            manifest,
            cfg.trajectories_per_shard,
            resume=cfg.resume,
            split_seed=cfg.split_seed,
        ) as writer:
            accepted = len(writer.committed_ids)
            if accepted > cfg.trajectories:
                raise ValueError(
                    "Requested trajectory count is smaller than committed data"
                )
            first = max(
                [int(value.rsplit(":", 1)[-1]) + 1 for value in writer.committed_ids]
                or [0]
            )
            cursor = max(first, int(progress["next_candidate"]))
            maximum = (
                cfg.max_attempts
                if cfg.max_attempts is not None
                else cfg.trajectories * 10
            )
            while accepted < cfg.trajectories:
                if cursor >= maximum:
                    raise RuntimeError(
                        f"Generation attempt limit reached: {accepted}/{cfg.trajectories} valid trajectories, {progress['rejected']} rejected"
                    )
                candidates = list(range(cursor, min(cursor + cfg.num_envs, maximum)))
                samples = [
                    _sample(
                        cfg.robot,
                        cfg.seed + i,
                        cfg.horizon,
                        cfg.motion_mode,
                        cfg.initial_states,
                        cfg.effort_scale,
                    )
                    for i in candidates
                ]
                samples.extend([samples[-1]] * (cfg.num_envs - len(samples)))
                q, qd, force = [
                    torch.stack([row[i] for row in samples]).to(host.device)
                    for i in range(3)
                ]
                host.reset(q, qd)
                states_list, contacts_list = [], []
                with torch.no_grad():
                    for index in range(cfg.horizon):
                        state, contacts = host.snapshot()
                        states_list.append(state)
                        contacts_list.append(contacts)
                        host.step(force[:, index])
                    state, _ = host.snapshot()
                    states_list.append(state)
                states = {
                    key: torch.stack([value[key] for value in states_list], dim=1)
                    .cpu()
                    .numpy()
                    for key in states_list[0]
                }
                transitions = {
                    key: torch.stack([value[key] for value in contacts_list], dim=1)
                    .cpu()
                    .numpy()
                    for key in contacts_list[0]
                }
                transitions["joint_f"] = force.cpu().numpy()
                transitions["gravity_dir"] = (
                    host.binding.gravity_dir[:, None]
                    .expand(-1, cfg.horizon, -1)
                    .cpu()
                    .numpy()
                )
                for row, candidate in enumerate(candidates):
                    if accepted >= cfg.trajectories:
                        break
                    cursor = candidate + 1
                    finite = all(
                        np.isfinite(v[row]).all()
                        for v in (*states.values(), *transitions.values())
                    )
                    bounded = (
                        finite
                        and np.max(np.abs(states["q"][row])) < 1e5
                        and np.max(np.abs(states["qd"][row])) < 1e5
                    )
                    if not bounded:
                        progress["rejected"] += 1
                        progress["rejections"].append(
                            {
                                "id": f"{cfg.seed}:{candidate}",
                                "sampling_seed": cfg.seed + candidate,
                                "reason": (
                                    "nonfinite"
                                    if not finite
                                    else "state_magnitude_limit"
                                ),
                            }
                        )
                        continue
                    writer.add_trajectory(
                        f"{cfg.seed}:{candidate}",
                        {k: v[row] for k, v in states.items()},
                        {k: v[row] for k, v in transitions.items()},
                        metadata={"sampling_seed": cfg.seed + candidate},
                    )
                    accepted += 1
                writer.commit()
                progress.update(next_candidate=cursor, accepted=accepted)
                _write_json(progress_path, progress)
    result = {
        "status": "completed",
        "dataset": str(output.resolve()),
        "trajectories": accepted,
        "transitions": accepted * cfg.horizon,
        "rejected": progress["rejected"],
        "rejection_evidence": str(progress_path.resolve()),
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(output / "generation-result.json", result)
    return result


class _FixedPolicy:
    """One deterministic policy artifact, evaluated independently on each rollout."""

    def __init__(
        self, robot: str, config: Mapping[str, Any], device: torch.device
    ) -> None:
        self.robot, self.device = robot, device
        self.config = dict(config)
        self.policy = None
        self.normalizer = None
        kind = self.config.get("kind")
        self.identity = {
            "kind": kind,
            "observation_schema": (
                "cartpole-state-4-v1" if robot == "cartpole" else "ant-com-state-27-v1"
            ),
            "deterministic": True,
            "implementation": "embodichain-reference-feedback-v1",
            "effort_limits": reference_profile(robot)["effort_limits"],
            "scope": "fixed_baseline_not_paper_policy",
        }
        if kind == "pd":
            allowed = {"kind", "gains"} if robot == "cartpole" else {"kind", "kp", "kd"}
            if set(self.config) - allowed:
                raise ValueError("Unknown PD policy setting")
            if robot == "cartpole":
                gains = self.config.get("gains", [5.0, 50.0, 5.0, 20.0])
                if (
                    not isinstance(gains, Sequence)
                    or len(gains) != 4
                    or any(
                        isinstance(v, bool)
                        or not isinstance(v, (int, float))
                        or not math.isfinite(v)
                        for v in gains
                    )
                ):
                    raise ValueError(
                        "Cartpole PD gains must contain four finite numbers"
                    )
                self.gains = torch.tensor(gains, device=device)
                self.identity["gains"] = list(gains)
            else:
                kp, kd = self.config.get("kp", 20.0), self.config.get("kd", 1.0)
                if any(
                    isinstance(v, bool)
                    or not isinstance(v, (int, float))
                    or not math.isfinite(v)
                    or v < 0
                    for v in (kp, kd)
                ):
                    raise ValueError("PD kp/kd must be finite nonnegative numbers")
                self.kp, self.kd = float(kp), float(kd)
                self.identity.update(kp=self.kp, kd=self.kd)
                self.target = torch.tensor(
                    reference_profile(robot)["default_q"][7:], device=device
                )
                self.identity["target"] = self.target.detach().cpu().tolist()
        elif kind == "actor_only":
            if set(self.config) != {"kind", "config", "checkpoint"}:
                raise ValueError(
                    "actor_only policy requires kind, config and checkpoint"
                )
            from embodichain.learning.models import build_mlp_from_cfg
            from embodichain.learning.rl.policies import build_policy
            from embodichain.learning.rl.normalization import (
                RunningObservationNormalizer,
            )
            from embodichain.utils.utility import load_config

            config_path, checkpoint_path = Path(self.config["config"]), Path(
                self.config["checkpoint"]
            )
            config_hash = _sha256(config_path)
            checkpoint_hash = _sha256(checkpoint_path)
            raw = load_config(config_path)
            block = raw.get("policy", raw)
            if block.get("name") != "actor_only" or block.get("obs_groups") is not None:
                raise ValueError(
                    "Fixed-policy evaluation requires flat actor_only observations"
                )
            obs_dim, action_dim = (4, 1) if robot == "cartpole" else (27, 8)
            actor = build_mlp_from_cfg(block["actor"], obs_dim, action_dim)
            self.policy = build_policy(block, obs_dim, action_dim, device, actor=actor)
            payload = torch.load(
                checkpoint_path, map_location=device, weights_only=True
            )
            if payload.get("schema_version") != 1 or "policy" not in payload:
                raise ValueError(
                    "Expected an EmbodiChain differentiable trainer policy checkpoint"
                )
            if (
                _sha256(config_path) != config_hash
                or _sha256(checkpoint_path) != checkpoint_hash
            ):
                raise ValueError("Policy artifact changed while loading")
            if (
                raw.get("trainer", {}).get("normalize_observations", False)
                and payload.get("observation_normalizer") is None
            ):
                raise ValueError(
                    "Policy config requires checkpoint observation normalization"
                )
            self.policy.load_state_dict(payload["policy"], strict=True)
            self.policy.eval().requires_grad_(False)
            if payload.get("observation_normalizer") is not None:
                self.normalizer = RunningObservationNormalizer(obs_dim, device)
                self.normalizer.load_state_dict(payload["observation_normalizer"])
            self.identity.update(
                config_sha256=config_hash,
                checkpoint_sha256=checkpoint_hash,
                scope="embodichain_fixed_actor_not_paper_policy",
            )
        else:
            raise ValueError("policy.kind must be pd or actor_only")
        self.identity["sha256"] = hashlib.sha256(
            json.dumps(self.identity, sort_keys=True).encode()
        ).hexdigest()

    def __call__(self, q: torch.Tensor, qd: torch.Tensor) -> torch.Tensor:
        obs, _, _ = _task_values(self.robot, q, qd)
        finite = torch.isfinite(obs).all(-1)
        obs = torch.nan_to_num(obs)
        force = torch.zeros_like(qd)
        if self.policy is not None:
            from tensordict import TensorDict

            if self.normalizer is not None:
                obs = self.normalizer.normalize(obs)
            action = self.policy.get_action(
                TensorDict({"obs": obs}, batch_size=[len(q)], device=q.device),
                deterministic=True,
            )["action"].clamp(-1, 1)
            if self.robot == "cartpole":
                force[:, 0] = action[:, 0] * 1500
            else:
                force[:, 6:] = action * 400
        elif self.robot == "cartpole":
            force[:, 0] = -(obs * self.gains).sum(-1).clamp(-1500, 1500)
        else:
            force[:, 6:] = (
                self.kp * (self.target - torch.nan_to_num(q[:, 7:]))
                - self.kd * torch.nan_to_num(qd[:, 6:])
            ).clamp(-400, 400)
        return torch.where(finite[:, None], force, 0)


def _evaluation_teacher(
    config: Mapping[str, Any],
    directory: Path,
    horizon: int,
    count: int,
    batch: int,
    policy: _FixedPolicy | None,
) -> dict[str, Any]:
    """Persist every requested teacher trajectory, including invalid outcomes.

    These are evaluation traces, never a filtered training dataset. NaN/Inf arrays
    are retained as evidence; no replacement candidate can hide teacher failures.
    """
    if directory.exists():
        raise FileExistsError(f"Evaluation trace directory already exists: {directory}")
    directory.mkdir(parents=True)
    robot = config["robot"]
    seed = config.get("seed", 500)
    manifest = {
        "schema_version": 1,
        "trace_type": "evaluation_including_failures",
        "split_seed": 0,
        "shards": [],
    }
    with DynamicsSimulation(
        robot,
        device=config.get("device", "cpu"),
        num_envs=batch,
        solver=config.get("teacher_solver"),
        num_substeps=config.get("num_substeps", 10),
        physics_profile=config.get("physics_profile", "native"),
    ) as teacher:
        manifest["metadata"] = teacher.manifest
        manifest["evaluation"] = {
            "seed": seed,
            "trajectories": count,
            "horizon": horizon,
            "num_envs": batch,
            "motion_mode": config.get("motion_mode", "random_effort"),
            "policy": None if policy is None else policy.identity,
        }
        for start in range(0, count, batch):
            n = min(batch, count - start)
            samples = [
                _sample(
                    robot,
                    seed + i,
                    horizon,
                    "passive" if policy else config.get("motion_mode", "random_effort"),
                    config.get("initial_states", "sample"),
                    config.get("effort_scale", 1.0),
                )
                for i in range(start, start + n)
            ]
            samples.extend([samples[-1]] * (batch - n))
            q, qd, efforts = [
                torch.stack([s[j] for s in samples]).to(teacher.device)
                for j in range(3)
            ]
            teacher.reset(q, qd)
            q_rows, qd_rows, force_rows = [], [], []
            with torch.no_grad():
                for step in range(horizon):
                    state, _ = teacher.snapshot()
                    q_rows.append(state["q"])
                    qd_rows.append(state["qd"])
                    force = (
                        efforts[:, step]
                        if policy is None
                        else policy(state["q"], state["qd"])
                    )
                    force_rows.append(force)
                    teacher.step(force)
                state, _ = teacher.snapshot()
                q_rows.append(state["q"])
                qd_rows.append(state["qd"])
            q_array = torch.stack(q_rows, 1)[:n].cpu().numpy()
            qd_array = torch.stack(qd_rows, 1)[:n].cpu().numpy()
            force_array = torch.stack(force_rows, 1)[:n].cpu().numpy()
            filename = f'teacher-{len(manifest["shards"]):06d}.h5'
            rows = []
            with h5py.File(directory / filename, "w") as stream:
                for row in range(n):
                    group = stream.create_group(str(row))
                    group.create_dataset(
                        "states/q", data=q_array[row], compression="lzf"
                    )
                    group.create_dataset(
                        "states/qd", data=qd_array[row], compression="lzf"
                    )
                    group.create_dataset(
                        "transitions/joint_f", data=force_array[row], compression="lzf"
                    )
                    rows.append(
                        {
                            "id": f"{seed}:{start+row}",
                            "group": str(row),
                            "length": horizon,
                            "split": "test",
                            "metadata": {"sampling_seed": seed + start + row},
                        }
                    )
            manifest["shards"].append(
                {
                    "file": filename,
                    "sha256": _sha256(directory / filename),
                    "trajectories": rows,
                }
            )
            _write_json(directory / "manifest.json", manifest)
    return manifest


def _teacher_origin(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve a recorded domain, never infer official provenance from a robot name."""
    origin = metadata.get("teacher_origin")
    if isinstance(origin, Mapping) and origin.get("kind") in {
        "current_newton",
        "official_warp",
    }:
        return dict(origin)
    teacher = metadata.get("teacher")
    if (
        metadata.get("teacher_profile") == "current_newton_import_generalized_effort"
        and isinstance(teacher, Mapping)
        and "newton" in teacher.get("versions", {})
        and isinstance(teacher.get("solver"), Mapping)
    ):
        return {
            "kind": "current_newton",
            **dict(teacher),
            "dt": metadata.get("dt"),
            "physics_signature": metadata.get("physics_signature"),
            "physics_signature_schema": metadata.get("physics_signature_schema"),
        }
    raise ValueError("Teacher dataset has no verifiable recorded teacher_origin")


def _source_motion(manifest: Mapping[str, Any]) -> str | None:
    """Read the persisted sampling contract, independently of the requested label."""
    evaluation = manifest.get("evaluation", {})
    if evaluation.get("motion_mode") is not None:
        return str(evaluation["motion_mode"])
    metadata = manifest["metadata"]
    generation = metadata.get("generation", {})
    if generation.get("motion_mode") is not None:
        return str(generation["motion_mode"])
    sampling = metadata.get("sampling", {})
    if type(sampling.get("passive")) is bool:
        return "passive" if sampling["passive"] else "random_effort"
    return None


def _verify_feedback_trace(manifest: Mapping[str, Any], policy: _FixedPolicy) -> None:
    """Require an actual evaluation trace bound to the declared PD feedback."""
    if policy.identity["kind"] != "pd":
        raise ValueError("Cached fixed-policy evaluation currently supports PD only")
    evaluation = manifest.get("evaluation", {})
    if (
        manifest.get("trace_type") != "evaluation_including_failures"
        or evaluation.get("motion_mode") != "fixed_policy"
        or evaluation.get("policy") != policy.identity
        or _teacher_origin(manifest["metadata"])["kind"] != "current_newton"
    ):
        raise ValueError("Teacher trace does not verify the declared feedback policy")


@torch.no_grad()
def _verify_feedback_actions(
    policy: _FixedPolicy, q: torch.Tensor, qd: torch.Tensor, efforts: torch.Tensor
) -> None:
    """Check all recorded actions on device and synchronize once per batch."""
    matched = torch.ones((), dtype=torch.bool, device=q.device)
    for step in range(efforts.shape[1]):
        expected = policy(q[:, step].contiguous(), qd[:, step].contiguous())
        matched = matched & (expected == efforts[:, step]).all()
    if not bool(matched):
        raise ValueError("Recorded teacher actions differ from its own-state feedback")


def _teacher_batches(
    directory: Path,
    manifest: Mapping[str, Any],
    records: list[dict[str, Any]],
    batch: int,
    horizon: int,
    device: torch.device,
):
    """Read at most one batch of bounded prefixes and keep at most four files open."""
    handles: OrderedDict[str, Any] = OrderedDict()
    try:
        for start in range(0, len(records), batch):
            selected = records[start : start + batch]
            values = []
            for row in selected:
                filename = row["file"]
                if filename not in handles:
                    if len(handles) >= 4:
                        _, stream = handles.popitem(last=False)
                        stream.close()
                    handles[filename] = h5py.File(directory / filename, "r")
                handles.move_to_end(filename)
                group = handles[filename][row["group"]]
                row_values = (
                    group["states/q"][: horizon + 1],
                    group["states/qd"][: horizon + 1],
                    group["transitions/joint_f"][:horizon],
                )
                values.append(row_values)
                row["reference_sha256"] = _teacher_row_sha256(row, row_values)
                row["has_nonzero_effort"] = bool(np.any(row_values[2] != 0))
            values.extend([values[-1]] * (batch - len(values)))
            arrays = [
                torch.from_numpy(np.stack([v[i] for v in values])).to(device)
                for i in range(3)
            ]
            yield selected, *arrays
    finally:
        for stream in handles.values():
            stream.close()


def evaluate_rollouts(config: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate batched independent rollouts with complete failure evidence.

    Fixed-policy actions are recomputed from each simulator's own state. Optional
    immutable teacher_dataset replays canonical test trajectories for open-loop
    comparisons. Recorded live PD feedback traces support closed-loop evaluation
    after their policy identity and all teacher actions have been verified.
    Metrics synchronize once per batch, never once per metric or simulation step.

    Args:
        config: Robot, model export, output path and evaluation settings. Optional
            teacher_dataset selects immutable canonical traces; split defaults to
            test. Fixed-policy mode needs live current Newton feedback, or a
            recorded live PD trace with verified policy/actions. A live teacher
            supports kind pd or actor_only. Policy artifact paths
            are already resolved by the CLI. num_envs is the actual replica count
            for both teacher generation and neural replay.

    Returns:
        Horizon/time error means, explicit valid/failure counts, policy returns,
        artifact hashes and paths to per-trajectory JSONL and HDF5 evidence.
    """
    allowed = {
        "robot",
        "model",
        "device",
        "num_envs",
        "trajectories",
        "horizons",
        "seed",
        "teacher_solver",
        "num_substeps",
        "motion_mode",
        "initial_states",
        "effort_scale",
        "output",
        "policy",
        "teacher_dataset",
        "split",
        "physics_profile",
    }
    if set(config) - allowed:
        raise ValueError(f"Unknown rollout settings: {sorted(set(config)-allowed)}")
    robot = config["robot"]
    reference_profile(robot)
    count = _positive_integer("trajectories", config.get("trajectories", 2048))
    batch = _positive_integer("num_envs", config.get("num_envs", 64))
    seed = _positive_integer("seed", config.get("seed", 500), 0)
    _positive_integer("num_substeps", config.get("num_substeps", 10))
    raw_horizons = config.get("horizons", [1, 10, 100, 1000])
    if (
        not isinstance(raw_horizons, Sequence)
        or isinstance(raw_horizons, str)
        or not raw_horizons
    ):
        raise ValueError("horizons must be a nonempty sequence")
    horizons = sorted({_positive_integer("horizon", value) for value in raw_horizons})
    horizon = horizons[-1]
    motion = config.get("motion_mode", "random_effort")
    if motion not in {"passive", "random_effort", "fixed_policy"}:
        raise ValueError("Unsupported evaluation motion_mode")
    if config.get("initial_states", "sample") not in {"sample", "default"}:
        raise ValueError("Unsupported initial_states")
    scale = config.get("effort_scale", 1.0)
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not math.isfinite(scale)
        or scale < 0
    ):
        raise ValueError("effort_scale must be finite and nonnegative")
    if (motion == "fixed_policy") != ("policy" in config):
        raise ValueError(
            "fixed_policy mode requires an explicit policy configuration only in that mode"
        )
    if motion == "fixed_policy" and scale != 1.0:
        raise ValueError(
            "effort_scale applies to random_effort; configure policy gains for fixed_policy"
        )
    device = torch.device(config.get("device", "cpu"))
    policy = (
        _FixedPolicy(robot, config["policy"], device)
        if motion == "fixed_policy"
        else None
    )
    output = Path(config["output"])
    model = Path(config["model"])
    model_sha = _sha256(model / "weights.pt")
    model_manifest_sha = _sha256(model / "manifest.json")
    evidence_path = output.with_name(output.stem + "-evidence.h5")
    rows_path = output.with_name(output.stem + "-trajectories.jsonl")
    if output.exists() or evidence_path.exists() or rows_path.exists():
        raise FileExistsError(
            "Evaluation output/evidence already exists; choose a new output path"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if config.get("teacher_dataset") is None:
        teacher_dir = output.with_name(output.stem + "-teacher-data")
        dataset = _evaluation_teacher(
            config, teacher_dir, horizon, count, batch, policy
        )
        split = "test"
    else:
        teacher_dir = Path(config["teacher_dataset"])
        dataset = load_manifest(teacher_dir, verify=True)
        split = config.get("split", "test")
        if split not in {"train", "validation", "test", "all"}:
            raise ValueError("split must be train, validation, test or all")
    if dataset["metadata"].get("robot") != robot:
        raise ValueError("Teacher dataset robot differs from evaluation robot")
    records = [
        {**row, "file": shard["file"]}
        for shard in dataset["shards"]
        for row in shard["trajectories"]
        if (split == "all" or row["split"] == split) and row["length"] >= horizon
    ]
    if len(records) < count:
        raise ValueError(
            f"Teacher dataset has only {len(records)} eligible trajectories, requested {count}"
        )
    records = records[:count]
    teacher_origin = _teacher_origin(dataset["metadata"])
    source_motion = _source_motion(dataset)
    if config.get("teacher_dataset") is not None:
        if motion == "fixed_policy":
            assert policy is not None
            _verify_feedback_trace(dataset, policy)
            exported = json.loads((model / "manifest.json").read_text())
            if dataset["metadata"].get("physics_signature") != exported.get(
                "physics_signature"
            ):
                raise ValueError(
                    "Feedback teacher trace/model physics signatures differ"
                )
        elif source_motion == "fixed_policy":
            raise ValueError(
                "Fixed-policy traces cannot be labeled passive or random_effort replay"
            )
        if motion == "random_effort" and source_motion != "random_effort":
            raise ValueError(
                "Teacher dataset does not establish a random_effort sampling contract"
            )
        # A passive label is additionally checked against every selected effort
        # value below. A claimed passive source cannot override nonzero evidence.
    selected_teacher_digest = hashlib.sha256()
    metric_names = ("q_mae", "qd_mae", "base_position_error", "base_orientation_error")
    errors_sum = torch.zeros(horizon, 4, device=device, dtype=torch.float64)
    valid_count = torch.zeros(horizon, device=device, dtype=torch.long)
    teacher_nonfinite_count = torch.zeros_like(valid_count)
    neural_nonfinite_count = torch.zeros_like(valid_count)
    teacher_failure_count = torch.zeros_like(valid_count)
    neural_failure_count = torch.zeros_like(valid_count)
    horizon_sum = {
        h: torch.zeros(4, device=device, dtype=torch.float64) for h in horizons
    }
    horizon_valid = {
        h: torch.zeros((), device=device, dtype=torch.long) for h in horizons
    }
    teacher_returns = torch.zeros(horizon, device=device, dtype=torch.float64)
    neural_returns = torch.zeros_like(teacher_returns)
    with (
        DynamicsSimulation(
            robot,
            device=str(device),
            num_envs=batch,
            solver={"solver_type": "nerd", "options": {"checkpoint": str(model)}},
            num_substeps=1,
            physics_profile=config.get("physics_profile", "native"),
        ) as neural,
        h5py.File(evidence_path, "w") as evidence,
        rows_path.open("w") as row_stream,
    ):
        runtime_identity = _runtime_identity(neural.binding.device)
        if (
            _sha256(model / "weights.pt") != model_sha
            or _sha256(model / "manifest.json") != model_manifest_sha
        ):
            raise ValueError("NeRD export changed while preparing evaluation")
        for batch_index, (selected, target_q, target_qd, efforts) in enumerate(
            _teacher_batches(teacher_dir, dataset, records, batch, horizon, device)
        ):
            n = len(selected)
            if policy is not None and config.get("teacher_dataset") is not None:
                _verify_feedback_actions(policy, target_q, target_qd, efforts)
            if motion == "passive" and any(
                record["has_nonzero_effort"] for record in selected
            ):
                raise ValueError(
                    "Cannot label nonzero teacher effort trajectories as passive"
                )
            for record in selected:
                selected_teacher_digest.update(
                    bytes.fromhex(record["reference_sha256"])
                )
            if (
                target_q.shape[-1] != neural.profile["q_dim"]
                or target_qd.shape[-1] != neural.profile["qd_dim"]
                or efforts.shape[-1] != neural.profile["qd_dim"]
            ):
                raise ValueError(
                    "Teacher data does not use canonical native q/qd/effort dimensions"
                )
            neural.reset(target_q[:, 0], target_qd[:, 0])
            actual_initial, _ = neural.snapshot()
            predicted_q = [actual_initial["q"]]
            predicted_qd = [actual_initial["qd"]]
            neural_efforts = []
            with torch.no_grad():
                for step in range(horizon):
                    force = (
                        efforts[:, step]
                        if policy is None
                        else policy(predicted_q[-1], predicted_qd[-1])
                    )
                    neural_efforts.append(force)
                    neural.step(force)
                    current, _ = neural.snapshot()
                    predicted_q.append(current["q"])
                    predicted_qd.append(current["qd"])
            q = torch.stack(predicted_q, 1)[:n]
            qd = torch.stack(predicted_qd, 1)[:n]
            tq = target_q[:n]
            tqd = target_qd[:n]
            teacher_nonfinite = (
                (~(torch.isfinite(tq).all(-1) & torch.isfinite(tqd).all(-1)))
                .cummax(1)
                .values[:, 1:]
            )
            neural_nonfinite = (
                (~(torch.isfinite(q).all(-1) & torch.isfinite(qd).all(-1)))
                .cummax(1)
                .values[:, 1:]
            )
            valid = ~(teacher_nonfinite | neural_nonfinite)
            _, tr, tf = _task_values(robot, tq, tqd)
            _, nr, nf = _task_values(robot, q, qd)
            tf = (
                (
                    tf
                    | ~torch.isfinite(tr)
                    | ~torch.isfinite(tq).all(-1)
                    | ~torch.isfinite(tqd).all(-1)
                )
                .cummax(1)
                .values
            )
            nf = (
                (
                    nf
                    | ~torch.isfinite(nr)
                    | ~torch.isfinite(q).all(-1)
                    | ~torch.isfinite(qd).all(-1)
                )
                .cummax(1)
                .values
            )
            # Terminal transition reward counts once. Post-terminal continuation
            # remains in error evidence but cannot inflate a policy's return.
            tr = torch.where(
                ~tf[:, :-1] & torch.isfinite(tr[:, 1:]), tr[:, 1:], 0
            ).cumsum(1)
            nr = torch.where(
                ~nf[:, :-1] & torch.isfinite(nr[:, 1:]), nr[:, 1:], 0
            ).cumsum(1)
            error = _error_values(robot, q[:, 1:], qd[:, 1:], tq[:, 1:], tqd[:, 1:])
            error = torch.where(valid[..., None], error, 0)
            errors_sum += error.double().sum(0)
            valid_count += valid.sum(0)
            teacher_nonfinite_count += teacher_nonfinite.sum(0)
            neural_nonfinite_count += neural_nonfinite.sum(0)
            teacher_failure_count += tf[:, 1:].sum(0)
            neural_failure_count += nf[:, 1:].sum(0)
            teacher_returns += tr.double().sum(0)
            neural_returns += nr.double().sum(0)
            cumulative = error.cumsum(1)
            for h in horizons:
                mask = valid[:, h - 1]
                horizon_sum[h] += (
                    torch.where(mask[:, None], cumulative[:, h - 1] / h, 0)
                    .double()
                    .sum(0)
                )
                horizon_valid[h] += mask.sum()
            # One batched transfer of each evidence array after the simulation loop.
            arrays = {
                k: v.detach().cpu().numpy()
                for k, v in {
                    "q": q,
                    "qd": qd,
                    "joint_f": torch.stack(neural_efforts, 1)[:n],
                    "errors": error,
                    "valid": valid,
                    "teacher_nonfinite": teacher_nonfinite,
                    "neural_nonfinite": neural_nonfinite,
                    "teacher_failed": tf[:, 1:],
                    "neural_failed": nf[:, 1:],
                    "teacher_return": tr,
                    "neural_return": nr,
                }.items()
            }
            group = evidence.create_group(str(batch_index))
            for key, value in arrays.items():
                group.create_dataset(key, data=value, compression="lzf")
            for row, record in enumerate(selected):
                item = {
                    "trajectory_id": record["id"],
                    "reference_sha256": record["reference_sha256"],
                    "sampling_seed": record.get("metadata", {}).get("sampling_seed"),
                    "evidence_group": str(batch_index),
                    "evidence_row": row,
                    "horizons": {},
                }
                for h in horizons:
                    mean = (
                        arrays["errors"][row, :h].mean(0)
                        if arrays["valid"][row, h - 1]
                        else [float("nan")] * 4
                    )
                    item["horizons"][str(h)] = {
                        **{
                            key: _safe_number(value)
                            for key, value in zip(metric_names, mean)
                        },
                        "teacher_nonfinite": bool(
                            arrays["teacher_nonfinite"][row, h - 1]
                        ),
                        "neural_nonfinite": bool(
                            arrays["neural_nonfinite"][row, h - 1]
                        ),
                        "teacher_failed": bool(arrays["teacher_failed"][row, h - 1]),
                        "neural_failed": bool(arrays["neural_failed"][row, h - 1]),
                        "teacher_policy_return": _safe_number(
                            arrays["teacher_return"][row, h - 1]
                        ),
                        "policy_return": _safe_number(
                            arrays["neural_return"][row, h - 1]
                        ),
                    }
                row_stream.write(json.dumps(item, allow_nan=False) + "\n")
    curve_mean = (errors_sum / valid_count.clamp_min(1)[:, None]).cpu().numpy()
    counts = (
        torch.stack(
            (
                valid_count,
                teacher_nonfinite_count,
                neural_nonfinite_count,
                teacher_failure_count,
                neural_failure_count,
            ),
            -1,
        )
        .cpu()
        .numpy()
    )
    returns = (
        torch.stack((teacher_returns / count, neural_returns / count), -1).cpu().numpy()
    )
    curve = []
    for step in range(horizon):
        curve.append(
            {
                "step": step + 1,
                **{
                    key: (_safe_number(value) if counts[step, 0] else None)
                    for key, value in zip(metric_names, curve_mean[step])
                },
                "valid_trajectories": int(counts[step, 0]),
                "teacher_nonfinite_count": int(counts[step, 1]),
                "neural_nonfinite_count": int(counts[step, 2]),
                "teacher_failed_count": int(counts[step, 3]),
                "failed_trajectories": int(counts[step, 4]),
                "teacher_nonfinite_rate": float(counts[step, 1] / count),
                "nonfinite_rate": float(counts[step, 2] / count),
                "teacher_failure_rate": float(counts[step, 3] / count),
                "failure_rate": float(counts[step, 4] / count),
                "teacher_policy_return": float(returns[step, 0]),
                "policy_return": float(returns[step, 1]),
            }
        )
    metrics = {}
    for h in horizons:
        number = int(horizon_valid[h].cpu())
        mean = (horizon_sum[h] / max(number, 1)).cpu().tolist()
        metrics[str(h)] = {
            **curve[h - 1],
            **{
                key: (_safe_number(value) if number else None)
                for key, value in zip(metric_names, mean)
            },
            "valid_trajectories": number,
            "numerically_invalid_trajectories": count - number,
        }
    result = {
        "status": "completed",
        "scope": (
            "current_newton_closed_loop_not_paper_policy"
            if policy
            else "immutable_teacher_open_loop_not_official_training_qualification"
        ),
        "robot": robot,
        "motion_mode": motion,
        "seed": seed,
        "model": str(model.resolve()),
        "model_sha256": model_sha,
        "model_manifest_sha256": model_manifest_sha,
        "teacher_dataset": str(teacher_dir.resolve()),
        "teacher_dataset_fingerprint": dataset_fingerprint(dataset),
        "teacher_metadata": dataset["metadata"],
        "teacher_origin": teacher_origin,
        "teacher_source_motion_mode": source_motion,
        "teacher_feedback_verification": (
            "all_selected_actions_verified_against_teacher_states"
            if policy is not None and config.get("teacher_dataset") is not None
            else "live_feedback_execution" if policy is not None else None
        ),
        "selected_teacher_sha256": selected_teacher_digest.hexdigest(),
        "selected_teacher_hash_schema": "ordered-teacher-content-v1",
        "teacher_split": split,
        "trajectories": count,
        "num_envs": batch,
        "state_dtype": "float32",
        "runtime_identity": runtime_identity,
        "policy": None if policy is None else policy.identity,
        "policy_sha256": None if policy is None else policy.identity["sha256"],
        "horizons": metrics,
        "error_vs_time": curve,
        "trajectory_evidence": str(rows_path.resolve()),
        "trajectory_evidence_sha256": _sha256(rows_path),
        "trace_evidence": str(evidence_path.resolve()),
        "trace_evidence_sha256": _sha256(evidence_path),
        "failure_rate": curve[-1]["failure_rate"],
        "nonfinite_rate": curve[-1]["nonfinite_rate"],
        "teacher_failure_rate": curve[-1]["teacher_failure_rate"],
        "teacher_nonfinite_rate": curve[-1]["teacher_nonfinite_rate"],
        "policy_return": curve[-1]["policy_return"] if policy else None,
        "teacher_policy_return": curve[-1]["teacher_policy_return"] if policy else None,
        "metric_population": "Error means condition on no numerical failure through that horizon; physical failures remain included and separately counted. Policy returns stop after first task/numerical failure.",
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(output, result)
    return result
