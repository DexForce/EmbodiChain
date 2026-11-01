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

"""Offline configured expert collection over one batched simulation World."""

from __future__ import annotations
from collections.abc import Mapping
from dataclasses import asdict
from contextlib import ExitStack
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from embodichain.data_pipeline.recording.media import MediaSpool, _atomic_json
from embodichain.lab.expansion.configured import (
    ExpansionRunCfg,
    preflight_expansion_job,
)
from embodichain.lab.expansion.contracts import EpisodeCandidate
from embodichain.lab.expansion.persistence import ArtifactEpisodeSink
from embodichain.lab.expansion.program import (
    collect_program_batch,
    collect_vector_program,
)
from embodichain.lab.sim.motion.expansion import (
    ExpansionSession,
    TrajectoryExpansionJobCfg,
    SceneCase,
    ExpertEpisode,
    ValidationCheck,
    ValidationResult,
)

__all__ = ["run_expansion_job", "inspect_expansion_job"]


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str, allow_nan=False).encode()
    ).hexdigest()


def _images(
    observation: object, row: int, prefix: tuple[str, ...] = ()
) -> dict[str, np.ndarray]:
    frames = {}
    if not hasattr(observation, "items"):
        return frames
    for key, value in observation.items():
        name = prefix + (
            (str(key),) if not isinstance(key, tuple) else tuple(map(str, key))
        )
        if isinstance(value, torch.Tensor):
            if value.ndim >= 3 and name[-1].split("_", 1)[0] in {
                "rgb",
                "depth",
                "color",
                "mask",
                "normal",
                "position",
                "disparity",
            }:
                frames["/".join(name)] = value[row].detach().cpu().numpy()
        else:
            frames.update(_images(value, row, name))
    return frames


def inspect_expansion_job(directory: str | Path) -> dict[str, object]:
    """Verify persisted receipts and report job progress without simulation."""
    root = Path(directory)
    job = json.loads((root / "job.json").read_text())
    receipts = ArtifactEpisodeSink(root).inspect()
    attempts = [
        json.loads(path.read_text())
        for path in sorted((root / "attempts").glob("*/attempt.json"))
    ]
    result = {
        "job_id": job["job_id"],
        "committed": len(receipts),
        "attempts": len(attempts),
        "rollout_attempted": sum(
            item.get("steps", item.get("executed_steps_lower_bound", 0)) > 0
            or item.get("status") == "running"
            for item in attempts
        ),
        "measured_successes": sum(
            bool(item.get("measured_accepted", item.get("status") == "committed"))
            for item in attempts
        ),
        "target_episodes": job["target_episodes"],
        "complete": len(receipts) >= job["target_episodes"],
        "outcomes": {
            status: sum(item.get("status") == status for item in attempts)
            for status in {item.get("status") for item in attempts}
        },
    }
    result["collection_complete"] = result["complete"]
    if job["config"].get("export_lerobot", False):
        result["complete"] = False
        if (root / "lerobot-export.json").exists():
            from .export import export_expansion_lerobot

            result["lerobot"] = export_expansion_lerobot(None, root)
            result["complete"] = result["collection_complete"]
    requested = job["config"].get("render_profiles", [])
    if requested:
        from .visual import inspect_observation_variants

        variants = inspect_observation_variants(root)
        expected = len(receipts) * len(requested)
        result["observation_variants"] = len(variants)
        result["complete"] = result["complete"] and len(variants) == expected
    return result


def run_expansion_job(
    cfg: ExpansionRunCfg, *, resume: bool = False
) -> dict[str, object]:
    """Collect qualified episodes with bounded media and durable per-attempt records.

    Args:
        cfg: Closed job configuration referencing an existing deployment.
        resume: Verify prior output and continue only uncommitted work.
    Returns:
        Verified progress report. Exhausting attempts does not imply completion.
    """
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg
    from embodichain.lab.gym.utils.registration import REGISTERED_ENVS
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.sim.scene import SceneView, capture_state
    from embodichain.lab.sim.scene.physics import (
        capture_physics_properties,
        capture_rigid_physics,
        apply_rigid_physics,
    )
    from embodichain.utils.utility import load_config

    report = preflight_expansion_job(cfg)
    root = Path(cfg.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".owner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = root / "job.json"
        if manifest.exists():
            if not resume:
                raise FileExistsError("job output already exists; use resume")
            if json.loads(manifest.read_text())["job_id"] != report["job_id"]:
                raise ValueError("job configuration or source content changed")
        elif resume:
            raise FileNotFoundError("resume requires an existing job manifest")
        else:
            _atomic_json(manifest, {**report, "config": cfg.to_dict()})
        source_artifacts = {
            field: {
                "path": str(Path(getattr(cfg, field)).resolve()),
                "sha256": checksum,
            }
            for field, checksum in report["input_files"].items()
        }
        source_artifacts["job"] = {
            "path": str(manifest.resolve()),
            "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        }
        sink = ArtifactEpisodeSink(root)
        if resume:
            verified = {receipt.commit_id: receipt for receipt in sink.inspect()}
            for attempt_path in sorted((root / "attempts").glob("*/attempt.json")):
                record = json.loads(attempt_path.read_text())
                receipt = verified.get(record["commit_id"])
                if receipt is None:
                    receipt = sink.recover(record["commit_id"])
                if receipt is None and record["status"] == "pending_write":
                    raise RuntimeError("pending write has lost its staged evidence")
                if receipt is None and record["status"] in {"prepared", "running"}:
                    record["executed_steps_lower_bound"] = (
                        1 if record["status"] == "running" else 0
                    )
                    record["status"] = "interrupted"
                    _atomic_json(attempt_path, record)
                if receipt is not None:
                    record["status"] = "committed"
                    record["storage_id"] = receipt.storage_id
                    _atomic_json(attempt_path, record)
        committed = len(sink.inspect())
        if committed >= cfg.target_episodes:
            status = inspect_expansion_job(root)
            if status["complete"]:
                return status
        attempts_root = root / "attempts"
        attempts_root.mkdir(exist_ok=True)
        previous = sorted(attempts_root.glob("*/attempt.json"))
        next_ordinal = (
            max(
                (json.loads(path.read_text())["ordinal"] for path in previous),
                default=-1,
            )
            + 1
        )
        # Source artifacts and camera buffers belong to this run, never normal reset-time writers.
        path = Path(cfg.deployment)
        raw = load_config(path)
        environment_cfg = config_to_cfg(
            raw, source_path=path, task_program_path_override=cfg.program
        )
        environment_cfg.num_envs = environment_cfg.sim_cfg.num_envs = cfg.batch_size
        environment_cfg.seed = cfg.seed
        environment_cfg.dataset = None
        environment_cfg.init_rollout_buffer = False
        environment_cfg.record_trajectory = False
        environment_cfg.sim_cfg.headless = True
        if not cfg.capture_media:
            environment_cfg.enable_sensor = False
            environment_cfg.sim_cfg.render_cfg.renderer = "no-render"
        else:
            environment_cfg.enable_sensor = True
            if environment_cfg.sim_cfg.render_cfg.renderer == "no-render":
                environment_cfg.sim_cfg.render_cfg.renderer = "hybrid"
        from embodichain.lab.sim.profiler import ProfilerCfg

        environment_cfg.profiler = ProfilerCfg(
            warmup_steps=0, output_path=str(root / "profile.json")
        )
        env = None
        collision = None
        appearance = None
        visual_registry = None
        if cfg.visual_profile_file is not None:
            from .visual import load_visual_profiles

            visual_registry = load_visual_profiles(cfg.visual_profile_file)
        started = time.monotonic()
        timings = {
            "prepare": 0.0,
            "execute": 0.0,
            "commit": 0.0,
            "render_variants": 0.0,
            "export": 0.0,
        }
        try:
            env = REGISTERED_ENVS[raw["id"]].make(cfg=environment_cfg)
            timings["startup"] = time.monotonic() - started
            if getattr(
                env, "event_manager", None
            ) is not None and env.event_manager.active_functors.get("interval"):
                raise ValueError(
                    "offline expansion requires fixed episode parameters; interval events are unsupported"
                )
            view = SceneView(env.sim, env_ids=tuple(range(cfg.batch_size)))
            managed = tuple(sorted(set(cfg.mass_scale) | set(cfg.friction)))
            baseline = capture_rigid_physics(view, managed) if managed else None
            auxiliary_baseline = (
                capture_physics_properties(view)
                if (
                    cfg.center_of_mass
                    or cfg.joint_stiffness_scale
                    or cfg.joint_damping_scale
                )
                else None
            )
            if cfg.collision_profile is not None:
                from .collision import _MeasuredCollision

                collision = _MeasuredCollision(
                    env, cfg.collision_profile, root / "collision"
                )
            profile = None
            if cfg.source == "real2sim":
                from embodichain.lab.task_program.integrations.simulation.real2sim_collection import (
                    load_real2sim_tabletop_profile,
                )

                profile = load_real2sim_tabletop_profile(cfg.source_profile)
            scene_source = None
            if cfg.scene_atlas is not None:
                from .scene import JointSceneSource

                scene_source = JointSceneSource(cfg.scene_atlas)
            motion_profile = None
            motion_recipes = ()
            if cfg.motion_profile is not None:
                from embodichain.lab.sim.motion.expansion.profile import (
                    load_expansion_profile,
                )
                from embodichain.lab.sim.motion.expansion.combined import (
                    enumerate_candidate_recipes,
                )

                motion_profile = load_expansion_profile(cfg.motion_profile)
                motion_recipes = enumerate_candidate_recipes(motion_profile)
            names = tuple(env.robot.joint_names[i] for i in env.active_joint_ids)
            limits = env.robot.get_qpos_limits()
            if limits.ndim == 3:
                limits = limits[0]
            limits = limits[list(env.active_joint_ids)].cpu()
            session = None
            if committed < cfg.target_episodes:
                session = ExpansionSession(
                    TrajectoryExpansionJobCfg.from_mapping(
                        {
                            "collection": {
                                "max_wall_time_s": cfg.max_wall_time_s,
                                "target_committed_episodes": cfg.target_episodes
                                - committed,
                                "max_rollout_attempts": max(
                                    1, cfg.max_attempts - next_ordinal
                                ),
                                "max_proposals": max(
                                    1, cfg.max_attempts - next_ordinal
                                ),
                            },
                            "execution": {
                                "ready_high_watermark": cfg.batch_size,
                                "ready_low_watermark": 0,
                            },
                            "persistence": {
                                "pending_episode_limit": cfg.batch_size,
                                "pending_max_bytes": cfg.batch_size
                                * cfg.episode_byte_budget,
                            },
                            "validation": {"require_path_collision": True},
                        }
                    )
                )
            while (
                next_ordinal < cfg.max_attempts
                and committed < cfg.target_episodes
                and not session.stop_reason
            ):
                for field, checksum in report["input_files"].items():
                    if (
                        hashlib.sha256(
                            Path(getattr(cfg, field)).read_bytes()
                        ).hexdigest()
                        != checksum
                    ):
                        raise ValueError(f"job input {field} changed after preflight")
                count = min(
                    cfg.batch_size,
                    cfg.max_attempts - next_ordinal,
                    cfg.target_episodes - committed,
                )
                ordinals = list(range(next_ordinal, next_ordinal + count))
                from .recipes import PinnedExpertSource, UniformRangeOperator

                expert_source = PinnedExpertSource(
                    cfg.source, report["recipe_revision"]
                )
                physical = {}
                for row, ordinal in enumerate(ordinals):
                    values = {}
                    for kind in (
                        "mass_scale",
                        "friction",
                        "joint_stiffness_scale",
                        "joint_damping_scale",
                    ):
                        if kind.startswith("joint_") and not getattr(cfg, kind):
                            continue
                        values[kind] = dict(
                            UniformRangeOperator(kind, getattr(cfg, kind)).sample(
                                expert_source.context(cfg.seed, ordinal)
                            )
                        )
                    if cfg.center_of_mass:
                        context = expert_source.context(cfg.seed, ordinal)
                        values["center_of_mass"] = {
                            uid: [
                                context.random(
                                    "center_of_mass", "1", f"{uid}/{axis}"
                                ).uniform(*bounds)
                                for axis, bounds in enumerate(axes)
                            ]
                            for uid, axes in cfg.center_of_mass.items()
                        }
                    physical[row] = values

                if appearance is not None:
                    appearance.close()
                appearance = ExitStack()

                def prepare():
                    if scene_source is not None:
                        from .scene import apply_joint_scene_candidates

                        apply_joint_scene_candidates(
                            SceneView(env.sim, env_ids=tuple(range(count))),
                            {
                                row: scene_source.candidate(ordinal)
                                for row, ordinal in enumerate(ordinals)
                            },
                        )
                    if baseline is not None:
                        masses = {
                            uid: torch.tensor(
                                [
                                    (
                                        physical[row]["mass_scale"].get(uid, 1.0)
                                        if row < count
                                        else 1.0
                                    )
                                    for row in range(cfg.batch_size)
                                ]
                            )
                            for uid in cfg.mass_scale
                        }
                        friction = {
                            uid: torch.tensor(
                                [
                                    (
                                        physical[row]["friction"][uid]
                                        if row < count
                                        else baseline.bodies[managed.index(uid)][3][row]
                                    )
                                    for row in range(cfg.batch_size)
                                ]
                            )
                            for uid in cfg.friction
                        }
                        apply_rigid_physics(
                            view, baseline, mass_scale=masses, friction=friction
                        )

                    if auxiliary_baseline is not None:
                        from .parameters import _apply_auxiliary_physics

                        _apply_auxiliary_physics(
                            view,
                            auxiliary_baseline,
                            robot_uid=env.robot.uid,
                            com_offsets={
                                uid: torch.tensor(
                                    [
                                        (
                                            physical[row]["center_of_mass"][uid]
                                            if row < count
                                            else [0.0, 0.0, 0.0]
                                        )
                                        for row in range(cfg.batch_size)
                                    ]
                                )
                                for uid in cfg.center_of_mass
                            },
                            stiffness_scale={
                                joint: torch.tensor(
                                    [
                                        (
                                            physical[row]["joint_stiffness_scale"][
                                                joint
                                            ]
                                            if row < count
                                            else 1.0
                                        )
                                        for row in range(cfg.batch_size)
                                    ]
                                )
                                for joint in cfg.joint_stiffness_scale
                            },
                            damping_scale={
                                joint: torch.tensor(
                                    [
                                        (
                                            physical[row]["joint_damping_scale"][joint]
                                            if row < count
                                            else 1.0
                                        )
                                        for row in range(cfg.batch_size)
                                    ]
                                )
                                for joint in cfg.joint_damping_scale
                            },
                        )

                    if cfg.visual_profile is not None:
                        from .visual import _visual_profile

                        if (
                            hashlib.sha256(
                                Path(cfg.visual_profile_file).read_bytes()
                            ).hexdigest()
                            != report["input_files"]["visual_profile_file"]
                        ):
                            raise ValueError("visual profile changed after preflight")
                        appearance.enter_context(
                            _visual_profile(env, visual_registry[cfg.visual_profile])
                        )

                phase_started = time.monotonic()

                def prepare_and_settle():
                    prepare()
                    if cfg.settling_steps:
                        env.sim.update(step=cfg.settling_steps)
                    env.sim.sync_render_state()

                initial_obs, _ = env.reset(
                    seed=cfg.seed,
                    options={"save_data": False, "episode_prepare": prepare_and_settle},
                )
                initial = capture_state(view)
                physics_before = capture_physics_properties(view)
                if collision is not None:
                    collision.prepare()
                assignments, bridges, services, records, spools = {}, {}, {}, {}, {}
                program = env.compile_task_program(environment_cfg.task_program)
                for row, ordinal in enumerate(ordinals):
                    if session.stop_reason:
                        break
                    recipe = {
                        "ordinal": ordinal,
                        "seed": cfg.seed,
                        "physics": physical[row],
                        "source": cfg.source,
                        "scene_candidate_id": (
                            scene_source.candidate(ordinal).candidate_id
                            if scene_source is not None
                            else None
                        ),
                        "motion_recipe": (
                            ordinal % len(motion_recipes) if motion_recipes else None
                        ),
                    }
                    recipe_id = _hash((report["recipe_revision"], recipe))
                    initial_values = [asdict(entity) for entity in initial.batches[row]]
                    initial_id = _hash(initial_values)
                    case = SceneCase(
                        _hash((report["recipe_revision"], physical[row])),
                        initial_id,
                        report["job_id"],
                        program.program_id,
                        env.robot.uid,
                    )
                    session.register_case(case, limits, joint_names=names)
                    identity, _ = session.propose(
                        case.scene_case_id,
                        case.initial_state_id,
                        source_id=program.program_id,
                        source_revision=report["recipe_revision"],
                        template_id=recipe_id,
                        operator_id="episode",
                    )
                    if collision is not None and not collision.initial[row]:
                        session.release(identity, reason="initial_validation_failed")
                        directory = attempts_root / f"{ordinal:08d}"
                        directory.mkdir(exist_ok=False)
                        _atomic_json(
                            directory / "attempt.json",
                            {
                                "ordinal": ordinal,
                                "status": "initial_rejected",
                                "identity": asdict(identity),
                                "commit_id": session.episode_ids(identity)[1],
                                "steps": 0,
                            },
                        )
                        continue
                    session.add_prepared_episode(
                        EpisodeCandidate(identity, json.dumps(recipe)),
                        ValidationResult(
                            (ValidationCheck("source_configuration", "passed"),)
                        ),
                    )
                    reserved = session.take_ready(
                        case.scene_case_id,
                        case.initial_state_id,
                        episode_byte_budget=cfg.episode_byte_budget,
                        candidate_id=identity.candidate_id,
                    )
                    if reserved is None:
                        raise RuntimeError("session cannot reserve requested cohort")
                    episode_id, commit_id = session.episode_ids(identity)
                    directory = attempts_root / f"{ordinal:08d}"
                    directory.mkdir(exist_ok=False)
                    record = {
                        "ordinal": ordinal,
                        "status": "prepared",
                        "recipe": recipe,
                        "case": asdict(case),
                        "identity": asdict(identity),
                        "episode_id": episode_id,
                        "commit_id": commit_id,
                        "joint_names": names,
                        "initial_state": initial_values,
                    }
                    _atomic_json(directory / "attempt.json", record)
                    assignments[row] = (
                        identity,
                        episode_id,
                        commit_id,
                        directory,
                        record,
                    )
                    if profile is None:
                        pass
                    else:
                        from .real2sim import _Real2SimRow

                        service = _Real2SimRow(
                            env,
                            row,
                            profile,
                            output=directory,
                            ordinal=ordinal,
                            attempt=0,
                            initial_state_id=initial_id,
                            recipe=profile.recipes[ordinal % len(profile.recipes)],
                        )
                        services[row] = service
                        bridges[row] = service.bridge
                    scene_fields = {
                        (entity.uid, name): tuple(getattr(entity, name))
                        for entity in initial.batches[row]
                        for name in (
                            "pose",
                            "velocity",
                            "qpos",
                            "qvel",
                            "target_qpos",
                            "target_qvel",
                            "qf",
                        )
                        if getattr(entity, name)
                    }
                    records[row] = {
                        "scene": {
                            key: [torch.tensor(value, dtype=torch.float32)]
                            for key, value in scene_fields.items()
                        },
                        "qpos": [env.robot.get_qpos()[row].detach().cpu().clone()],
                        "qvel": [env.robot.get_qvel()[row].detach().cpu().clone()],
                        "qf": [env.robot.get_qf()[row].detach().cpu().clone()],
                        "actions": [],
                        "objects": {
                            uid: [
                                env.sim.get_rigid_object(uid)
                                .body_state[row]
                                .detach()
                                .cpu()
                                .clone()
                            ]
                            for uid in env.sim.get_rigid_object_uid_list()
                        },
                    }
                    if cfg.capture_media:
                        from .visual import _calibrated_frames, _camera_calibration

                        frames = _calibrated_frames(env, initial_obs, row)
                        if not frames:
                            raise ValueError(
                                "capture_media requires configured RGB/depth sensor observations"
                            )
                        spool = MediaSpool(
                            directory / "media",
                            max_buffer_bytes=max(
                                1, cfg.media_buffer_bytes // cfg.batch_size
                            ),
                            parent_episode_id=episode_id,
                            calibration=_camera_calibration(env, row),
                        )
                        spool.append(0.0, frames)
                        spools[row] = spool

                def observe(ids, action, observation):
                    if collision is not None:
                        collision.sample(ids, action)
                    frame_state = capture_state(view)
                    measured_qpos = env.robot.get_qpos().detach().cpu().clone()
                    measured_qvel = env.robot.get_qvel().detach().cpu().clone()
                    measured_qf = env.robot.get_qf().detach().cpu().clone()
                    measured_objects = {
                        uid: env.sim.get_rigid_object(uid)
                        .body_state.detach()
                        .cpu()
                        .clone()
                        for uid in env.sim.get_rigid_object_uid_list()
                    }
                    for row in ids:
                        record = records[row]
                        value = (
                            {"qpos": action.value}
                            if isinstance(action.value, torch.Tensor)
                            else action.value
                        )

                        def active_command(name):
                            data = value[name][row]
                            if data.shape[-1] == env.robot.dof:
                                data = data[list(env.active_joint_ids)]
                            elif data.shape[-1] != len(env.active_joint_ids):
                                raise ValueError(
                                    "submitted command joint layout differs"
                                )
                            return data.detach().cpu()

                        command = active_command("qpos")
                        if (
                            env.expert_action_spec.joint_command_mode
                            == "position_velocity"
                        ):
                            command = torch.cat(
                                (
                                    command,
                                    active_command("qvel"),
                                )
                            )
                        entity_states = {
                            entity.uid: entity for entity in frame_state.batches[row]
                        }
                        for (uid, field), values in record["scene"].items():
                            values.append(
                                torch.tensor(
                                    getattr(entity_states[uid], field),
                                    dtype=torch.float32,
                                )
                            )
                        record["actions"].append(command.clone())
                        record["qpos"].append(measured_qpos[row].clone())
                        record["qvel"].append(measured_qvel[row].clone())
                        record["qf"].append(measured_qf[row].clone())
                        for uid, values in record["objects"].items():
                            values.append(measured_objects[uid][row].clone())
                        if row in spools:
                            spools[row].append(
                                len(record["actions"]) * env.step_dt,
                                _calibrated_frames(env, observation, row),
                            )
                        size = sum(
                            x.numel() * x.element_size()
                            for key in ("actions", "qpos", "qvel", "qf")
                            for x in record[key][-1:]
                        )
                        size += sum(
                            x.numel() * x.element_size()
                            for values in record["objects"].values()
                            for x in values[-1:]
                        )
                        size += sum(
                            values[-1].numel() * values[-1].element_size()
                            for values in record["scene"].values()
                        )
                        record["bytes"] = record.get("bytes", 0) + size
                        if record["bytes"] > cfg.episode_byte_budget:
                            raise BufferError("numeric episode buffer budget exceeded")

                def mark_started(row):
                    session.mark_rollout_started(assignments[row][0])
                    record = assignments[row][4]
                    record["status"] = "running"
                    _atomic_json(assignments[row][3] / "attempt.json", record)

                timings["prepare"] += time.monotonic() - phase_started
                phase_started = time.monotonic()
                if not assignments:
                    next_ordinal += count
                    continue
                if profile is None:
                    result = collect_vector_program(
                        env,
                        program,
                        tuple(assignments),
                        max_steps=cfg.max_steps,
                        observe_step=observe,
                        on_started=mark_started,
                        expansion_profile=motion_profile,
                        recipe_indices=(
                            tuple(
                                (next_ordinal + row) % len(motion_recipes)
                                for row in range(cfg.batch_size)
                            )
                            if motion_recipes
                            else None
                        ),
                    )
                else:
                    result = collect_program_batch(
                        env,
                        bridges,
                        max_steps=cfg.max_steps,
                        observe_step=observe,
                        on_started=mark_started,
                        reject_row=(
                            None
                            if collision is None
                            else lambda row: row in collision.results
                            and collision.results[row].status != "passed"
                        ),
                    )
                timings["execute"] += time.monotonic() - phase_started
                phase_started = time.monotonic()
                physics_after = capture_physics_properties(view)
                pending_commits = []
                for result_row, row in enumerate(result.env_ids):
                    identity, episode_id, commit_id, directory, record = assignments[
                        row
                    ]
                    record["program_segments"] = result.segments[result_row]
                    record["steps"] = result.lengths[result_row]
                    record["reason"] = result.reasons[result_row]
                    if not result.accepted[result_row]:
                        session.release(identity, reason=result.reasons[result_row])
                        record["status"] = "rejected"
                        _atomic_json(directory / "attempt.json", record)
                        continue
                    numeric = records[row]
                    positions = torch.stack(numeric["qpos"])
                    observations = {
                        "joint_positions": positions[:, list(env.active_joint_ids)],
                        "full_joint_positions": positions,
                        "joint_velocities": torch.stack(numeric["qvel"])[
                            :, list(env.active_joint_ids)
                        ],
                        "joint_efforts": torch.stack(numeric["qf"])[
                            :, list(env.active_joint_ids)
                        ],
                    }
                    observations.update(
                        {
                            f"rigid/{uid}/state": torch.stack(values)
                            for uid, values in numeric["objects"].items()
                        }
                    )
                    scene_keys = []
                    for index, ((uid, field), values) in enumerate(
                        numeric["scene"].items()
                    ):
                        key = f"scene_state_{index}"
                        observations[key] = torch.stack(values)
                        scene_keys.append(
                            {"key": key, "entity_uid": uid, "field": field}
                        )
                    metadata = {
                        "expansion_source_artifacts": source_artifacts,
                        "data_type": "sim",
                        "observation_origin": "simulation",
                        "visual_profile": cfg.visual_profile,
                        "visual_profile_revision": report["input_files"].get(
                            "visual_profile_file"
                        ),
                        "recipe": record["recipe"],
                        "scene_case": record["case"],
                        "joint_names": names,
                        "source_episode_uuid": _hash(
                            ("task_program", program.program_id)
                        ),
                        "expert_action_spec": env.expert_action_spec.metadata(
                            step_dt=env.step_dt
                        ),
                        "initial_state": record["initial_state"],
                        "scene_state_fields": scene_keys,
                        "physics_backend": env.sim.physics_backend,
                        "program_segments": result.segments[result_row],
                        "expansion": (
                            result.expansion_records[result_row]
                            if result.expansion_records
                            else ()
                        ),
                        "segments": result.segments[result_row],
                        "instruction": program.instruction,
                        "instruction_source": "task_program",
                    }
                    physics_unchanged = all(
                        torch.allclose(
                            physics_before[key][row],
                            physics_after[key][row],
                            atol=1e-6,
                            rtol=1e-6,
                        )
                        for key in physics_before
                    )

                    def parameter_value(value):
                        if isinstance(value, list):
                            return [parameter_value(item) for item in value]
                        return (
                            value
                            if math.isfinite(value)
                            else ("+inf" if value > 0 else "-inf")
                        )

                    metadata["physics_parameters"] = {
                        key: parameter_value(value[row].detach().cpu().tolist())
                        for key, value in physics_before.items()
                    }
                    metadata["physics_fingerprint"] = _hash(
                        metadata["physics_parameters"]
                    )
                    checks = [
                        ValidationCheck(
                            "physics_parameters",
                            "passed" if physics_unchanged else "failed",
                            "Measured episode parameter baseline",
                        ),
                        ValidationCheck(
                            "task_success",
                            "passed",
                            "Canonical Task Program and segment validators completed",
                        ),
                    ]
                    if collision is not None:
                        measured = collision.terminal.get(row)
                        if measured is None:
                            checks.append(
                                ValidationCheck("task.measured", "unavailable")
                            )
                        else:
                            checks.extend(measured.checks)
                        checked = collision.results.get(row)
                        checks.append(
                            ValidationCheck(
                                "path_collision",
                                (
                                    "passed"
                                    if checked is not None
                                    and checked.status == "passed"
                                    else "failed"
                                ),
                                (
                                    "Measured control-boundary collision checks"
                                    if checked is None
                                    else checked.reason
                                ),
                            )
                        )
                    if row in services:
                        request = services[row].qualify(
                            qpos=positions,
                            object_pose=torch.stack(
                                numeric["objects"][profile.object_uid]
                            )[:, :7],
                        )
                        metadata.update(dict(request.metadata))
                        metadata.pop("source_episode_uuid", None)
                        metadata["source_episode_uuids"] = sorted(
                            {
                                _hash(
                                    (
                                        "real2sim",
                                        unit.unit.source.source_id,
                                        unit.unit.source.source_revision,
                                    )
                                )
                                for unit in services[row].prototypes.values()
                            }
                        )
                        checks.extend(request.validation.checks)
                        checks.append(
                            ValidationCheck(
                                "path_collision",
                                "passed",
                                "Source-qualified planned and measured collision checks",
                            )
                        )
                    if row in spools:
                        metadata["media_manifest"] = str(spools[row].finish())
                        metadata["media_manifest_sha256"] = hashlib.sha256(
                            Path(metadata["media_manifest"]).read_bytes()
                        ).hexdigest()
                    episode = ExpertEpisode(
                        identity,
                        observations,
                        torch.stack(numeric["actions"]),
                        torch.arange(len(positions), dtype=torch.float64) * env.step_dt,
                        (
                            "joint_position_velocity"
                            if env.expert_action_spec.joint_command_mode
                            == "position_velocity"
                            else "joint_position"
                        ),
                        ValidationResult(tuple(checks)),
                        episode_id,
                        commit_id,
                        metadata=metadata,
                    )
                    record["measured_accepted"] = episode.validation.accepted
                    record["validation"] = [
                        dict(
                            check_id=check.check_id,
                            status=check.status,
                            detail=check.detail,
                            metrics=dict(check.metrics),
                        )
                        for check in episode.validation.checks
                    ]
                    if session.accept_episode(episode):
                        sink.stage(episode)
                        record["status"] = "pending_write"
                        _atomic_json(directory / "attempt.json", record)
                        pending_commits.append((episode, directory, record))
                    else:
                        record["status"] = "coverage_rejected"
                    _atomic_json(directory / "attempt.json", record)
                # Freeze every accepted row before the first potentially ambiguous write.
                for episode, directory, record in pending_commits:
                    receipt = sink.submit(episode, submission_id=0)
                    session.apply_receipt(receipt)
                    committed += 1
                    record["status"] = "committed"
                    record["storage_id"] = receipt.storage_id
                    _atomic_json(directory / "attempt.json", record)
                timings["commit"] += time.monotonic() - phase_started
                next_ordinal += count
                _atomic_json(
                    root / "progress.json",
                    {
                        "committed": committed,
                        "next_ordinal": next_ordinal,
                        "elapsed_seconds": time.monotonic() - started,
                    },
                )
            if appearance is not None:
                appearance.close()
                appearance = None
            if cfg.render_profiles and committed >= cfg.target_episodes:
                from .visual import render_observation_variants

                phase_started = time.monotonic()
                render_observation_variants(
                    env,
                    root,
                    profile_file=cfg.visual_profile_file,
                    profiles=cfg.render_profiles,
                    buffer_bytes=cfg.media_buffer_bytes,
                )
                timings["render_variants"] += time.monotonic() - phase_started
            if cfg.export_lerobot and committed >= cfg.target_episodes:
                from .export import export_expansion_lerobot

                phase_started = time.monotonic()
                export_expansion_lerobot(env, root)
                timings["export"] += time.monotonic() - phase_started
            import resource

            timings["elapsed"] = time.monotonic() - started
            _atomic_json(
                root / "runtime.json",
                {
                    "seconds": timings,
                    "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    * 1024,
                    "committed": committed,
                    "stop_reason": session.stop_reason if session is not None else None,
                    "profiler": env._profiler.report(),
                },
            )
            return inspect_expansion_job(root)
        finally:
            if appearance is not None:
                appearance.close()
            if collision is not None:
                collision.close()
            if env is not None:
                # Release row runtimes before destroying their native entity bindings.
                if "bridges" in locals():
                    bridges.clear()
                    services.clear()
                env._active_task_program_bridge = None
                env.close(exit_process=False)
            env = None
            SimulationManager.flush_cleanup_queue()
