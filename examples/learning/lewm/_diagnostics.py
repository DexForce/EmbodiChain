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

"""Validation-only comparisons between visual predictions and physical progress."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from embodichain_models.dynamics.lewm import load_lewm, preprocess_rgb

from ._evaluation import _check_contract
from ._sim import close_env, frame, goal_image, make_env, render_contract, reset


def rank_summary(
    predicted: np.ndarray, observed: np.ndarray, distances: np.ndarray
) -> dict[str, Any]:
    """Measure goal ranking, masking physically indistinguishable outcomes.

    Candidate zero is the recorded guided continuation, and candidate one is
    the zero-action control. Geometry is used only to diagnose visual scores.
    """
    from scipy.stats import spearmanr

    if (
        predicted.ndim != 1
        or len(predicted) < 2
        or observed.shape != predicted.shape
        or distances.shape != predicted.shape
        or not all(np.isfinite(x).all() for x in (predicted, observed, distances))
    ):
        raise ValueError("ranking needs matching finite candidate vectors")
    selected = int(np.argmin(predicted))
    actual_best = int(np.argmin(distances))
    informative = float(np.ptp(distances)) >= 0.001

    def correlation(left: np.ndarray, right: np.ndarray) -> float | None:
        if not informative or np.ptp(left) < 1e-8 or np.ptp(right) < 1e-8:
            return None
        value = float(spearmanr(left, right).statistic)
        return value if np.isfinite(value) else None

    return {
        "physically_informative": informative,
        "distance_range_m": float(np.ptp(distances)),
        "predicted_vs_physical_spearman": correlation(predicted, distances),
        "observed_vs_physical_spearman": correlation(observed, distances),
        "predicted_vs_observed_spearman": correlation(predicted, observed),
        "selected_candidate": selected,
        "actual_best_candidate": actual_best,
        "selected_distance_m": float(distances[selected]),
        "actual_best_distance_m": float(distances[actual_best]),
        "selection_regret_m": float(distances[selected] - distances[actual_best]),
        "guided_distance_m": float(distances[0]),
        "zero_action_distance_m": float(distances[1]),
    }


def _physical_error(before: Any, after: Any) -> float:
    errors = []
    if len(before.batches[0]) != len(after.batches[0]):
        raise RuntimeError("diagnostic scene topology changed")
    for left, right in zip(before.batches[0], after.batches[0]):
        if left.uid != right.uid:
            raise RuntimeError("diagnostic scene ownership changed")
        for name in (
            "pose",
            "velocity",
            "qpos",
            "qvel",
            "target_qpos",
            "target_qvel",
            "qf",
        ):
            a, b = np.asarray(getattr(left, name)), np.asarray(getattr(right, name))
            if a.size:
                errors.append(float(np.max(np.abs(a - b))))
    return max(errors, default=0.0)


@torch.no_grad()
def rescore_candidates(
    export: str, reference_export: str, trace_path: str, device: str
) -> dict[str, Any]:
    """Compare predictor changes on the exact same executed physical candidates."""
    model, manifest = load_lewm(export, device)
    reference, reference_manifest = load_lewm(reference_export, "cpu")
    if model.cfg.to_dict() != reference.cfg.to_dict():
        raise ValueError("candidate rescoring needs the same model architecture")
    current_state, reference_state = model.state_dict(), reference.state_dict()
    frozen_names = [
        name
        for name in reference_state
        if name.startswith(("encoder.", "projector."))
        or name in {"action_mean", "action_std"}
    ]
    for name in frozen_names:
        if not torch.equal(current_state[name].cpu(), reference_state[name]):
            raise ValueError(
                f"candidate rescoring requires unchanged visual representation and action statistics: {name}"
            )
    del reference
    path = Path(trace_path)
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    rows = []
    with h5py.File(path, "r") as evidence:
        if not evidence.attrs.get("complete", False):
            raise ValueError("candidate evidence is incomplete")
        metadata = json.loads(evidence.attrs["metadata"])
        if metadata["weights_sha256"] != reference_manifest["weights_sha256"]:
            raise ValueError("candidate evidence differs from the reference model")
        horizons = metadata["horizons"]
        for seed, group in evidence.items():
            pixels = torch.from_numpy(group["history_pixels"][:]).to(device)[None]
            history = model.encode(preprocess_rgb(pixels, model.cfg.image_size))
            prefix = group["prefix_actions"][:]
            past = (
                prefix[-(model.cfg.history_size - 1) :]
                if model.cfg.history_size > 1
                else prefix[:0]
            )
            bank = torch.from_numpy(group["candidates"][:]).to(device)[None]
            predicted = (
                model.rollout(history, torch.from_numpy(past).to(device)[None], bank)[0]
                .cpu()
                .numpy()
            )
            goal = group["goal_latent"][:]
            observed = group["actual_latent"][:]
            distances = group["actual_distance"][:]
            scores = {}
            for position, horizon in enumerate(horizons):
                predicted_cost = ((predicted[:, horizon - 1] - goal) ** 2).sum(-1)
                observed_cost = ((observed[:, position] - goal) ** 2).sum(-1)
                scores[str(horizon)] = rank_summary(
                    predicted_cost, observed_cost, distances[:, position]
                )
                spread = float(abs(distances[1, position] - distances[-1, position]))
                scores[str(horizon)]["repeated_zero_action_spread_m"] = spread
                scores[str(horizon)]["repeat_control_consistent"] = spread < 0.001
            rows.append({"seed": int(seed), "status": "measured", "horizons": scores})
    return {
        "metadata": {
            **metadata,
            "weights_sha256": manifest["weights_sha256"],
            "reference_weights_sha256": reference_manifest["weights_sha256"],
            "frozen_visual_verified": True,
            "physical_evidence_sha256": digest.hexdigest(),
        },
        "rows": rows,
        "trace": str(path.resolve()),
    }


@torch.no_grad()
def diagnose_candidates(
    export: str,
    corpus: str,
    device: str,
    seeds: list[int],
    trace_path: str,
    *,
    candidates: int = 32,
    horizons: tuple[int, ...] = (5, 10, 20),
    goal_render_frames: int = 32,
) -> dict[str, Any]:
    """Compare a fixed action bank at measured pre-contact validation states.

    Each candidate starts from a full seeded reset, an identical recorded
    action prefix and an explicit numeric physical-state restore. Solver
    internals and renderer state are not claimed to be checkpointed; repeated
    zero-action candidates measure remaining branch variation.
    Candidates run open loop for the full diagnostic horizon even after task
    success. This is not a policy evaluation or a controller using geometry.
    """
    from embodichain.lab.sim.scene import SceneView, capture_state, restore_state

    if candidates < 4 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(
            "provide distinct validation seeds and at least four candidates"
        )
    if not horizons or any(type(x) is not int or x < 1 for x in horizons):
        raise ValueError("horizons must contain positive integers")
    horizons = tuple(sorted(set(horizons)))
    maximum = max(horizons)
    output = Path(trace_path)
    temporary = output.with_name(output.name + ".tmp")
    if output.exists() or temporary.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model, manifest = load_lewm(export, device)
    if model.cfg.action_block != 1:
        raise ValueError(
            "candidate-bank diagnosis currently requires single-step models"
        )
    if set(seeds) & set(manifest["metadata"]["train_seeds"]):
        raise ValueError("candidate diagnosis cannot use training seeds")
    digest = hashlib.sha256()
    with Path(corpus).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    expected_source = (
        f"lewm-hdf5:{digest.hexdigest()}:validation:{model.cfg.history_size}:1"
    )
    if manifest["metadata"]["validation_source"] != expected_source:
        raise ValueError("diagnostic corpus differs from the model validation source")
    with h5py.File(corpus, "r") as file:
        if file.attrs["split"] != "validation":
            raise ValueError("candidate diagnosis requires validation data")
        recorded = {
            int(e.attrs["seed"]): np.asarray(e["actions"], dtype=np.float32)
            for e in file["episodes"].values()
            if int(e.attrs["seed"]) in seeds
        }
    if set(recorded) != set(seeds):
        raise ValueError("requested diagnostic seeds are absent from the corpus")
    env, config = make_env(device)
    raw = env.unwrapped
    view = SceneView(raw.sim)
    expected_contract = manifest["metadata"]["data_contract"]
    actual_contract = {
        "action_dim": raw.single_action_space.shape[0],
        "action_semantics": "joint_position.relative@1",
        "step_dt": raw.step_dt,
        "action_scale": 0.1,
        "action_clip": 1.0,
        "camera": config["sensor"][0],
        "environment": "PushCubeRL",
        "physics": config["physics"],
        "render": render_contract(env),
    }
    try:
        _check_contract(expected_contract, actual_contract)
    except Exception:
        close_env(env)
        raise
    metadata = {
        "weights_sha256": manifest["weights_sha256"],
        "source_id": expected_source,
        "render": render_contract(env),
        "goal_render_frames": goal_render_frames,
        "horizons": horizons,
        "candidates": candidates,
        "device": device,
        "purpose": "validation diagnosis; privileged geometry is not a policy input",
        "start_method": "seeded reset, replayed prefix and numeric physical restore; repeated zero-action control",
    }
    rows = []

    def encode(images: np.ndarray) -> torch.Tensor:
        pixels = torch.from_numpy(np.ascontiguousarray(images)).to(device)[None]
        return model.encode(preprocess_rgb(pixels, model.cfg.image_size))

    def start(seed: int, prefix: np.ndarray) -> tuple[Any, np.ndarray]:
        obs, _ = reset(env, seed)
        target = goal_image(env, goal_render_frames)
        with raw.sim.render_frame():
            obs = raw.get_obs()
        for action in prefix:
            obs, *_ = raw.step(torch.as_tensor(action, device=device)[None])
        return obs, target

    try:
        with h5py.File(temporary, "x") as evidence:
            evidence.attrs["format"] = "embodichain.lewm.candidate_diagnosis.v1"
            evidence.attrs["metadata"] = json.dumps(metadata)
            for seed in seeds:
                actions = recorded[seed]
                obs, target = start(seed, actions[:0])
                images = [frame(obs)]
                cube_start = (
                    raw.sim.get_rigid_object("cube").get_local_pose()[0, :2].clone()
                )
                reference_state = None
                prefix_length = None
                history_pixels = None
                first_motion_step = None
                for index, action in enumerate(actions):
                    before = capture_state(view)
                    qpos = raw.robot.get_qpos()
                    limits = raw.robot.get_qpos_limits()
                    admissible = bool(
                        (
                            (qpos >= limits[..., 0] - 1e-4)
                            & (qpos <= limits[..., 1] + 1e-4)
                        ).all()
                    )
                    if index >= model.cfg.history_size - 1 and admissible:
                        reference_state, prefix_length = before, index
                        history_pixels = np.stack(images[-model.cfg.history_size :])
                    obs, *_ = raw.step(torch.as_tensor(action, device=device)[None])
                    cube_now = raw.sim.get_rigid_object("cube").get_local_pose()[0, :2]
                    if (
                        index >= model.cfg.history_size - 1
                        and float(torch.linalg.vector_norm(cube_now - cube_start))
                        > 0.002
                    ):
                        first_motion_step = index + 1
                        break
                    images.append(frame(obs))
                if (
                    reference_state is None
                    or prefix_length is None
                    or first_motion_step is None
                ):
                    rows.append(
                        {
                            "seed": seed,
                            "status": "no_measured_contact_in_recorded_prefix",
                        }
                    )
                    continue
                prefix = actions[:prefix_length]
                assert history_pixels is not None
                history = encode(history_pixels)
                past_actions = (
                    prefix[-(model.cfg.history_size - 1) :]
                    if model.cfg.history_size > 1
                    else prefix[:0]
                )
                past = torch.as_tensor(past_actions, device=device)[None]
                goal = encode(target[None])[:, 0]
                rng = np.random.default_rng(seed)
                bank = np.empty(
                    (candidates, maximum, model.cfg.action_dim), dtype=np.float32
                )
                continuation = actions[prefix_length : prefix_length + maximum]
                bank[0] = np.pad(
                    continuation, ((0, maximum - len(continuation)), (0, 0))
                )
                bank[1] = 0
                middle = 2 + (candidates - 2) // 2
                bank[2:middle] = bank[0] + rng.normal(
                    0, 0.12, size=bank[2:middle].shape
                )
                held = rng.uniform(
                    -1,
                    1,
                    size=(
                        candidates - middle,
                        (maximum + 4) // 5,
                        model.cfg.action_dim,
                    ),
                )
                bank[middle:] = np.repeat(held, 5, axis=1)[:, :maximum]
                bank[-1] = 0
                np.clip(bank, -1, 1, out=bank)
                predicted = model.rollout(
                    history, past, torch.from_numpy(bank).to(device)[None]
                )[0]
                goal_array = goal[0].cpu().numpy()
                predicted_array = predicted.cpu().numpy()
                group = evidence.create_group(str(seed))
                group.create_dataset(
                    "history_pixels", data=history_pixels, compression="lzf"
                )
                group.create_dataset("goal_pixels", data=target, compression="lzf")
                group.create_dataset("prefix_actions", data=prefix)
                group.create_dataset("candidates", data=bank)
                group.create_dataset("predicted_latent", data=predicted_array)
                group.create_dataset("goal_latent", data=goal_array)
                actual_latent = np.empty(
                    (candidates, len(horizons), model.cfg.embed_dim), dtype=np.float32
                )
                distances = np.empty((candidates, len(horizons)), dtype=np.float64)
                state_errors = []
                prefix_errors = []
                for candidate_index, sequence in enumerate(bank):
                    start(seed, prefix)
                    prefix_errors.append(
                        _physical_error(reference_state, capture_state(view))
                    )
                    restore_state(view, reference_state, qpos_tolerance=1e-4)
                    error = _physical_error(reference_state, capture_state(view))
                    state_errors.append(error)
                    if error > 1e-4:
                        raise RuntimeError(
                            f"candidate {candidate_index} replay state differs by {error}"
                        )
                    for step, action in enumerate(sequence, start=1):
                        obs, _, _, _, info = raw.step(
                            torch.as_tensor(action, device=device)[None]
                        )
                        if step in horizons:
                            position = horizons.index(step)
                            pixels = frame(obs)
                            actual_latent[candidate_index, position] = (
                                encode(pixels[None])[0, 0].cpu().numpy()
                            )
                            distances[candidate_index, position] = float(
                                info["metrics"]["distance_to_goal"][0]
                            )
                            group.create_dataset(
                                f"pixels/{candidate_index}/{step}",
                                data=pixels,
                                compression="lzf",
                            )
                    if candidate_index % 8 == 0:
                        print(
                            f"seed={seed} candidate={candidate_index + 1}/{candidates}",
                            flush=True,
                        )
                group.create_dataset("actual_latent", data=actual_latent)
                group.create_dataset("actual_distance", data=distances)
                group.create_dataset("start_max_abs_error", data=state_errors)
                group.create_dataset("prefix_max_abs_error", data=prefix_errors)
                scores = {}
                for position, horizon in enumerate(horizons):
                    predicted_cost = (
                        (predicted_array[:, horizon - 1] - goal_array) ** 2
                    ).sum(-1)
                    observed_cost = (
                        (actual_latent[:, position] - goal_array) ** 2
                    ).sum(-1)
                    scores[str(horizon)] = rank_summary(
                        predicted_cost, observed_cost, distances[:, position]
                    )
                    control_spread = float(
                        abs(distances[1, position] - distances[-1, position])
                    )
                    scores[str(horizon)][
                        "repeated_zero_action_spread_m"
                    ] = control_spread
                    scores[str(horizon)]["repeat_control_consistent"] = (
                        control_spread < 0.001
                    )
                row = {
                    "seed": seed,
                    "status": "measured",
                    "prefix_steps": prefix_length,
                    "first_recorded_cube_motion_step": first_motion_step,
                    "start_max_abs_error": max(state_errors),
                    "prefix_max_abs_error": max(prefix_errors),
                    "horizons": scores,
                }
                rows.append(row)
                evidence.flush()
                print(json.dumps(row), flush=True)
            evidence.attrs["complete"] = True
        temporary.replace(output)
        return {"metadata": metadata, "rows": rows, "trace": str(output.resolve())}
    finally:
        close_env(env)
