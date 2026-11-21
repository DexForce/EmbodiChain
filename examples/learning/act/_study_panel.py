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

"""Score native ACT checkpoints on synchronized, independently seeded PushCube arenas."""

from __future__ import annotations

import multiprocessing as mp
import time
import traceback
from pathlib import Path
from typing import Any
from collections.abc import Callable

import torch

from ._environment import close_env, make_env, schema_for_env
from .evaluate import _interval, _validate_policy_identity


def score(
    env: Any,
    checkpoint: Path | None,
    seeds: list[int],
    panel: str,
    baseline: str = "policy",
) -> dict[str, Any]:
    """Measure the official task predicate and XY distance at each lane's first terminal step."""
    from embodichain.extensions.models.policy.act import load_act_policy

    raw = env.unwrapped
    raw._replay_no_auto_reset = True
    device = raw.device
    model = None
    if baseline != "fixed":
        if checkpoint is None:
            raise ValueError(
                "A native ACT checkpoint or initializer reference is required"
            )
        torch.manual_seed(19001)
        model = load_act_policy(
            checkpoint, str(device), untrained=baseline == "untrained"
        )
        _validate_policy_identity(model.metadata, checkpoint, panel=panel, seeds=seeds)
        if model.metadata["schema"] != schema_for_env(env):
            raise ValueError("ACT state/action schema differs from the live robot")
    rows = []
    low = torch.as_tensor(raw.single_action_space.low, device=device)
    high = torch.as_tensor(raw.single_action_space.high, device=device)
    rigid_cube = raw.sim.get_rigid_object("cube")
    for start in range(0, len(seeds), raw.num_envs):
        group = seeds[start : start + raw.num_envs]
        for lane in range(raw.num_envs):
            obs, _ = env.reset(
                seed=group[min(lane, len(group) - 1)],
                options={"save_data": False, "reset_ids": [lane]},
            )
        if model is not None:
            model.reset()
        initial = obs["robot"]["qpos"].clone()
        cube = rigid_cube.get_local_pose()[: len(group), :2].tolist()
        goals = raw.goal_pose[: len(group), :2, 3].tolist()
        fixed = torch.cat((initial[:, :7], torch.full_like(initial[:, :1], -1)), dim=-1)
        done = torch.zeros(len(group), dtype=torch.bool, device=device)
        success = done.clone()
        distance = torch.zeros(len(group), device=device)
        final_xyz = torch.zeros(len(group), 3, device=device)
        lengths = torch.zeros(len(group), device=device, dtype=torch.long)
        for step in range(1, raw.max_episode_steps + 1):
            if model is None:
                action = fixed
            else:
                rgb = (
                    obs["sensor"]["cam_high"]["color"][..., :3]
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                action = model.select_action(rgb, obs["robot"]["qpos"]).clamp(low, high)
            obs, _, terminated, truncated, info = env.step(action)
            term = (
                torch.as_tensor(terminated, device=device)
                | torch.as_tensor(truncated, device=device)
            ).expand(raw.num_envs)[: len(group)]
            finished = ~done & (term | (step == raw.max_episode_steps))
            success |= finished & info["success"][: len(group)]
            distance = torch.where(
                finished, info["metrics"]["distance_to_goal"][: len(group)], distance
            )
            lengths = torch.where(finished, step, lengths)
            final_xyz = torch.where(
                finished[:, None],
                rigid_cube.get_local_pose()[: len(group), :3],
                final_xyz,
            )
            done |= finished
            if done.all():
                break
        for seed, xy, goal, xyz, ok, dist, n in zip(
            group,
            cube,
            goals,
            final_xyz.tolist(),
            success.tolist(),
            distance.tolist(),
            lengths.tolist(),
            strict=True,
        ):
            rows.append(
                dict(
                    seed=seed,
                    initial_cube_xy=xy,
                    goal_xy=goal,
                    final_cube_xyz=xyz,
                    success=ok,
                    goal_distance=dist,
                    steps=n,
                    strict_success=bool(ok and dist < 0.03),
                )
            )
    fingerprint = (
        model.metadata["policy_sha256"]
        if model is not None and baseline == "policy"
        else None
    )
    return _panel_report(rows, checkpoint, panel, baseline, fingerprint)


def _panel_report(
    rows: list[dict[str, Any]],
    checkpoint: Path | None,
    panel: str,
    baseline: str,
    fingerprint: str | None,
) -> dict[str, Any]:
    count = sum(row["success"] for row in rows)
    return {
        "panel": panel,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "baseline": baseline,
        "policy_sha256": fingerprint,
        "records": rows,
        "successes": count,
        "episodes": len(rows),
        "success_rate": count / len(rows),
        "success_rate_wilson_95": _interval(count, len(rows)),
        "strict_successes": sum(r["strict_success"] for r in rows),
        "mean_goal_distance": sum(r["goal_distance"] for r in rows) / len(rows),
    }


def _worker(pipe: Any, arenas: int, spacing: float | None) -> None:
    env = None
    try:
        torch.set_num_threads(2)
        env = make_env("cuda:0", camera=True, num_envs=arenas, spacing=spacing)
        pipe.send(("ready", None))
        while (request := pipe.recv()) is not None:
            pipe.send(("result", score(env, **request)))
    except BaseException:
        pipe.send(("error", traceback.format_exc()))
    finally:
        close_env(env)
        pipe.close()


class Evaluator:
    """Own one independent simulator and load immutable native policies per request."""

    def __init__(self, arenas: int, spacing: float | None = None) -> None:
        ctx = mp.get_context("spawn")
        self.pipe, child = ctx.Pipe()
        self.process = ctx.Process(target=_worker, args=(child, arenas, spacing))
        self.process.start()
        child.close()

    def receive(self, progress: Callable[[], None] | None = None) -> Any:
        deadline = time.monotonic() + 900
        while not self.pipe.poll(
            min(0.1 if progress else 900, max(0.0, deadline - time.monotonic()))
        ):
            if time.monotonic() >= deadline or not self.process.is_alive():
                raise RuntimeError("ACT evaluator stopped or missed its deadline")
            if progress is not None:
                progress()
        kind, value = self.pipe.recv()
        if kind == "error":
            raise RuntimeError(value)
        return value

    def evaluate(
        self,
        checkpoint: Path | None,
        seeds: list[int],
        panel: str,
        baseline: str = "policy",
        *,
        progress: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        self.pipe.send(
            dict(checkpoint=checkpoint, seeds=seeds, panel=panel, baseline=baseline)
        )
        return self.receive(progress)

    def close(self) -> None:
        if self.process.is_alive():
            self.pipe.send(None)
        self.process.join(40)
        self.pipe.close()
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()
            raise RuntimeError("ACT evaluator required forced shutdown")
        if self.process.exitcode != 0:
            raise RuntimeError(f"ACT evaluator exited with {self.process.exitcode}")


class IndependentEvaluators:
    """Split a panel across single-arena simulators to preserve the source RGB background."""

    def __init__(self, workers: int) -> None:
        self.workers = []
        try:
            for _ in range(workers):
                self.workers.append(Evaluator(1))
        except BaseException:
            self.close()
            raise

    def receive(self) -> None:
        for worker in self.workers:
            worker.receive()

    def evaluate(
        self,
        checkpoint: Path | None,
        seeds: list[int],
        panel: str,
        baseline: str = "policy",
        *,
        progress: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        active = self.workers[: len(seeds)]
        for index, worker in enumerate(active):
            worker.pipe.send(
                dict(
                    checkpoint=checkpoint,
                    seeds=seeds[index :: len(active)],
                    panel=panel,
                    baseline=baseline,
                )
            )
        results = [worker.receive(progress) for worker in active]
        hashes = {p["policy_sha256"] for p in results}
        records = {row["seed"]: row for p in results for row in p["records"]}
        if (
            len(hashes) != 1
            or set(records) != set(seeds)
            or len(seeds) != sum(p["episodes"] for p in results)
        ):
            raise RuntimeError("Incomplete or inconsistent single-arena evaluation")
        return _panel_report(
            [records[seed] for seed in seeds], checkpoint, panel, baseline, hashes.pop()
        )

    def close(self) -> None:
        errors = []
        for worker in self.workers:
            try:
                worker.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError("Failed to close ACT evaluation workers") from errors[0]
