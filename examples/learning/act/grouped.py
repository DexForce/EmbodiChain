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

"""Train one visual ACT policy from local or shared grouped ODS producers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import threading
import time
from typing import Any, TYPE_CHECKING
from contextlib import ExitStack

from ._environment import task_config, task_identity

if TYPE_CHECKING:
    from ._grouped_ddp import Ranks
    from ._grouped_replay import GroupReplay

__all__ = ["main", "make_engine", "seeded_expert"]


def _scene_shard(
    seed: int, width: int, offset: int, lanes: int, stratified: bool
) -> list[tuple[int, int | None]]:
    if not 0 <= offset < offset + lanes <= width or (stratified and width != 10):
        raise ValueError("Invalid Push Cube lane shard for the reset declaration")
    strata = list(range(width))
    if stratified:
        random.Random(seed).shuffle(strata)
    return [
        (seed + lane, strata[lane] if stratified else None)
        for lane in range(offset, offset + lanes)
    ]


def seeded_expert(env: Any, **kwargs: Any) -> Any:
    """Give each attempt a distinct deterministic scene in the worker process."""
    from embodichain.lab.gym.envs.demo import execute_demo_episode

    raw = env.unwrapped
    spec = raw.cfg.extensions["act_study"]
    attempt = getattr(raw, "_act_next_attempt", 0)
    width = spec["logical_width"]
    stratified = spec["stratified_resets"]
    scenes = _scene_shard(
        spec["seed_base"] + attempt * width,
        width,
        spec["lane_offset"],
        raw.num_envs,
        stratified,
    )
    raw._act_next_attempt = attempt + 1
    if attempt == 0:
        import torch

        _write(
            Path(spec["attempts"]).with_name("producer.json"),
            {"pid": os.getpid(), "gpu_uuid": _gpu_uuid(torch.device(raw.device))},
        )
    if stratified:
        import copy

        manager = raw.event_manager
        original = manager.get_functor_cfg("randomize_cube")
        low, high = original.params["position_range"]
    try:
        for lane, (seed, cell) in enumerate(scenes):
            if stratified:
                term = copy.copy(original)
                term.params = original.params.copy()
                bounds = [list(low), list(high)]
                if cell < 9:
                    for axis, rank in enumerate(divmod(cell, 3)):
                        span = high[axis] - low[axis]
                        # Include outer-edge starts in every small publication.
                        center = low[axis] + span * (0.075 + 0.425 * rank)
                        radius = span * 0.075
                        bounds[0][axis] = center - radius
                        bounds[1][axis] = center + radius
                term.params["position_range"] = bounds
                manager.set_functor_cfg("randomize_cube", term)
            env.reset(seed=seed, options={"save_data": False, "reset_ids": [lane]})
    finally:
        if stratified:
            manager.set_functor_cfg("randomize_cube", original)
    xy = raw.sim.get_rigid_object("cube").get_local_pose()[:, :2].tolist()
    result = execute_demo_episode(env, **kwargs)
    with Path(spec["attempts"]).open("a") as stream:
        for lane, length in enumerate(result.lengths):
            seed, stratum = scenes[lane]
            stream.write(
                json.dumps(
                    {
                        "seed": seed,
                        "attempt": attempt,
                        "initial_cube_xy": xy[lane],
                        "stratum": stratum,
                        "accepted": bool(result.completed and result.all_success),
                        "steps": length,
                    }
                )
                + "\n"
            )
    return result


def make_engine(
    path: Path,
    seed: int,
    capacity: int,
    *,
    lanes: int = 1,
    spacing: float | None = None,
    floor: int,
    stratified_resets: bool = False,
    device: str = "cuda:0",
    logical_width: int | None = None,
    lane_offset: int = 0,
) -> Any:
    """Start an explicit single-renderer producer with raw PushCube policy requests."""
    from embodichain.data_pipeline.engine.data import (
        OnlineDataEngine,
        OnlineDataEngineCfg,
    )

    cfg = task_config(device, camera=True, num_envs=lanes, spacing=spacing)
    cfg["seed"] = seed + lane_offset
    cfg["env"]["extensions"]["act_study"] = {
        "seed_base": seed,
        "attempts": str(path / "attempts.jsonl"),
        "stratified_resets": stratified_resets,
        "logical_width": logical_width if logical_width is not None else lanes,
        "lane_offset": lane_offset,
    }
    return OnlineDataEngine(
        OnlineDataEngineCfg(
            gym_config=cfg,
            state_dim=9,
            action_dim=8,
            buffer_size=capacity,
            max_episode_steps=300,
            buffer_device="cpu",
            refill_policy="manual",
            group_min_draws=floor,
            action_semantics="policy_request",
            action_contracts=("joint_position.absolute@1", "gripper.parallel@1"),
            include_provenance=True,
            episode_runner="examples.learning.act.grouped:seeded_expert",
            max_generation_attempts=4,
            initialization_timeout=300,
            shutdown_timeout=40,
        )
    )


def _source_hashes() -> dict[str, str]:
    import embodichain

    package = Path(embodichain.__file__).resolve().parent
    example = Path(__file__).resolve()
    paths = (
        example,
        example.with_name("_grouped_ddp.py"),
        example.with_name("_grouped_replay.py"),
        example.with_name("_study_panel.py"),
        package / "extensions/models/policy/act/runtime.py",
        package / "lab/gym/envs/base_env.py",
        package / "data_pipeline/engine/data.py",
    )
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def _write(path: Path, value: Any) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def _gpu_uuid(device: Any) -> str:
    import torch

    return "GPU-" + str(torch.cuda.get_device_properties(device).uuid).removeprefix(
        "GPU-"
    )


class _Producer:
    def __init__(
        self, engine: Any, path: Path, rank: int, width: int, slots: int, cap: int
    ) -> None:
        self.engine = engine
        self.path = path
        self.rank = rank
        self.width = width
        self.slots = slots
        self.cap = cap
        self.releases: list[dict] = []
        self.requested = 1
        self.sampler: GroupReplay | None = None
        self.floor_waits = 0
        self.floor_wait_seconds = 0.0
        self.floor_blocked_since: float | None = None

    def refill(self) -> None:
        if self.requested * self.width >= self.cap:
            self.engine.check_health()
            return
        request = (
            self.sampler.refill()
            if self.sampler is not None
            else self.engine.try_refill_group()
        )
        now = time.monotonic()
        if request is None:
            self.floor_waits += 1
            if self.floor_blocked_since is None:
                self.floor_blocked_since = now
            return
        if self.floor_blocked_since is not None:
            self.floor_wait_seconds += now - self.floor_blocked_since
            self.floor_blocked_since = None
        if request > self.requested:
            self.releases.append(self.engine.get_group_release())
            if self.sampler is not None:
                self.sampler.release_before_step.append(len(self.sampler.steps) + 1)
                self.sampler.release_monotonic.append(now)
            self.requested = request

    def stats(self) -> dict:
        stats = self.engine.get_stats()
        elapsed = self.floor_wait_seconds
        if self.floor_blocked_since is not None:
            elapsed += time.monotonic() - self.floor_blocked_since
        return {**stats, "floor_waits": self.floor_waits, "floor_wait_s": elapsed}

    def evidence(self) -> dict:
        return {
            **json.loads((self.path / "producer.json").read_bytes()),
            "rank": self.rank,
            "width": self.width,
            "slots": self.slots,
            "cap": self.cap,
            "engine": self.stats(),
            "releases": self.releases,
        }


class _Deadline:
    def __init__(self, spec: dict, out: Path, clock: Any) -> None:
        self.spec = spec
        self.out = out
        self.clock = clock
        self.cutoff = spec["deadline_seconds"] - spec.get(
            "submission_margin_seconds", 0
        )
        self.events: list[dict] = []
        self.lock = threading.Lock()
        self.timer = threading.Timer(max(0, self.cutoff - clock()), self.seal)
        self.timer.daemon = True
        self.timer.start()

    def close(self) -> None:
        self.timer.cancel()
        self.timer.join()

    def record(self, event: dict) -> bool:
        with self.lock:
            self.events.append({**event, "result_seconds": self.clock()})
        _write(self.out / "development.json", self.events)
        return (
            len(self.events) >= 2
            and self.events[-2]["successes"] >= self.spec["dev_min_successes"]
            and event["successes"]
            >= self.spec.get("submit_dev_successes", float("inf"))
            and self.events[-1]["result_seconds"] <= self.cutoff
        )

    def seal(self) -> None:
        with self.lock:
            decision = self.out / "deadline.json"
            if decision.exists():
                return
            eligible = [
                b
                for a, b in zip(self.events, self.events[1:])
                if b["result_seconds"] <= self.cutoff
                and min(a["successes"], b["successes"])
                >= self.spec["dev_min_successes"]
            ]
            best = max(
                eligible,
                key=lambda e: (e["successes"], -e["goal_distance"], -e["step"]),
                default=None,
            )
            receipt = {"budget_seconds": self.spec["deadline_seconds"], "chosen": best}
            if "submission_margin_seconds" in self.spec:
                receipt.update(
                    selection_cutoff_seconds=self.cutoff, locked_seconds=self.clock()
                )
            _write(decision, receipt)
            if self.spec.get("publish_receipt"):
                _write(
                    self.out / "publication.json",
                    {
                        "published_seconds": self.clock(),
                        "decision_sha256": hashlib.sha256(
                            decision.read_bytes()
                        ).hexdigest(),
                    },
                )


def _calibration_rows(engine: Any) -> dict:
    from ._grouped_ddp import QPOS

    initial = engine.snapshot_group(1, ["valid", QPOS, "actions"])
    mask = initial["valid"]
    return {
        "state": initial[QPOS][mask],
        "action": initial["actions"][mask],
        "frames": mask.nonzero()[:, 1],
        "lengths": mask.sum(1),
    }


def _calibration_stats(
    samples: list[dict | None], elapsed: bool
) -> tuple[dict, float | None]:
    import torch

    rows = [sample for sample in samples if sample is not None]

    def merged(key: str) -> torch.Tensor:
        return torch.cat([sample[key] for sample in rows])

    state = merged("state")
    action = merged("action")
    mean = [state.mean(0)]
    std = [state.std(0, correction=0)]
    time_scale = None
    if elapsed:
        time_scale = float(merged("lengths").float().median())
        progress = (merged("frames").float() / time_scale).clamp_max(1)
        mean.append(progress.mean().reshape(1))
        std.append(progress.std(correction=0).clamp_min(0.05).reshape(1))
    return {
        "observation.state": {"mean": torch.cat(mean), "std": torch.cat(std)},
        "action": {"mean": action.mean(0), "std": action.std(0, correction=0)},
    }, time_scale


def _qualification(spec: dict, seed: int, path: Path) -> tuple:
    producer_seed = spec["producer_seed_base"] + 1000 * spec["seeds"].index(seed)
    train = list(range(producer_seed, producer_seed + 4 * spec["episodes"]))
    dev = list(range(spec["dev_base"], spec["dev_base"] + spec["dev_scenes"]))
    test = list(range(spec["test_base"], spec["test_base"] + spec["test_scenes"]))
    report = json.loads(path.read_bytes())
    panels = [set(train), set(dev), set(test), {v["seed"] for v in report["records"]}]
    if (
        report["mode"] != "qualify"
        or report["task_identity"] != task_identity()
        or len(panels[-1]) != len(report["records"])
        or len(panels[-1]) < 100
        or sum(row["success"] for row in report["records"]) / len(panels[-1]) < 0.95
        or any(a & b for i, a in enumerate(panels) for b in panels[i + 1 :])
    ):
        raise ValueError("Expert qualification or scene separation failed")
    return producer_seed, train, dev, test, report["schema"]


def _policy(schema: dict, spec: dict, device: Any, stats: dict, seed: int) -> tuple:
    import numpy as np
    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.act.processor_act import make_act_pre_post_processors
    from lerobot.utils.constants import IMAGENET_STATS
    from embodichain.extensions.models.policy.act.training import _act_config

    from ._grouped_ddp import IMAGE

    names = list(schema["observation"]["names"])
    if spec.get("elapsed_control_input", False):
        names.append("elapsed_control_fraction")
    feature_schema = {
        **schema,
        "observation": {**schema["observation"], "names": names},
    }
    side = spec.get("backbone_image_side", 224)
    # The common preset validates the offline command's single-device interface.
    cfg = _act_config(feature_schema, IMAGE, (side, side, 3), "cuda:0", False)
    cfg.device = str(device)
    cfg.dim_model = 128
    cfg.n_heads = 4
    cfg.dim_feedforward = 512
    cfg.n_encoder_layers = 3
    cfg.n_decoder_layers = 1
    cfg.n_vae_encoder_layers = 2
    cfg.dropout = 0.05
    cfg.optimizer_lr = 3e-4
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    policy = ACTPolicy(cfg).to(device)
    stats = {
        IMAGE: {key: torch.tensor(value) for key, value in IMAGENET_STATS.items()},
        **stats,
    }
    pre, post = make_act_pre_post_processors(cfg, stats)
    settings = cfg.get_optimizer_preset()
    return policy, pre, post, settings.build(policy.get_optim_params()), settings


def _inputs(batch: Any, device: Any, side: int, time_scale: float | None) -> dict:
    import torch
    from embodichain.extensions.models.policy.act.runtime import _backbone_image

    from ._grouped_ddp import IMAGE, QPOS, RGB

    rgb = (
        batch[RGB]
        .to(device)[..., :3]
        .permute(0, 3, 1, 2)
        .to(dtype=torch.float32)
        .div_(255)
    )
    state = batch[QPOS].to(device)
    if time_scale is not None:
        tick = (batch["ods_frame_index"].to(rgb).reshape(-1, 1) / time_scale).clamp_max(
            1
        )
        state = torch.cat((state, tick), dim=-1)
    return {
        IMAGE: _backbone_image(rgb, (side, side)),
        "observation.state": state,
        "action": batch["actions"],
        "action_is_pad": batch["action_is_pad"],
    }


def _checkpoint(
    path: Path,
    policy: Any,
    pre: Any,
    post: Any,
    *,
    spec: dict,
    schema: dict,
    train: list[int],
    dev: list[int],
    step: int,
    time_scale: float | None,
    smoke: bool,
) -> str:
    from embodichain.extensions.models.policy.act import policy_fingerprint

    from ._grouped_ddp import IMAGE

    path.mkdir()
    policy.save_pretrained(path)
    pre.save_pretrained(path)
    post.save_pretrained(path)
    fingerprint = policy_fingerprint(path)
    manifest = {
        "recipe": "lerobot-act-v1",
        "image_key": IMAGE,
        "schema": schema,
        "task_identity": task_identity(),
        "scene_seeds": {"train": train, "validation": []},
        "backbone_initialization": policy.config.pretrained_backbone_weights,
        "selection": {"panel": "development", "seeds": dev},
        "update": step,
        "smoke": smoke,
        "policy_sha256": fingerprint,
        "query_period": spec.get("query_period", 1),
    }
    if time_scale is not None:
        manifest["elapsed_control_scale"] = time_scale
    _write(path / "embodichain_act.json", manifest)
    return fingerprint


def _train(
    ranks: Ranks,
    spec: dict,
    seed: int,
    out: Path,
    qual: Path,
    started: float,
    layout: str,
    smoke: bool,
) -> None:
    import torch
    from torch.nn.parallel import DistributedDataParallel

    from ._grouped_ddp import BatchTransfer, MaskedACTLoss, QPOS, RGB, training_batch
    from ._study_panel import Evaluator

    def clock() -> float:
        return time.monotonic() - started

    root = ranks.rank == 0
    producers = ranks.size if layout == "per-rank" else 1
    lanes, slots, cap, local_batch = (
        spec[key] // parts
        for key, parts in (
            ("producer_envs", producers),
            ("shared_slots", producers),
            ("episodes", producers),
            ("batch_size", ranks.size),
        )
    )
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True

    with ranks.stage("qualification and device setup"):
        if (
            (layout == "per-rank" and (ranks.size != 2 or spec["producer_envs"] != 10))
            or any(
                spec[key] % parts
                for key, parts in (
                    ("producer_envs", producers),
                    ("shared_slots", producers),
                    ("episodes", producers),
                    ("batch_size", ranks.size),
                )
            )
            or lanes < 1
            or slots < 2 * lanes
            or slots % lanes
            or cap < 3 * lanes
            or cap % lanes
            or local_batch < 1
        ):
            raise ValueError(
                "Grouped ACT requires whole producer groups and equal positive rank batches"
            )
        record_draws = spec.get("retired_history", False) or spec.get(
            "record_group_draws", False
        )
        if record_draws and ranks.size != 1:
            raise ValueError(
                "The ACT replay and per-step draw journal require a single GPU"
            )
        producer_seed, train, dev, test, schema = _qualification(spec, seed, qual)
        device_info = {
            "rank": ranks.rank,
            "pid": os.getpid(),
            "gpu_uuid": _gpu_uuid(ranks.device),
        }
    devices = ranks.gather(device_info)

    with ExitStack() as cleanup:
        producer = panel = deadline = None
        with ranks.stage("study setup"):
            if root:
                if len({record["gpu_uuid"] for record in devices}) != ranks.size:
                    raise ValueError("Use a distinct physical GPU for each rank")
                _write(
                    out / "recipe.json",
                    {
                        "seed": seed,
                        "protocol": spec,
                        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
                        "world_size": ranks.size,
                        "producer_layout": layout,
                        "ranks": devices,
                        "smoke": smoke,
                        "task_identity": task_identity(),
                        "clock_origin_monotonic": started,
                        "source_hashes": _source_hashes(),
                    },
                )
                _write(out / "development.json", [])
                if not smoke:
                    deadline = _Deadline(spec, out, clock)
                    cleanup.callback(deadline.close)
                panel = Evaluator(spec["eval_envs"], spec["arena_spacing"])
                cleanup.callback(panel.close)
            if ranks.rank < producers:
                path = out / f"producer_{ranks.rank}"
                path.mkdir()
                engine = make_engine(
                    path,
                    producer_seed,
                    slots,
                    lanes=lanes,
                    spacing=spec["arena_spacing"],
                    floor=spec["draw_floor"],
                    stratified_resets=spec.get("stratified_resets", False),
                    device=str(ranks.device),
                    logical_width=spec["producer_envs"],
                    lane_offset=ranks.rank * lanes,
                )
                cleanup.callback(engine.stop)
                producer = _Producer(engine, path, ranks.rank, lanes, slots, cap)

        with ranks.stage("ODS warmup"):
            if producer is not None:
                producer.engine.start()
        ready = clock()
        with ranks.stage("first-publication calibration"):
            calibration = _calibration_rows(producer.engine) if producer else None
            if panel is not None:
                panel.receive()
        samples = ranks.gather(calibration)
        with ranks.stage("calibration aggregation"):
            normalized = (
                _calibration_stats(samples, spec.get("elapsed_control_input", False))
                if root
                else None
            )
        stats, time_scale = ranks.broadcast(normalized)
        del calibration, samples, normalized

        with ranks.stage("ACT initialization"):
            policy, pre, post, opt, opt_cfg = _policy(
                schema, spec, ranks.device, stats, seed
            )
        model = MaskedACTLoss(policy, ranks.size)
        if ranks.size == 2:
            model = DistributedDataParallel(
                model, device_ids=[ranks.device.index], broadcast_buffers=False
            )
            model._set_ddp_runtime_logging_sample_rate(20)
        torch.manual_seed(seed + ranks.rank)
        rng = torch.Generator().manual_seed(1_000_000 + seed + ranks.rank)
        fields = (
            [RGB, QPOS, "ods_frame_index"] if time_scale is not None else [RGB, QPOS]
        )
        transfer = (
            BatchTransfer(ranks, local_batch, fields)
            if ranks.size > producers
            else None
        )
        draw = None
        if record_draws:
            from ._grouped_replay import GroupReplay

            producer.sampler = GroupReplay(
                producer.engine,
                seed,
                spec.get("retired_history", False),
                local_batch,
                policy.config.chunk_size,
                fields,
                rng,
            )
            draw = producer.sampler.draw
        elif producer is not None:

            def draw():
                return producer.engine.group_batch(
                    spec["batch_size"] // producers,
                    policy.config.chunk_size,
                    anchor_fields=fields,
                    generator=rng,
                )

        first = None
        step = 0
        early = False
        smoke_checkpoint = None
        cutoff = spec["deadline_seconds"] - spec.get("submission_margin_seconds", 0)
        window = []
        window_start = window_end = None
        source_start = source_end = None
        nccl = {}
        for step_idx in range(1, spec["updates"] + 1):
            if not ranks.decision(not early and clock() < cutoff):
                break
            measured = 101 <= step_idx <= 600
            if step_idx == 101:
                window_start = clock()
                source_start = producer.stats() if producer else None
            timings: dict[str, float] = {}
            wait_start = ranks.wait_seconds
            with ranks.stage("ODS refill"):
                if producer is not None:
                    begin = time.perf_counter()
                    producer.refill()
                    timings["refill_s"] = time.perf_counter() - begin
            with training_batch(ranks, draw, transfer, timings) as batch:
                with ranks.stage("ACT preprocessing"):
                    begin = time.perf_counter()
                    count = (
                        int((~batch["action_is_pad"]).sum())
                        * batch["actions"].shape[-1]
                    )
                    inputs = pre(
                        _inputs(
                            batch,
                            ranks.device,
                            spec.get("backbone_image_side", 224),
                            time_scale,
                        )
                    )
                    timings["preprocess_s"] = time.perf_counter() - begin
                global_valid = ranks.sum_int(count)
                with ranks.stage("ACT forward"):
                    begin = time.perf_counter()
                    loss = model(inputs, global_valid)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("ACT loss is nonfinite")
                    timings["forward_s"] = time.perf_counter() - begin
                with ranks.stage("ACT optimizer"):
                    begin = time.perf_counter()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        policy.parameters(),
                        opt_cfg.grad_clip_norm,
                        error_if_nonfinite=True,
                    )
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.synchronize(ranks.device)
                    timings["optimizer_s"] = time.perf_counter() - begin
            step = step_idx
            if first is None:
                first = clock()
            if measured:
                window_end = clock()
                timings["rank_wait_s"] = ranks.wait_seconds - wait_start
                window.append(timings)
            if step == 600:
                source_end = producer.stats() if producer else None
            if ranks.size == 2 and 102 <= step <= 601:
                ddp_log = model._get_ddp_logging_data()
                iteration = ddp_log.get("iteration", 0)
                if 101 <= iteration <= 600 and "backward_comm_time" in ddp_log:
                    nccl[iteration] = ddp_log["backward_comm_time"] / 1e9

            if smoke or step in spec["checkpoints"]:
                own_episodes = (
                    producer.engine.get_flow_status()["successful_episodes"]
                    if producer
                    else 0
                )
                published = ranks.sum_int(own_episodes)
                if smoke and published < 3 * spec["producer_envs"]:
                    continue
                if (
                    smoke
                    and ranks.sum_int(
                        int(producer is not None and own_episodes >= 3 * lanes)
                    )
                    != producers
                ):
                    continue
                with ranks.stage("checkpoint and development evaluation"):
                    if root:
                        ck = out / f"step_{step:06d}"
                        scenes = dev[:2] if smoke else dev
                        fingerprint = _checkpoint(
                            ck,
                            policy,
                            pre,
                            post,
                            spec=spec,
                            schema=schema,
                            train=train,
                            dev=scenes,
                            step=step,
                            time_scale=time_scale,
                            smoke=smoke,
                        )
                        saved = clock()
                        result = panel.evaluate(
                            ck,
                            scenes,
                            "development",
                            progress=(
                                producer.refill
                                if ranks.size == 1
                                and spec.get("refill_during_dev", True)
                                else None
                            ),
                        )
                        if result["policy_sha256"] != fingerprint:
                            raise RuntimeError(
                                "Development evaluator loaded a different checkpoint"
                            )
                        _write(out / f"dev_{step}.json", result)
                        if smoke:
                            smoke_checkpoint = ck.name
                            early = True
                        else:
                            early = deadline.record(
                                dict(
                                    step=step,
                                    checkpoint=ck.name,
                                    sha256=fingerprint,
                                    successes=result["successes"],
                                    goal_distance=result["mean_goal_distance"],
                                    published_episodes=published,
                                    save_seconds=saved,
                                )
                            )
        with ranks.stage("publish decision"):
            if root:
                if smoke and smoke_checkpoint is None:
                    raise RuntimeError(
                        "Smoke run did not exercise a replacement on every producer"
                    )
                if deadline is not None:
                    deadline.close()
                    deadline.seal()

        with ranks.stage("producer shutdown"):
            evidence = None
            if producer is not None:
                producer.engine.stop()
                evidence = producer.evidence()
                if producer.sampler is not None:
                    _write(out / "sampling.json", producer.sampler.evidence())
        summaries = ranks.gather(
            {
                "rank": ranks.rank,
                "committed_draws": step * local_batch,
                "producer": evidence,
                "window": {
                    "start_s": window_start,
                    "end_s": window_end,
                    "steps": len(window),
                    "phase_totals_s": {
                        key: sum(row.get(key, 0.0) for row in window)
                        for key in (
                            "refill_s",
                            "sample_s",
                            "transfer_s",
                            "preprocess_s",
                            "forward_s",
                            "optimizer_s",
                            "rank_wait_s",
                        )
                    },
                    "nccl_backward_s": list(nccl.values()),
                    "source_start": source_start,
                    "source_end": source_end,
                },
            }
        )
        with ranks.stage("write supply evidence"):
            if root:
                _write(
                    out / "supply.json",
                    {
                        "step": step,
                        "source_ready": ready,
                        "first_update": first,
                        "ranks": summaries,
                        "producers": [
                            record["producer"]
                            for record in summaries
                            if record["producer"]
                        ],
                        "test_scenes": test,
                        "smoke_checkpoint": smoke_checkpoint,
                    },
                )


def _verify(out: Path, spec: dict, seed: int, smoke: bool) -> None:
    from ._study_panel import IndependentEvaluators

    supply = json.loads((out / "supply.json").read_bytes())
    if smoke:
        _write(
            out / "metrics.json",
            {"seed": seed, "smoke_checkpoint": supply["smoke_checkpoint"]},
        )
        return
    decision = json.loads((out / "deadline.json").read_bytes())
    chosen = decision["chosen"]
    published = decision.get("locked_seconds", 0.0)
    if spec.get("publish_receipt"):
        published = json.loads((out / "publication.json").read_bytes())[
            "published_seconds"
        ]
    successes = None
    if chosen:
        evaluators = IndependentEvaluators(spec["verification_workers"])
        try:
            evaluators.receive()
            result = evaluators.evaluate(
                out / chosen["checkpoint"], supply["test_scenes"], "test"
            )
        finally:
            evaluators.close()
        if result["policy_sha256"] != chosen["sha256"]:
            raise RuntimeError("Selected weights changed")
        _write(out / "test.json", result)
        successes = result["successes"]
    _write(
        out / "metrics.json",
        dict(
            seed=seed,
            selected=chosen,
            reserved_successes=successes,
            pass_gate=successes is not None
            and successes >= spec["required_reserved_successes"]
            and published <= spec["deadline_seconds"],
        ),
    )


def main() -> None:
    """Run one seed of the acquisition-inclusive grouped study."""
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clock-origin", type=float, default=None)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).with_name("grouped_elapsed_protocol.json"),
    )
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument(
        "--producer-layout", choices=("per-rank", "shared"), default="shared"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run until every producer recycles a group, then reload a checkpoint",
    )
    args = parser.parse_args()
    spec = json.loads(args.protocol.read_bytes())
    if args.clock_origin is not None:
        if not 0 < args.clock_origin <= started:
            parser.error("clock origin must precede CLI entry")
        started = args.clock_origin
    elif spec.get("require_cold_launcher"):
        parser.error("Use the timed_grouped launcher for this protocol")

    from ._grouped_ddp import grouped_ranks

    def interrupted(signum: int, _frame: Any) -> None:
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    rank = int(os.environ.get("RANK", 0))
    output_claimed = False
    try:
        with grouped_ranks() as ranks:
            with ranks.stage("output directory admission"):
                if rank == 0:
                    args.output.mkdir(parents=True, exist_ok=False)
            output_claimed = True
            _train(
                ranks,
                spec,
                args.seed,
                args.output,
                args.qualification,
                started,
                args.producer_layout,
                args.smoke,
            )
        if rank == 0:
            _verify(args.output, spec, args.seed, args.smoke)
    except (Exception, SystemExit, KeyboardInterrupt) as error:
        if output_claimed:
            _write(
                args.output / f"failure_rank_{rank}.json",
                {
                    "rank": rank,
                    "error": str(error),
                    "seconds": time.monotonic() - started,
                },
            )
        raise


if __name__ == "__main__":
    main()
