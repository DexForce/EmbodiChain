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

"""Train native visual ACT directly from grouped ODS on one GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import threading
import time
from typing import Any
from contextlib import ExitStack

from ._environment import task_config, task_identity


def seeded_expert(env: Any, **kwargs: Any) -> Any:
    """Give each attempt a distinct deterministic scene in the worker process."""
    from embodichain.lab.gym.envs.demo import execute_demo_episode

    raw = env.unwrapped
    spec = raw.cfg.extensions["act_study"]
    seed = getattr(raw, "_act_next_seed", spec["seed_base"])
    raw._act_next_seed = seed + raw.num_envs
    # The physical reset pipeline still positions the cube, its relative goal
    # and the marker. Only the training producer narrows each lane's draw.
    strata = list(range(10))
    stratified = spec.get("stratified_resets", False)
    if stratified:
        if raw.num_envs != len(strata):
            raise ValueError("The 3x3 plus uniform design requires ten producer lanes")
        import copy

        random.Random(seed).shuffle(strata)
        manager = raw.event_manager
        original = manager.get_functor_cfg("randomize_cube")
        low, high = original.params["position_range"]
    try:
        for lane in range(raw.num_envs):
            if stratified:
                cell = strata[lane]
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
            env.reset(
                seed=seed + lane, options={"save_data": False, "reset_ids": [lane]}
            )
    finally:
        if stratified:
            manager.set_functor_cfg("randomize_cube", original)
    xy = raw.sim.get_rigid_object("cube").get_local_pose()[:, :2].tolist()
    result = execute_demo_episode(env, **kwargs)
    with Path(spec["attempts"]).open("a") as stream:
        for lane, length in enumerate(result.lengths):
            stream.write(
                json.dumps(
                    {
                        "seed": seed + lane,
                        "initial_cube_xy": xy[lane],
                        "stratum": strata[lane] if stratified else None,
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
) -> Any:
    """Start an explicit single-renderer producer with raw PushCube policy requests."""
    from embodichain.data_pipeline.engine.data import (
        OnlineDataEngine,
        OnlineDataEngineCfg,
    )

    cfg = task_config("cuda:0", camera=True, num_envs=lanes, spacing=spacing)
    cfg["seed"] = seed
    cfg["env"]["extensions"]["act_study"] = {
        "seed_base": seed,
        "attempts": str(path / "attempts.jsonl"),
        "stratified_resets": stratified_resets,
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


RGB = ("obs", "sensor", "cam_high", "color")
QPOS = ("obs", "robot", "qpos")
IMAGE = "observation.images.cam_high"


def _write(path: Path, value: Any) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def _run(spec: dict, seed: int, out: Path, qual: Path, started: float) -> None:
    import numpy as np
    import torch
    from accelerate import Accelerator
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.act.processor_act import make_act_pre_post_processors
    from lerobot.scripts.lerobot_train import update_policy
    from lerobot.utils.logging_utils import MetricsTracker, AverageMeter
    from lerobot.utils.constants import IMAGENET_STATS
    from embodichain.extensions.models.policy.act import policy_fingerprint
    from embodichain.extensions.models.policy.act.runtime import _image_state
    from embodichain.extensions.models.policy.act.training import _act_config
    from ._study_panel import Evaluator, IndependentEvaluators

    def clock() -> float:
        return time.monotonic() - started

    idx = spec["seeds"].index(seed)
    producer_seed = spec["producer_seed_base"] + 1000 * idx
    dev = list(range(spec["dev_base"], spec["dev_base"] + spec["dev_scenes"]))
    test = list(range(spec["test_base"], spec["test_base"] + spec["test_scenes"]))
    train = list(range(producer_seed, producer_seed + 4 * spec["episodes"]))
    q = json.loads(qual.read_bytes())
    panels = [set(train), set(dev), set(test), {v["seed"] for v in q["records"]}]
    if (
        q["mode"] != "qualify"
        or q["task_identity"] != task_identity()
        or len(panels[-1]) != len(q["records"])
        or len(panels[-1]) < 100
        or sum(v["success"] for v in q["records"]) / len(panels[-1]) < 0.95
        or any(a & b for i, a in enumerate(panels) for b in panels[i + 1 :])
    ):
        raise ValueError("Expert qualification or scene separation failed")
    out.mkdir(parents=True, exist_ok=False)
    _write(
        out / "recipe.json",
        {
            "seed": seed,
            "protocol": spec,
            "gpu": os.environ["CUDA_VISIBLE_DEVICES"],
            "task_identity": task_identity(),
            "clock_origin_monotonic": started,
            "source_hashes": {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [
                    Path(__file__),
                    Path(__file__).with_name("_study_panel.py"),
                    Path(__file__).parents[3]
                    / "embodichain/extensions/models/policy/act/runtime.py",
                    Path(__file__).parents[3] / "embodichain/lab/gym/envs/base_env.py",
                    Path(__file__).parents[3]
                    / "embodichain/data_pipeline/engine/data.py",
                ]
            },
        },
    )
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    deadline = spec["deadline_seconds"]
    selection_cutoff = deadline - spec.get("submission_margin_seconds", 0)
    events = []
    guard = threading.Lock()

    def seal():
        with guard:
            if (out / "deadline.json").exists():
                return
            eligible = [
                b
                for a, b in zip(events, events[1:])
                if b["result_seconds"] <= selection_cutoff
                and min(a["successes"], b["successes"]) >= spec["dev_min_successes"]
            ]
            best = max(
                eligible,
                key=lambda e: (e["successes"], -e["goal_distance"], -e["step"]),
                default=None,
            )
            receipt = {"budget_seconds": deadline, "chosen": best}
            if "submission_margin_seconds" in spec:
                receipt.update(
                    selection_cutoff_seconds=selection_cutoff, locked_seconds=clock()
                )
            _write(out / "deadline.json", receipt)

    timer = threading.Timer(max(0, selection_cutoff - clock()), seal)
    timer.daemon = True
    timer.start()
    with ExitStack() as cleanup:
        cleanup.callback(timer.cancel)
        panel = Evaluator(spec["eval_envs"], spec["arena_spacing"])
        panel_cleanup = cleanup.enter_context(ExitStack())
        panel_cleanup.callback(panel.close)
        width = spec["producer_envs"]
        engine = make_engine(
            out,
            producer_seed,
            spec["shared_slots"],
            lanes=width,
            spacing=spec["arena_spacing"],
            floor=spec["draw_floor"],
            stratified_resets=spec.get("stratified_resets", False),
        )
        worker_cleanup = cleanup.enter_context(ExitStack())
        worker_cleanup.callback(engine.stop)
        engine.start()
        ready = clock()
        panel.receive()
        initial = engine.snapshot_group(1, ["valid", RGB, QPOS, "actions"])
        mask = initial["valid"]
        initial_frames = int(mask.sum())
        stats = {IMAGE: {k: torch.tensor(v) for k, v in IMAGENET_STATS.items()}}
        for k, v in (
            ("observation.state", initial[QPOS]),
            ("action", initial["actions"]),
        ):
            data = v[mask]
            stats[k] = {"mean": data.mean(0), "std": data.std(0, correction=0)}
        marks = []
        for row in range(width):
            for images in initial[RGB][row, mask[row], :, :, :3].split(32):
                rgb = images.permute(0, 3, 1, 2).cuda()
                marks.append(_image_state(rgb, rgb.new_empty((len(rgb), 0))).cpu())
        z = torch.cat(marks)
        for k, v in (
            ("mean", z.mean(0)),
            ("std", z.std(0, correction=0).clamp_min(0.05)),
        ):
            stats["observation.state"][k] = torch.cat(
                (stats["observation.state"][k], v)
            )
        del initial, marks, data, rgb, images, z
        releases = []
        last_request = 1

        def refill():
            nonlocal last_request
            if last_request * width >= spec["episodes"]:
                return
            request = engine.try_refill_group()
            if request is not None and request > last_request:
                releases.append(engine.get_group_release())
                last_request = request

        refill()
        schema = q["schema"]
        feature_schema = {
            **schema,
            "observation": {
                **schema["observation"],
                "names": schema["observation"]["names"]
                + [
                    "red_u",
                    "green_u",
                    "red_v",
                    "green_v",
                    "red_visible",
                    "green_visible",
                ],
            },
        }
        cfg = _act_config(feature_schema, IMAGE, (224, 224, 3), "cuda:0", False)
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
        policy = ACTPolicy(cfg).to("cuda:0")
        pre, post = make_act_pre_post_processors(cfg, stats)
        opt_cfg = cfg.get_optimizer_preset()
        opt = opt_cfg.build(policy.get_optim_params())
        ac = Accelerator(mixed_precision="no", step_scheduler_with_optimizer=False)
        policy, opt = ac.prepare(policy, opt)
        batch = spec["batch_size"]
        meter = MetricsTracker(
            batch,
            initial_frames,
            width,
            {
                k: AverageMeter(k, ":.5f")
                for k in ("loss", "grad_norm", "lr", "update_s", "gpu_mem_gb")
            },
        )
        rng = torch.Generator().manual_seed(1000000 + seed)
        step = 0
        first = None
        for step_idx in range(1, spec["updates"] + 1):
            if clock() >= selection_cutoff:
                break
            refill()
            with engine.group_batch(
                batch, cfg.chunk_size, anchor_fields=[RGB, QPOS], generator=rng
            ) as x:
                rgb = (
                    x[RGB][..., :3]
                    .permute(0, 3, 1, 2)
                    .to("cuda:0", dtype=torch.float32)
                    .div_(255)
                )
                inputs = {
                    IMAGE: rgb,
                    "observation.state": _image_state(rgb, x[QPOS].cuda()),
                    "action": x["actions"],
                    "action_is_pad": x["action_is_pad"],
                }
                update_policy(
                    meter, policy, pre(inputs), opt, opt_cfg.grad_clip_norm, ac
                )
                torch.cuda.synchronize()
            step = step_idx
            if first is None:
                first = clock()
            if step not in spec["checkpoints"]:
                continue
            ck = out / f"step_{step:06d}"
            ck.mkdir()
            ac.unwrap_model(policy).save_pretrained(ck)
            pre.save_pretrained(ck)
            post.save_pretrained(ck)
            manifest = {
                "recipe": "lerobot-act-v1",
                "image_key": IMAGE,
                "schema": schema,
                "task_identity": task_identity(),
                "scene_seeds": {"train": train, "validation": []},
                "backbone_initialization": cfg.pretrained_backbone_weights,
                "selection": {"panel": "development", "seeds": dev},
                "update": step,
                "smoke": False,
                "policy_sha256": policy_fingerprint(ck),
                "image_landmarks": True,
                "query_period": spec.get("query_period", 1),
            }
            _write(ck / "embodichain_act.json", manifest)
            saved = clock()
            result = panel.evaluate(
                ck,
                dev,
                "development",
                progress=refill if spec.get("refill_during_dev", True) else None,
            )
            _write(out / f"dev_{step}.json", result)
            with guard:
                events.append(
                    dict(
                        step=step,
                        checkpoint=ck.name,
                        sha256=result["policy_sha256"],
                        successes=result["successes"],
                        goal_distance=result["mean_goal_distance"],
                        save_seconds=saved,
                        result_seconds=clock(),
                    )
                )
            _write(out / "development.json", events)
            if (
                "submit_dev_successes" in spec
                and len(events) >= 2
                and events[-2]["successes"] >= spec["dev_min_successes"]
                and result["successes"] >= spec["submit_dev_successes"]
                and events[-1]["result_seconds"] <= selection_cutoff
            ):
                break
        timer.cancel()
        timer.join()
        seal()
        _write(
            out / "supply.json",
            dict(
                releases=releases,
                engine=engine.get_stats(),
                step=step,
                source_ready=ready,
                first_update=first,
            ),
        )
        worker_cleanup.close()
        panel_cleanup.close()
        chosen = json.loads((out / "deadline.json").read_bytes())["chosen"]
        successes = None
        if chosen:
            ver = IndependentEvaluators(spec["verification_workers"])
            cleanup.callback(ver.close)
            ver.receive()
            result = ver.evaluate(out / chosen["checkpoint"], test, "test")
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
            and successes >= spec["required_reserved_successes"],
        ),
    )


def main() -> None:
    """Run one seed of the frozen, acquisition-inclusive grouped study."""
    started = time.monotonic()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--clock-origin",
        type=float,
        default=None,
        help="Earlier host monotonic timestamp supplied by the cold launcher",
    )
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).with_name("grouped_protocol.json"),
    )
    p.add_argument("--qualification", type=Path, required=True)
    args = p.parse_args()
    spec = json.loads(args.protocol.read_bytes())
    if args.clock_origin is not None:
        if not 0 < args.clock_origin <= started:
            p.error("clock origin must precede CLI entry")
        started = args.clock_origin
    elif spec.get("require_cold_launcher"):
        p.error("Use the timed_grouped launcher for this protocol")
    _run(
        spec,
        args.seed,
        args.output,
        args.qualification,
        started,
    )


if __name__ == "__main__":
    main()
