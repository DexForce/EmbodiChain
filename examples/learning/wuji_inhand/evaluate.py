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
"""Independent native-physics qualification of a Wuji20 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from _dependency import _activate

__all__ = ["main"]


def main() -> None:
    """Retain physical pose/reward traces and enforce each size's success gates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--seeds", type=int, nargs="+", default=[91001])
    parser.add_argument(
        "--stage",
        choices=["development", "validation", "holdout"],
        default="development",
    )
    parser.add_argument("--zero-action", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Use a fresh evaluation directory")
    output.mkdir(parents=True)
    _activate(args.upstream.resolve(), output / "learner_dependency")

    import torch
    from _task import _WujiTask
    from native_full import _evaluate, _wait_memory
    from omegaconf import OmegaConf
    from sharpa_rl_unilab.training.configuration import configure_threads
    from sharpa_rl_unilab.training.policy import make_actor

    snapshot = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if (
        not snapshot.get("native_environment")
        or snapshot["native_identity"].get("physical_dof") != 20
    ):
        raise ValueError("Expected a native-trained20-DOF Wuji checkpoint")
    experiment = json.loads((args.checkpoint.parent / "experiment.json").read_text())
    inputs = json.loads(
        (args.checkpoint.parent / "runtime_inputs_manifest.json").read_text()
    )
    for name, digest in inputs["assets"].items():
        if hashlib.sha256((args.assets / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Physical asset changed since training: {name}")
    repository = Path(__file__).resolve().parents[3]
    for name, digest in inputs["core"].items():
        if hashlib.sha256((repository / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Runtime source changed since training: {name}")
    hashes = {}
    for name in ["_task.py", "_physics.py", "_dependency.py", "_asset.py"]:
        current = hashlib.sha256(
            (Path(__file__).parent / name).read_bytes()
        ).hexdigest()
        if current != experiment["source_hashes"][name]:
            raise ValueError(f"Physical source changed since training: {name}")
        hashes[name] = current
    cfg = OmegaConf.create(snapshot["config"])
    configure_threads(cfg)
    actor = make_actor(cfg, "cuda:0", stage="teacher")
    actor.load_state_dict(snapshot["actor"], strict=True)
    if args.zero_action:

        class ZeroDistribution:
            def __init__(self, batch: int) -> None:
                self.batch = batch

            def deterministic(self) -> torch.Tensor:
                return torch.zeros(self.batch, 20, device="cuda:0")

        actor.policy = lambda data: ZeroDistribution(data.batch_size[0])
    _wait_memory("cuda:0", 6.0)

    class RecordedTask(_WujiTask):
        def __init__(self, *a: Any, **kw: Any) -> None:
            self.initial_states = []
            self.initial_contexts = []
            self.physical_states = []
            self.measured_rewards = []
            self.grasp_flags = []
            self.tip_positions = []
            super().__init__(*a, **kw)
            self.initial_states.clear()
            self.initial_contexts.clear()

        def reset(self, *a: Any, **kw: Any) -> Any:
            result = super().reset(*a, **kw)
            self.initial_states.append(self.s.detach().cpu().clone())
            self.initial_contexts.append(self.xi.detach().cpu().clone())
            return result

        def step(self, action: torch.Tensor) -> Any:
            _wait_memory("cuda:0", 4.0)
            result = super().step(action)
            self.physical_states.append(self.s.detach().cpu().clone())
            self.measured_rewards.append(result.reward.detach().cpu().clone())
            self.grasp_flags.append(self.grasp_valid.detach().cpu().clone())
            self.tip_positions.append(
                self.physical.last_tip_positions.detach().cpu().clone()
            )
            return result

    task = None
    try:
        task = RecordedTask(
            cfg,
            args.assets.resolve(),
            output / "physical",
            num_envs=args.num_envs,
            seed=777,
            gain=float(experiment["gain"]),
        )
        result = _evaluate(task, actor, args.seeds, output)
        traces = torch.load(
            output / "physical_rotation_traces.pt",
            map_location="cpu",
            weights_only=False,
        )
        for batch, seed in enumerate(args.seeds):
            valid = torch.stack(task.grasp_flags[batch * 400 : (batch + 1) * 400])
            fraction = (valid * traces[batch][..., 1]).sum(0) / 400
            for record in result["records"]:
                if record["seed"] == seed:
                    record["valid_grasp_fraction"] = float(fraction[record["row"]])
                    record["stable"] &= record["valid_grasp_fraction"] >= 0.8

        def update_semantics(value: dict, rows: list[dict]) -> None:
            value["stable"] = sum(row["stable"] for row in rows) / len(rows)
            value["valid_grasp_fraction"] = sum(
                row["valid_grasp_fraction"] for row in rows
            ) / len(rows)
            value["passed"] = bool(
                value["dropped"] <= 0.05
                and value["stable"] >= 0.9
                and value["median_angle_rad"] >= 2 * torch.pi
                and value["positive_fraction"] >= 0.8
                and value["tail_positive_fraction"] >= 0.8
                and value["valid_grasp_fraction"] >= 0.8
            )

        for scale, value in result["by_scale"].items():
            update_semantics(
                value, [r for r in result["records"] if r["scale"] == float(scale)]
            )
        update_semantics(result["summary"], result["records"])
        result["summary"]["overall_passed"] = result["summary"]["passed"]
        result["summary"]["all_scales_passed"] = all(
            value["passed"] for value in result["by_scale"].values()
        )
        result["summary"]["passed"] &= result["summary"]["all_scales_passed"]
        result["qualification_version"] = "palm-up-v2-opposing-pads"
        (output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
        torch.save(
            {
                "initial_states": task.initial_states,
                "initial_contexts": task.initial_contexts,
                "physical_states": task.physical_states,
                "measured_rewards": task.measured_rewards,
                "seeds": args.seeds,
                "dt": 0.05,
                "state_layout": "position3,quatWXYZ4,linear3,angular3,q20,qvel20,tactile10,target20",
                "reward_default_hand_pose": task.mdp.default.cpu(),
                "effort_limit": task.mdp.effort.cpu(),
                "tip_positions": task.tip_positions,
                "grasp_flags": task.grasp_flags,
                "grasp_rule": "thumb force>.02N plus two opposing non-thumb forces>.02N; radial dot<-.15; pad centers within80% object half-height; grace20 control steps",
            },
            output / "measured_states.pt",
        )
        provenance = {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": hashlib.sha256(
                args.checkpoint.read_bytes()
            ).hexdigest(),
            "hand": "Wuji Hand V1 right",
            "physical_dof": 20,
            "native_environment": True,
            "source_hashes_match_training": True,
            "assets_and_runtime_match_recorded_training_inputs": True,
            "physical_hashes": hashes,
            "evaluation_seeds": args.seeds,
            "training_iteration": snapshot["iteration"],
            "episodes": args.num_envs * len(args.seeds),
            "qualification_stage": args.stage,
            "grasp_qualification_passed": bool(result["summary"]["passed"]),
            "native_policy_qualified": bool(
                result["summary"]["passed"]
                and not args.zero_action
                and args.stage == "holdout"
                and args.num_envs * len(args.seeds) >= 1536
            ),
            "zero_action_control": args.zero_action,
            "scope": experiment["scope"],
            "task_identity": task.identity,
        }
        (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        print(json.dumps(result["summary"]), flush=True)
    finally:
        if task is not None:
            task.close()


if __name__ == "__main__":
    main()
