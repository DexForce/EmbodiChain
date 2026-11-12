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

"""Independently qualify a native-trained checkpoint and retain physical states."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from _native_task import _NativeTask
from native import _evaluate
from omegaconf import OmegaConf
from sharpa_rl_unilab.training.configuration import configure_threads
from sharpa_rl_unilab.training.policy import make_actor
from unilab.base.torch_env import TorchEnvState

__all__ = ["main"]


class _RecordedTask(_NativeTask):
    """Retain measured states for audit and read-only video rendering."""

    def __init__(
        self, cfg: Any, upstream: Path, output: Path, *, num_envs: int, seed: int
    ) -> None:
        self.initial_states = []
        self.physical_states = []
        self.measured_rewards = []
        self.initial_contexts = []
        super().__init__(cfg, upstream, output, num_envs=num_envs, seed=seed)
        self.initial_states.clear()
        self.initial_contexts.clear()

    def reset(
        self, env_indices: torch.Tensor | None = None, *, seed: int | None = None
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        result = super().reset(env_indices, seed=seed)
        self.initial_states.append(self.s.detach().cpu().clone())
        self.initial_contexts.append(self.xi.detach().cpu().clone())
        return result

    def step(self, action: torch.Tensor) -> TorchEnvState:
        result = super().step(action)
        self.physical_states.append(self.s.detach().cpu().clone())
        self.measured_rewards.append(result.reward.detach().cpu().clone())
        return result


def main() -> None:
    """Evaluate a fresh native checkpoint without changing training state."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=[10001, 10002, 10003])
    parser.add_argument("--allow-reference-diagnostic", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Use a fresh physical evaluation output")
    output.mkdir(parents=True)
    snapshot = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    native = bool(snapshot.get("native_environment", False))
    if not native and not args.allow_reference_diagnostic:
        raise ValueError(
            "Native training qualification requires a native-trained checkpoint"
        )
    code_hashes = {
        name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in [
            "native_evaluate.py",
            "native.py",
            "_native_task.py",
            "_native_asset.py",
            "_simulation.py",
        ]
    }
    if native:
        training_experiment = json.loads(
            (args.checkpoint.parent / "experiment.json").read_text()
        )
        for name in ["_native_task.py", "_native_asset.py", "_simulation.py"]:
            if code_hashes[name] != training_experiment["source_hashes"][name]:
                raise ValueError(
                    f"Physical environment source changed since training: {name}"
                )
    cfg = OmegaConf.create(snapshot["config"])
    cfg.training.device = "cuda:0"
    configure_threads(cfg)
    torch.manual_seed(777)
    actor = make_actor(cfg, "cuda:0", stage="teacher")
    actor.load_state_dict(snapshot["actor"], strict=True)
    task = None
    try:
        task = _RecordedTask(
            cfg, args.upstream.resolve(), output, num_envs=args.num_envs, seed=777
        )
        result = _evaluate(task, actor, args.seeds, output)
        provenance = {
            "checkpoint": str(args.checkpoint.resolve()),
            "evaluation_code_sha256": code_hashes,
            "checkpoint_sha256": hashlib.sha256(
                args.checkpoint.read_bytes()
            ).hexdigest(),
            "checkpoint_trained_in_native_environment": native,
            "training_iteration": snapshot.get("iteration"),
            "native_training_success_qualified": native and result["summary"]["passed"],
            "full_upstream_baseline_qualified": False,
            "scope": "Fixed scale1.0 and fixed parameters; native contact-force tactile approximation",
            "evaluation_seeds": args.seeds,
            "episodes": args.num_envs * len(args.seeds),
        }
        (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        torch.save(
            {
                "initial_states": task.initial_states,
                "physical_states": task.physical_states,
                "measured_rewards": task.measured_rewards,
                "initial_contexts": task.initial_contexts,
                "seeds": args.seeds,
                "dt": task.step_dt,
                "state_layout": "position3, quaternionWXYZ4, linear3, angular3, q22, qvel22, tactile10, target22",
            },
            output / "measured_states.pt",
        )
    finally:
        if task is not None:
            task.close()


if __name__ == "__main__":
    main()
