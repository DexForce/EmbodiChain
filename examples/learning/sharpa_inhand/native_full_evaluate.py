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
import subprocess
from pathlib import Path
from typing import Any

import torch
from _full_native_task import _FullNativeTask as _NativeTask
from _native_checkpoint import _runtime_source
from native_full import _evaluate, _wait_memory
from omegaconf import OmegaConf
from reference import _REVISION, _verify_imported_learner
from sharpa_rl_unilab.training.configuration import configure_threads
from sharpa_rl_unilab.training.policy import make_actor
from unilab.base.torch_env import TorchEnvState

__all__ = ["main"]


class _RecordedTask(_NativeTask):
    """Retain measured states for audit and read-only video rendering."""

    def __init__(
        self,
        cfg: Any,
        upstream: Path,
        output: Path,
        *,
        num_envs: int,
        seed: int,
        minimum_free: float = 4.0,
    ) -> None:
        self.initial_states = []
        self.physical_states = []
        self.measured_rewards = []
        self.initial_contexts = []
        self.minimum_free = minimum_free
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
        _wait_memory(str(self.device), self.minimum_free)
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
    parser.add_argument("--num-envs", type=int, default=80)
    parser.add_argument("--seeds", type=int, nargs="+", default=[10001, 10002, 10003])
    parser.add_argument("--allow-reference-diagnostic", action="store_true")
    parser.add_argument("--min-free-gpu-gib", type=float, default=4.0)
    parser.add_argument(
        "--allow-implementation-transfer",
        action="store_true",
        help="Diagnose a native checkpoint against this implementation; does not qualify unchanged training/evaluation physics.",
    )
    args = parser.parse_args()
    if args.min_free_gpu_gib < 0:
        raise ValueError("Minimum free GPU memory must be nonnegative")
    upstream = args.upstream.resolve()
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=upstream, text=True
    ).strip()
    if revision != _REVISION:
        raise ValueError("Expected the qualified upstream reference revision")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=upstream,
        text=True,
    ).strip():
        raise ValueError("The upstream reference sources must remain clean")
    _verify_imported_learner(upstream)
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
    physical_files = [
        "_full_native_task.py",
        "_native_task.py",
        "_simulation.py",
        "_native_asset.py",
    ]
    code_hashes = {
        name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in physical_files
    }
    adapter_files = [
        "objects/articulation.py",
        "physics/_joint_drive_view.py",
        "physics/_cylinder_variants.py",
        "physics/_contact_material_view.py",
        "physics/_source_joint_properties.py",
        "sensors/contact/_geometry_contact_sensor.py",
        "sensors/contact/_warp/geometry_contact.py",
    ]
    for name in adapter_files:
        path = _runtime_source(f"embodichain/lab/sim/{name}")
        code_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if native and not args.allow_implementation_transfer:
        training_experiment = json.loads(
            (args.checkpoint.parent / "experiment.json").read_text()
        )
        for name in physical_files:
            if code_hashes[name] != training_experiment["source_hashes"][name]:
                raise ValueError(f"Physical source changed since training: {name}")
        for name in adapter_files:
            original = args.checkpoint.parent / "sources" / name
            if code_hashes[name] != hashlib.sha256(original.read_bytes()).hexdigest():
                raise ValueError(
                    f"Native physics/sensor adapter changed since training: {name}"
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
            cfg,
            upstream,
            output,
            num_envs=args.num_envs,
            seed=777,
            minimum_free=args.min_free_gpu_gib,
        )
        result = _evaluate(task, actor, args.seeds, output)
        provenance = {
            "checkpoint": str(args.checkpoint.resolve()),
            "upstream_revision": revision,
            "upstream_checkout_clean": True,
            "evaluation_min_free_gpu_gib": args.min_free_gpu_gib,
            "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "evaluation_aggregation_sha256": hashlib.sha256(
                (Path(__file__).parent / "native_full.py").read_bytes()
            ).hexdigest(),
            "evaluation_code_sha256": code_hashes,
            "checkpoint_sha256": hashlib.sha256(
                args.checkpoint.read_bytes()
            ).hexdigest(),
            "checkpoint_trained_in_native_environment": native,
            "training_iteration": snapshot.get("iteration"),
            "native_training_success_qualified": native
            and not args.allow_implementation_transfer
            and result["summary"]["passed"],
            "native_eight_scale_randomized_variant_qualified": native
            and not args.allow_implementation_transfer
            and result["summary"]["passed"],
            "full_upstream_baseline_qualified": False,
            "controlled_original_budget_algorithm_comparison": False,
            "scope": "Native eight scales, source PD/friction-scale/mass/COM/gravity distributions and persistent forces; geometry-pair tactile, Newton/Euler. Hand self-contact dimensionality, metal friction/impedance, learner resource sizes and backend versions differ from upstream. Strict physics parity and original-budget algorithm comparison are not qualified.",
            "evaluation_seeds": args.seeds,
            "episodes": args.num_envs * len(args.seeds),
        }
        provenance["physical_implementation_matches_training_receipt"] = (
            native and not args.allow_implementation_transfer
        )
        provenance["native_policy_implementation_transfer_passed"] = (
            native
            and args.allow_implementation_transfer
            and result["summary"]["passed"]
        )
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
