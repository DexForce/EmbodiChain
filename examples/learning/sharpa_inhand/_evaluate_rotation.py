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

"""Read-only physical rotation traces around the original 240-scene evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
from _aligned_apg import _load_apg
from sharpa_rl_unilab.training import evaluation as evaluator

__all__ = ["main"]

# Fix acceptance before observing candidate results. Every stable scene must
# survive, turn at least once, and continue net positive motion in its last 5 s.
_CRITERIA = {
    "window_seconds": 20.0,
    "drop_fraction_max": 0.05,
    "stable_scene_fraction_min": 0.90,
    "median_angle_min_rad": float(2 * np.pi),
    "last_five_seconds_angle_min_rad": 0.5,
    "mean_positive_speed_fraction_min": 0.80,
    "mean_last_five_positive_speed_fraction_min": 0.80,
}


def main() -> None:
    """Evaluate a final teacher with unchanged task terms and measured traces."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--manifest", type=Path)
    group.add_argument("--seeds", type=int, nargs=3)
    parser.add_argument("--upstream", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    batches: list[dict[str, torch.Tensor]] = []

    class TraceEnv(evaluator.SharpaTeacherEnv):
        def __init__(self, *values: Any, **kwargs: Any) -> None:
            super().__init__(*values, **kwargs)
            self._trace_alive = torch.ones(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self._trace_omega: list[torch.Tensor] = []
            self._trace_active: list[torch.Tensor] = []

        def step(self, actions: torch.Tensor) -> Any:
            before = self._trace_alive.clone()
            state = super().step(actions)
            source = self.observation_manager.get_term_cfg("actor", "frame").func
            omega = (source.object_angvel * source.task_state.rotation_axis).sum(-1)
            self._trace_omega.append(torch.where(before, omega, 0).detach().cpu())
            self._trace_active.append(before.cpu())
            self._trace_alive &= ~(state.terminated | state.truncated)
            return state

        def close(self) -> None:
            if self._trace_omega:
                batches.append(
                    {
                        "omega": torch.stack(self._trace_omega),
                        "active": torch.stack(self._trace_active),
                    }
                )
            # SharpaTeacherEnv forwards close through __getattr__, so there is
            # no concrete parent close method for super() to resolve.
            self.env.close()

    settings: dict[str, Any] = {}
    if args.manifest:
        settings["manifest"] = str(args.manifest.resolve())
    if args.seeds:
        settings.update(seeds=args.seeds, manifest=None)
    loader = evaluator.load_policy if args.upstream else _load_apg
    with (
        patch.object(evaluator, "load_policy", loader),
        patch.object(evaluator, "SharpaTeacherEnv", TraceEnv),
    ):
        result = evaluator.evaluate_checkpoint(
            args.checkpoint,
            output=output / "evaluation.json",
            device="cuda:0",
            evaluation=settings or None,
        )
    padded = []
    for batch in batches:
        steps, rows = batch["omega"].shape
        values = torch.zeros((400, rows))
        values[:steps] = batch["omega"]
        padded.append(values)
    omega = torch.cat(padded, dim=1)
    if omega.shape != (400, len(result["episodes"])):
        raise ValueError("Trace rows do not match original evaluator episodes")
    measured = omega.double().sum(0).numpy() * 0.05
    expected = np.array([row["signed_angle"] for row in result["episodes"]])
    np.testing.assert_allclose(measured, expected, rtol=1e-4, atol=3e-4)
    tail = omega[-100:].double().sum(0).numpy() * 0.05
    scenes = []
    for index, row in enumerate(result["episodes"]):
        stable = (
            not bool(row["dropped"])
            and row["signed_angle"] >= 2 * np.pi
            and tail[index] >= _CRITERIA["last_five_seconds_angle_min_rad"]
        )
        scenes.append(
            {
                **row,
                "last_five_seconds_angle": float(tail[index]),
                "positive_speed_fraction": float(
                    (omega[:, index] > 0.1).float().mean()
                ),
                "last_five_positive_speed_fraction": float(
                    (omega[-100:, index] > 0.1).float().mean()
                ),
                "stable_rotation": bool(stable),
            }
        )
    stable_fraction = float(np.mean([row["stable_rotation"] for row in scenes]))
    median_angle = float(np.median(expected))
    mean_positive = float((omega > 0.1).float().mean())
    mean_tail_positive = float((omega[-100:] > 0.1).float().mean())
    passed = (
        result["summary"]["dropped"] <= _CRITERIA["drop_fraction_max"]
        and stable_fraction >= _CRITERIA["stable_scene_fraction_min"]
        and median_angle >= _CRITERIA["median_angle_min_rad"]
        and mean_positive >= _CRITERIA["mean_positive_speed_fraction_min"]
        and mean_tail_positive
        >= _CRITERIA["mean_last_five_positive_speed_fraction_min"]
    )
    report = {
        "passed": bool(passed),
        "criteria": _CRITERIA,
        "stage": result["stage"],
        "algorithm": result["algorithm"],
        "training_seed": result["training_seed"],
        "manifest_sha256": result["manifest_sha256"],
        "original_summary": result["summary"],
        "stable_scene_fraction": stable_fraction,
        "median_angle": median_angle,
        "median_last_five_seconds_angle": float(np.median(tail)),
        "mean_positive_speed_fraction": mean_positive,
        "mean_last_five_positive_speed_fraction": mean_tail_positive,
        "scenes": scenes,
    }
    torch.save({"batches": batches, "omega": omega}, output / "physical_traces.pt")
    (output / "stability.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "scenes"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
