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

"""Compare the APG collection boundary with the untouched upstream task."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from _aligned_task import _AlignedTask
from sharpa_rl_unilab.assets import ensure_assets
from sharpa_rl_unilab.cli import compose_config
from sharpa_rl_unilab.tasks.sharpa_inhand.teacher_env import SharpaTeacherEnv
from sharpa_rl_unilab.training.supermimic_mdp import SharpaSuperMimicCapture

__all__ = ["main"]


def _snapshot(
    env: SharpaTeacherEnv, state: Any, capture: Any
) -> dict[str, torch.Tensor]:
    result = {f"obs/{key}": value for key, value in state.obs.items()}
    result.update({f"physical/{key}": value for key, value in capture().items()})
    result.update(
        {
            f"terminal/{key}": value
            for key, value in (state.final_observation or {}).items()
        }
    )
    result.update(
        reward=state.reward,
        terminated=state.terminated,
        truncated=state.truncated,
        target=env.action_manager.get_term("hand").target,
        # Read existing manager outputs; recomputing rewards could alter state.
        reward_terms=env.reward_manager._step_reward,
    )
    return {key: value.detach().cpu().clone() for key, value in result.items()}


def _shared_native(cfg: Any, actions: torch.Tensor, output: Path) -> None:
    """Compare adapter calls with native receipts on the same GPU trajectory."""
    task = _AlignedTask(cfg)
    env, capture = task.env, task.capture
    native_step, native_reset = env.step, env.reset
    receipt: dict[str, Any] = {}
    errors: dict[str, float] = {}
    traces = []

    def compare(
        expected: dict[str, torch.Tensor], actual: dict[str, torch.Tensor]
    ) -> None:
        if expected.keys() != actual.keys():
            raise AssertionError("Native/adapter snapshot groups differ")
        for key, value in expected.items():
            torch.testing.assert_close(actual[key], value, rtol=0, atol=0, msg=key)
            errors[key] = max(
                errors.get(key, 0.0),
                float((actual[key].float() - value.float()).abs().max()),
            )

    def step(action: torch.Tensor) -> Any:
        assert env.state is not None
        compare(receipt["before"], _snapshot(env, env.state, capture))
        torch.testing.assert_close(action, receipt["action"], rtol=0, atol=0)
        state = native_step(action)
        receipt["native"] = _snapshot(env, state, capture)
        return state

    def reset(*args: Any, **kwargs: Any) -> Any:
        result = native_reset(*args, **kwargs)
        assert env.state is not None
        receipt["reset_obs"] = {
            key: value.detach().cpu().clone() for key, value in env.state.obs.items()
        }
        return result

    try:
        torch.manual_seed(5241)
        task.reset(seed=5241)
        env.step, env.reset = step, reset
        dones = timeouts = 0
        for action in actions:
            assert env.state is not None
            receipt["before"] = _snapshot(env, env.state, capture)
            receipt["action"] = action.clone()
            state = task.step(action)
            actual = _snapshot(env, state, capture)
            compare(receipt["native"], actual)
            traces.append(actual)
            done = state.terminated | state.truncated
            dones += int(done.sum())
            timeouts += int(state.truncated.sum())
            observed = task.reset_done(state)
            if done.any():
                compare(
                    receipt["reset_obs"],
                    {
                        key: value.detach().cpu().clone()
                        for key, value in observed.items()
                    },
                )
                assert task.transition is not None
                torch.testing.assert_close(
                    task.transition["s1"].cpu(), actual["physical/s"], rtol=0, atol=0
                )
        report = {
            "backend": str(cfg.training.sim_backend),
            "mode": "shared_native_step",
            "steps": len(actions),
            "num_envs": 8,
            "seed": 5241,
            "max_absolute_errors": errors,
            "completed_episodes": dones,
            "timeouts": timeouts,
            "policy_updates": 0,
            "algorithm_comparison_allowed": False,
            "scope": "Exact native action, pre-step state, return values and selective reset receipts; no independent GPU trajectory determinism claim",
        }
        torch.save({"actions": actions.cpu(), "traces": traces}, output / "traces.pt")
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        env.close()


def main() -> None:
    """Persist matched traces and fail on any observable task difference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mujoco", "mjwarp"), default="mujoco")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shared-native-step", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ["SHARPA_RL_UNILAB_ASSET_CACHE"] = str(output / "asset_cache")
    ensure_assets()
    torch.set_num_threads(2)
    device = "cpu" if args.backend == "mujoco" else "cuda:0"
    cfg = compose_config(
        "supermimic",
        args.backend,
        ["training.num_envs=8", f"training.device={device}"],
    )
    steps = 401  # Covers a complete original 20-second episode plus its boundary.
    generator = torch.Generator().manual_seed(7241)
    actions = (torch.randn(steps, 8, 22, generator=generator) * 0.15).to(device)
    # Exercise raw clipping and target saturation through the original owner.
    actions[::41] *= 20
    if args.shared_native_step:
        _shared_native(cfg, actions, output)
        return
    all_traces = []
    max_reward_gap = 0.0
    done_counts = []
    truncation_counts = []
    for wrapped in (False, False, True):
        torch.manual_seed(5241)
        np.random.seed(5241)
        adapter = _AlignedTask(cfg) if wrapped else None
        env = adapter.env if adapter else SharpaTeacherEnv(cfg, 8, auto_reset=False)
        capture = adapter.capture if adapter else SharpaSuperMimicCapture(env)
        trace = []
        dones = 0
        truncations = 0
        try:
            if adapter:
                adapter.reset(seed=5241)
            else:
                env.reset(seed=5241)
                capture.refresh()
            assert env.state is not None
            trace.append(_snapshot(env, env.state, capture))
            for action in actions:
                state = adapter.step(action) if adapter else env.step(action)
                trace.append(_snapshot(env, state, capture))
                done = state.terminated | state.truncated
                dones += int(done.sum())
                truncations += int(state.truncated.sum())
                if adapter:
                    assert adapter.transition is not None
                    from sharpa_rl_unilab.algos.supermimic.state import unflatten_state

                    measured = adapter.transition
                    analytic = adapter.mdp.reward(
                        unflatten_state(measured["s1"]), measured["xi"]
                    )
                    max_reward_gap = max(
                        max_reward_gap, float((analytic - measured["r"]).abs().max())
                    )
                    adapter.reset_done(state)
                    # Captured terminal data must survive manager resets.
                    torch.testing.assert_close(
                        measured["s1"].cpu(), trace[-1]["physical/s"], rtol=0, atol=0
                    )
                elif done.any():
                    ids = done.nonzero(as_tuple=False).flatten()
                    env.reset(ids)
                    capture.refresh(ids)
                assert env.state is not None
                trace[-1].update(
                    {
                        f"reset_obs/{key}": value.detach().cpu().clone()
                        for key, value in env.state.obs.items()
                    }
                )
            all_traces.append(trace)
            done_counts.append(dones)
            truncation_counts.append(truncations)
        finally:
            env.close()
    torch.save({"actions": actions.cpu(), "traces": all_traces}, output / "traces.pt")
    comparisons = []
    for actual_trace in all_traces[1:]:
        errors: dict[str, float] = {}
        first_difference = None
        for index, (reference, actual) in enumerate(
            zip(all_traces[0], actual_trace, strict=True)
        ):
            for key, expected in reference.items():
                error = float((actual[key].float() - expected.float()).abs().max())
                errors[key] = max(errors.get(key, 0.0), error)
                if error and first_difference is None:
                    first_difference = {"step": index, "key": key, "max_error": error}
        comparisons.append({"max_errors": errors, "first_difference": first_difference})
    errors = comparisons[1]["max_errors"]
    if max_reward_gap > 1e-6:
        raise AssertionError(f"Imagined/measured reward gap: {max_reward_gap}")
    report = {
        "backend": args.backend,
        "steps": steps,
        "num_envs": 8,
        "seed": 5241,
        "max_absolute_errors": errors,
        "original_repeat": comparisons[0],
        "adapter": comparisons[1],
        "analytic_reward_max_absolute_error": max_reward_gap,
        "completed_episodes": done_counts,
        "timeouts": truncation_counts,
        "scales": all_traces[0][0]["physical/xi"][:, 49].tolist(),
        "policy_updates": 0,
        "algorithm_comparison_allowed": False,
        "scope": "Original task versus APG physical collection boundary; no learner parity claim",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if any(errors.values()):
        raise AssertionError(
            "Task trace differs; inspect report and original-repeat control"
        )


if __name__ == "__main__":
    main()
