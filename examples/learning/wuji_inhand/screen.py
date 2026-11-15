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
"""Screen FK-fitted Wuji grasps with actual native physical holding rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pytorch_kinematics as pk
import torch
from _asset import _prepared_sha256
from _physics import SCALES, _WujiPhysics

from embodichain.lab.sim import SimulationManager

__all__ = ["main"]


class _GraspSettler:
    """Prepare grasps with bounded fingertip force/height feedback, then freeze."""

    def __init__(self, physical: _WujiPhysics, assets: Path) -> None:
        self.physical = physical
        text = (assets / "compiled/hand.urdf").read_bytes()
        self.chains = [
            pk.build_serial_chain_from_urdf(
                text, f"right_finger{i}_link4", "right_palm_link"
            ).to(dtype=torch.float32, device=physical.device)
            for i in range(1, 6)
        ]
        for i, chain in enumerate(self.chains, 1):
            expected = [f"right_finger{i}_joint{j}" for j in range(1, 5)]
            if chain.get_joint_parameter_names() != expected:
                raise ValueError("Unexpected preparation-controller joint ordering")

    @torch.no_grad()
    def update(self) -> None:
        p = self.physical
        pos, _, q = p.state()
        tips = p.tip_positions()
        force = p.contact.measure()
        radial = torch.nn.functional.normalize(tips[..., :2] - pos[:, None, :2], dim=-1)
        error = torch.zeros_like(tips)
        error[..., :2] = -radial * (0.5 - force).clamp(-0.15, 0.5)[..., None] * 0.004
        error[..., 2] = (pos[:, None, 2] - tips[..., 2]).clamp(-0.0015, 0.0015)
        eye = torch.eye(3, device=p.device)[None] * 0.0001
        for i, chain in enumerate(self.chains):
            full = chain.jacobian(q[:, i * 4 : (i + 1) * 4])
            jacobian = p.root_rotation @ full[:, :3]
            angular = p.root_rotation @ full[:, 3:]
            link = p.robot.get_link_pose(f"right_finger{i + 1}_link4")
            offset = tips[:, i] - (link[:, :3] - p.arena_offsets)
            jacobian += torch.linalg.cross(
                angular.transpose(-1, -2), offset[:, None], dim=-1
            ).transpose(-1, -2)
            delta = jacobian.transpose(-1, -2) @ torch.linalg.solve(
                jacobian @ jacobian.transpose(-1, -2) + eye, error[:, i, :, None]
            )
            p.target[:, i * 4 : (i + 1) * 4] += delta[..., 0].clamp(-0.04, 0.04)
        p.target.clamp_(p.lower, p.upper)


def main() -> None:
    """Retain native measured poses after stable physical settling."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--adapt-steps", type=int, default=0)
    args = parser.parse_args()
    # Cache construction reads every batch this run lists; never mix older runs.
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a fresh screening output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    prepared = _prepared_sha256(args.assets)
    physical = _WujiPhysics(
        args.assets, args.output, num_envs=256, seed=101, gain=args.gain
    )
    sources = {
        name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in ["_physics.py", "_asset.py", "prepare.py", "screen.py"]
    }
    kept = [[] for _ in SCALES]
    records = []
    try:
        settler = _GraspSettler(physical, args.assets) if args.adapt_steps else None
        batches = (min(len(cache) for cache in physical.cache) + 31) // 32
        for batch in range(batches):
            choices = torch.arange(32, device=physical.device).repeat(8) + batch * 32
            poses = physical.reset(choices=choices)
            for step in range(args.adapt_steps):
                if step % 2 == 0:
                    settler.update()
                physical.step(physical.target)
            initial, _, _ = physical.state()
            alive = torch.ones(256, dtype=torch.bool, device=physical.device)
            saved = None
            good_contact = torch.zeros(256, device=physical.device)
            contact_samples = 0
            contact_trace = []
            pose_trace = []
            for step in range(args.steps):
                physical.step(physical.target)
                pos, quat, q = physical.state()
                alive &= (pos[:, 2] - initial[:, 2]).abs() < 0.025
                alive &= (
                    torch.linalg.vector_norm(pos[:, :2] - initial[:, :2], dim=-1) < 0.05
                )
                force, valid_grasp = physical.grasp_contact()
                contact_trace.append(force.cpu())
                pose_trace.append(torch.cat((q, pos, quat), -1).cpu())
                if step >= args.steps // 2:
                    good_contact += valid_grasp
                    contact_samples += 1
                if step == args.steps - 1:
                    saved = torch.cat((q, pos, quat, physical.target), -1).cpu().numpy()
            assert saved is not None
            accepted = alive & (good_contact / max(1, contact_samples) >= 0.8)
            flags = accepted.cpu().numpy()
            states = torch.cat((q, pos, quat), -1).cpu().numpy()
            torch.save(
                {
                    "initial": poses.cpu(),
                    "final": torch.from_numpy(states),
                    "stable": alive.cpu(),
                    "accepted": accepted.cpu(),
                    "valid_grasp_fraction": (
                        good_contact / max(1, contact_samples)
                    ).cpu(),
                    "fingertip_contact_forces": torch.stack(contact_trace),
                    "physical_pose_trace": torch.stack(pose_trace),
                    # Feedback may have moved the preload; this target was held.
                    "target": physical.target.detach().cpu().clone(),
                },
                args.output / f"batch_{batch}.pt",
            )
            for scale in range(8):
                selected = saved[scale * 32 : (scale + 1) * 32][
                    flags[scale * 32 : (scale + 1) * 32]
                ]
                kept[scale].extend(selected)
            record = {
                "batch": batch,
                "survived": int(alive.sum()),
                "accepted": int(accepted.sum()),
                "mean_valid_grasp_fraction": float(
                    (good_contact / max(1, contact_samples)).mean()
                ),
                "survived_by_scale": [
                    int(flags[k * 32 : (k + 1) * 32].sum()) for k in range(8)
                ],
            }
            records.append(record)
            print(json.dumps(record), flush=True)
        for scale, rows in zip(SCALES, kept, strict=True):
            np.save(
                args.output / f"candidates_{scale:g}.npy",
                np.asarray(rows, dtype=np.float32).reshape(-1, 47),
            )
        (args.output / "screening.json").write_text(
            json.dumps(
                {
                    "gain": args.gain,
                    "source_hashes": sources,
                    "prepared_assets_sha256": prepared,
                    "control_steps": args.steps,
                    "preparation_feedback_steps": args.adapt_steps,
                    "feedback_active_during_qualification_holding": False,
                    "seconds": args.steps * 0.05,
                    "batches": records,
                    "kept_by_scale": [len(x) for x in kept],
                    "selection": "Palm-up native holding; thumb and at least two opposing non-thumb fingertip forces >.02N during>=80% of latter half; retain final measured pose and controller preload. No learned-rotation success claim.",
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        physical.close()
        physical = None
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    main()
