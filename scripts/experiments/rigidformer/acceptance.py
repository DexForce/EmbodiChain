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

"""Pair the pinned Box2D reference recipe with the shared SL learner.

The upstream examples are loaded from an explicitly supplied, revision-checked
checkout. They retain their own MIT license and are not copied into production.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate

from embodichain.learning.models.rigidformer import (
    RIGIDFORMER_REVISION,
    RigidFormerCfg,
    RigidFormerModel,
)
from embodichain.learning.sl import SupervisedLearner
from embodichain.learning.sl.rigidformer import RigidFormerObjective, rigidformer_batch

__all__: list[str] = []

_NUMERICAL_ATOL = 1e-5
_NUMERICAL_RTOL = 1e-4


def _verify_sources(path: Path, installed: Path) -> dict[str, Any]:
    """Check source bytes against Git objects, including a dirty checkout."""
    revision = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != RIGIDFORMER_REVISION:
        raise ValueError(f"Expected reference revision {RIGIDFORMER_REVISION}")
    package_files = subprocess.check_output(
        [
            "git",
            "-C",
            str(path),
            "ls-tree",
            "-r",
            "--name-only",
            revision,
            "rigidformer",
        ],
        text=True,
    ).splitlines()
    sources = ["examples/box2d_dataset.py", "examples/train_box2d.py"]
    sources += [name for name in package_files if name.endswith(".py")]
    verified = {}
    for name in sources:
        pinned = subprocess.check_output(
            ["git", "-C", str(path), "show", f"{revision}:{name}"]
        )
        if (path / name).read_bytes() != pinned:
            raise ValueError(f"Reference source differs from pinned Git blob: {name}")
        if name.startswith("rigidformer/"):
            destination = installed / Path(name).relative_to("rigidformer")
            if destination.read_bytes() != pinned:
                raise ValueError(
                    f"Installed source differs from pinned Git blob: {name}"
                )
        verified[name] = {
            "git_blob": subprocess.check_output(
                ["git", "-C", str(path), "rev-parse", f"{revision}:{name}"], text=True
            ).strip(),
            "sha256": hashlib.sha256(pinned).hexdigest(),
        }
    return verified


def _load_reference(path: Path) -> tuple[Any, dict[str, Any]]:
    # Locate the package without importing it before its source is verified.
    package = importlib.util.find_spec("rigidformer")
    if package is None or package.origin is None:
        raise ImportError("The optional rigidformer dependency is unavailable")
    verified = _verify_sources(path, Path(package.origin).parent)
    source = path / "examples" / "box2d_dataset.py"
    spec = importlib.util.spec_from_file_location("rigidformer_box2d_reference", source)
    if spec is None or spec.loader is None:
        raise ValueError("Reference Box2D data module is unavailable")
    module = importlib.util.module_from_spec(spec)
    # Execute the checked bytes directly rather than accepting a cached bytecode file.
    exec(compile(source.read_bytes(), str(source), "exec"), module.__dict__)
    return module, verified


def _digest(values: Any) -> str:
    digest = hashlib.sha256()
    for value in values:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().contiguous().numpy()
        digest.update(np.asarray(value).tobytes())
    return digest.hexdigest()


def _write(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def _sample(dataset: Any, size: int, device: str) -> dict[str, torch.Tensor]:
    raw = default_collate(
        [dataset[np.random.randint(len(dataset))] for _ in range(size)]
    )
    return {key: value.to(device) for key, value in raw.items()}


def _rollout(
    model: Any, trajectory: Any, *, collision: bool, device: str
) -> dict[str, Any]:
    from rigidformer import RigidformerRolloutWrapper

    positions, props = trajectory
    first = 10 if collision else 30
    frames = torch.from_numpy(positions[first : first + 10 * 13 : 10])[:13].to(device)
    gt = frames.unsqueeze(1)
    network = model.network if isinstance(model, RigidFormerModel) else model
    with torch.no_grad():
        predicted = torch.stack(
            RigidformerRolloutWrapper(network)(
                delta_times=torch.full((1,), 1 / 6, device=device),
                vertex_properties=torch.from_numpy(props).unsqueeze(0).to(device),
                object_positions=[gt[:1, 0], gt[1:2, 0]],
                num_steps=11,
            )
        )
    constant_velocity = [gt[0], gt[1]]
    for _ in range(11):
        constant_velocity.append(2 * constant_velocity[-1] - constant_velocity[-2])
    constant = torch.stack(constant_velocity)
    stay = gt[1].expand_as(gt)
    centers = lambda value: value.mean(-2)
    point_error = (predicted - gt).norm(dim=-1).mean(dim=(1, 2, 3))
    center_error = (centers(predicted) - centers(gt)).norm(dim=-1).mean(dim=(1, 2))
    constant_error = (centers(constant) - centers(gt)).norm(dim=-1).mean(dim=(1, 2))
    stay_error = (centers(stay) - centers(gt)).norm(dim=-1).mean(dim=(1, 2))
    rigidity = (
        (
            torch.cdist(
                predicted[-1],
                predicted[-1],
                compute_mode="donot_use_mm_for_euclid_dist",
            )
            - torch.cdist(gt[0], gt[0], compute_mode="donot_use_mm_for_euclid_dist")
        )
        .abs()
        .max()
    )
    return {
        "center_error_m": center_error.cpu().tolist(),
        "point_error_m": point_error.cpu().tolist(),
        "constant_velocity_center_error_m": constant_error.cpu().tolist(),
        "stay_center_error_m": stay_error.cpu().tolist(),
        "rigidity_max_distance_error_m": float(rigidity),
    }


def _step_parity(dataset: Any, device: str) -> dict[str, Any]:
    candidate = RigidFormerModel().to(device)
    reference = deepcopy(candidate.network)
    optimizer = torch.optim.AdamW(reference.parameters(), lr=3e-4, weight_decay=1e-2)
    candidate_optimizer = torch.optim.AdamW(
        candidate.parameters(), lr=3e-4, weight_decay=1e-2
    )
    batch = _sample(dataset, 8, device)
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.startswith("cuda") else None
    loss, _ = reference(**batch)
    loss.backward()
    expected_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in reference.named_parameters()
        if parameter.grad is not None
    }
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
    optimizer.step()
    torch.set_rng_state(rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state(cuda_rng, device)
    learner = SupervisedLearner(
        candidate, RigidFormerObjective(), candidate_optimizer, max_grad_norm=1.0
    )
    # Capture unmodified gradients before the common clipping operation.
    observed_gradients: dict[str, torch.Tensor] = {}
    handles = [
        parameter.register_hook(
            lambda gradient, name=name: observed_gradients.__setitem__(
                name, gradient.detach().clone()
            )
        )
        for name, parameter in candidate.network.named_parameters()
    ]
    metrics = learner.update(rigidformer_batch(batch))
    for handle in handles:
        handle.remove()
    gradient_match = all(
        torch.allclose(observed_gradients[name], value, atol=1e-5, rtol=1e-4)
        for name, value in expected_gradients.items()
    )
    weight_match = all(
        torch.allclose(
            candidate.network.state_dict()[name], value, atol=1e-5, rtol=1e-4
        )
        for name, value in reference.state_dict().items()
    )
    return {
        "passed": bool(
            gradient_match
            and weight_match
            and np.isclose(metrics["loss"], float(loss.detach()), atol=1e-5, rtol=1e-4)
        ),
        "reference_loss": float(loss),
        "sl_loss": metrics["loss"],
        "gradients_match": gradient_match,
        "updated_parameters_match": weight_match,
        "atol": 1e-5,
        "rtol": 1e-4,
    }


def _fit(
    train: Any,
    validation: Any,
    *,
    seed: int,
    collision: bool,
    steps: int,
    device: str,
    use_sl: bool,
    output: Path,
) -> dict[str, Any]:
    from rigidformer import Rigidformer

    np.random.seed(seed)
    torch.manual_seed(seed)
    cfg = RigidFormerCfg()
    network_cfg = cfg.to_dict()
    network_cfg.pop("max_pairwise_bytes")
    model = (
        RigidFormerModel(cfg).to(device)
        if use_sl
        else Rigidformer(**network_cfg).to(device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    learner = (
        SupervisedLearner(model, RigidFormerObjective(), optimizer, max_grad_norm=1.0)
        if use_sl
        else None
    )
    losses, curve = [], []
    _sync(device)
    started = time.monotonic()
    for step in range(steps):
        batch = _sample(train, 8, device)
        if learner is not None:
            metrics = learner.update(rigidformer_batch(batch))
            scalar_loss = metrics["loss"]
        else:
            loss, _ = model(**batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scalar_loss = float(loss.detach())
        losses.append(scalar_loss)
        if not np.isfinite(scalar_loss):
            raise FloatingPointError("Reference recipe returned a non-finite loss")
        if (step + 1) % 500 == 0:
            model.eval()
            measured = _rollout(
                model, validation.trajectories[0], collision=collision, device=device
            )
            curve.append({"step": step + 1, **measured})
            model.train()
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "sl": use_sl,
                        "collision": collision,
                        "step": step + 1,
                        "loss": scalar_loss,
                    }
                ),
                flush=True,
            )
    _sync(device)
    training_seconds = time.monotonic() - started
    model.eval()
    single_step: dict[str, list[float]] = {
        key: [] for key in ("model", "constant_velocity", "stay")
    }
    # Preserve the reference 32 sampled single-step windows and RNG consumption.
    with torch.no_grad():
        for _ in range(32):
            batch = _sample(validation, 1, device)
            target = batch.pop("object_pos_next")
            prediction = model(**batch).object_pos_next
            for key, value in (
                ("model", prediction),
                (
                    "constant_velocity",
                    2 * batch["object_pos"] - batch["object_pos_prev"],
                ),
                ("stay", batch["object_pos"]),
            ):
                single_step[key].append(float((value - target).norm(dim=-1).mean()))
    reports = [
        _rollout(model, row, collision=collision, device=device)
        for row in validation.trajectories
    ]
    network = model.network if isinstance(model, RigidFormerModel) else model
    checkpoint = output / ("sl.pt" if use_sl else "reference.pt")
    torch.save(
        {
            "model": network.state_dict(),
            "optimizer": optimizer.state_dict(),
            "seed": seed,
            "steps": steps,
            "config": network_cfg,
        },
        checkpoint,
    )
    report = {
        "seed": seed,
        "mode": "collision" if collision else "friction",
        "engine": "sl" if use_sl else "reference",
        "steps": steps,
        "training_seconds": training_seconds,
        "losses": losses,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "single_step_mean_point_distance_m": {
            key: float(np.mean(value)) for key, value in single_step.items()
        },
        "reference_trajectory": reports[0],
        "validation_trajectories": reports,
        "mean_rollout_endpoint_center_error_m": float(
            np.mean([row["center_error_m"][-1] for row in reports])
        ),
        "mean_rollout_endpoint_point_error_m": float(
            np.mean([row["point_error_m"][-1] for row in reports])
        ),
        "curve": curve,
    }
    _write(output / ("sl.json" if use_sl else "reference.json"), report)
    return report


def _gate(reference: list[float], candidate: list[float]) -> dict[str, Any]:
    baseline, actual = np.asarray(reference), np.asarray(candidate)
    absolute = bool((baseline <= 1e-5).any())
    differences = actual - baseline if absolute else actual / baseline - 1
    draws = np.random.default_rng(0).integers(
        0, len(differences), size=(10000, len(differences))
    )
    upper = float(np.percentile(differences[draws].mean(-1), 95))
    threshold = 1e-5 if absolute else 0.05
    return {
        "status": (
            "pending"
            if len(reference) < 5
            else "passed" if np.isfinite(upper) and upper <= threshold else "failed"
        ),
        "one_sided_95_upper": upper,
        "threshold": threshold,
        "difference_mode": "absolute" if absolute else "relative",
        "reference": reference,
        "candidate": candidate,
    }


def _loss_curve_gate(
    reference: list[float], candidate: list[float], *, steps: int
) -> dict[str, Any]:
    """Require every paired optimization loss to match at the stated tolerance."""
    baseline = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(candidate, dtype=np.float64)
    complete = baseline.shape == actual.shape == (steps,)
    finite = bool(np.isfinite(baseline).all() and np.isfinite(actual).all())
    matching = (
        np.isclose(actual, baseline, atol=_NUMERICAL_ATOL, rtol=_NUMERICAL_RTOL)
        if complete and finite
        else np.zeros(steps, dtype=bool)
    )
    return {
        "passed": bool(complete and finite and matching.all()),
        "expected_steps": steps,
        "reference_steps": len(reference),
        "candidate_steps": len(candidate),
        "finite": finite,
        "mismatched_steps": int((~matching).sum()),
        "first_mismatched_step": (
            int(np.flatnonzero(~matching)[0]) + 1 if (~matching).any() else None
        ),
        "max_absolute_difference": (
            float(np.abs(actual - baseline).max()) if complete and finite else None
        ),
        "atol": _NUMERICAL_ATOL,
        "rtol": _NUMERICAL_RTOL,
    }


def _run(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.threads)
    reference, verified = _load_reference(args.reference)
    versions = {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "numpy",
            "rigidformer",
            "roma",
            "einx",
            "torch-einops-utils",
            "x-mlps-pytorch",
            "box2d-py",
        )
    }
    manifest = {
        "reference_revision": RIGIDFORMER_REVISION,
        "reference_dataset_sha256": verified["examples/box2d_dataset.py"]["sha256"],
        "verified_sources": verified,
        "device": args.device,
        "threads": args.threads,
        "versions": versions,
        "steps": args.steps,
        "seeds": args.seeds,
        "modes": args.modes,
        "batch_size": 8,
        "dt": 1 / 6,
        "objects": 3,
        "points_per_object": 48,
        "candidate_is_upstream_model": True,
    }
    _write(args.output / "manifest.json", manifest)
    summary: dict[str, Any] = {"passed": False, "modes": {}}
    for mode in args.modes:
        collision = mode == "collision"
        train = reference.Box2DDataset(
            [reference.trajectory(seed, collision) for seed in range(64)], collision
        )
        validation = reference.Box2DDataset(
            [reference.trajectory(seed, collision) for seed in range(1000, 1016)],
            collision,
        )
        corpus_hash = _digest(
            [
                value
                for row in (*train.trajectories, *validation.trajectories)
                for value in row
            ]
        )
        paired: list[dict[str, Any]] = []
        for seed in args.seeds:
            output = args.output / mode / str(seed)
            output.mkdir(parents=True)
            np.random.seed(seed)
            torch.manual_seed(seed)
            parity = _step_parity(train, args.device)
            _write(output / "step-parity.json", parity)
            baseline = _fit(
                train,
                validation,
                seed=seed,
                collision=collision,
                steps=args.steps,
                device=args.device,
                use_sl=False,
                output=output,
            )
            candidate = _fit(
                train,
                validation,
                seed=seed,
                collision=collision,
                steps=args.steps,
                device=args.device,
                use_sl=True,
                output=output,
            )
            paired.append(
                {
                    "reference": baseline,
                    "candidate": candidate,
                    "parity": parity,
                    "loss_curve": _loss_curve_gate(
                        baseline["losses"], candidate["losses"], steps=args.steps
                    ),
                }
            )
            _write(
                args.output / "progress.json",
                {
                    "mode": mode,
                    "completed_seeds": [row["reference"]["seed"] for row in paired],
                },
            )
        gates = {
            key: _gate(
                [row["reference"][key] for row in paired],
                [row["candidate"][key] for row in paired],
            )
            for key in (
                "mean_rollout_endpoint_center_error_m",
                "mean_rollout_endpoint_point_error_m",
            )
        }
        summary["modes"][mode] = {
            "corpus_sha256": corpus_hash,
            "gates": gates,
            "step_parity_passed": all(row["parity"]["passed"] for row in paired),
            "loss_curves": {
                str(row["reference"]["seed"]): row["loss_curve"] for row in paired
            },
        }
        _write(args.output / "summary.json", summary)
    summary["passed"] = args.steps >= 1000 and all(
        row["step_parity_passed"]
        and all(gate["passed"] for gate in row["loss_curves"].values())
        and all(gate["status"] == "passed" for gate in row["gates"].values())
        for row in summary["modes"].values()
    )
    _write(args.output / "summary.json", summary)
    if not summary["passed"]:
        raise SystemExit("Acceptance incomplete or failed; inspect summary.json")


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 47)))
    parser.add_argument(
        "--modes",
        choices=("friction", "collision"),
        nargs="+",
        default=["friction", "collision"],
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if (
        args.steps < 1
        or args.threads < 1
        or len(set(args.seeds)) != len(args.seeds)
        or len(set(args.modes)) != len(args.modes)
    ):
        parser.error("Use positive steps/threads and distinct seeds/modes")
    # Admission stays outside the failure guard: never write into an existing run.
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        _run(args)
    except BaseException as exc:
        _write(
            args.output / "failure.json",
            {"type": type(exc).__name__, "message": str(exc)},
        )
        raise


if __name__ == "__main__":
    _main()
