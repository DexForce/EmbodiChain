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

"""Optional pinned reference and explicit offline algebraic artifact builder.

Run ``python -m embodichain.lab.sim.motion.solvers._ssik_reference --help``.
No import of this module is needed for Warp execution.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import torch

from embodichain.compute.kinematics._ssik_model import _SSIKModel, _extract_model

__all__: list[str] = []
_PIN = "4109147f37a7ca3e2516c83853aca583612783f3"
_PREBUILT_PREFIX_TOLERANCE = 5e-7


def _require_ssik() -> Any:
    try:
        import ssik
    except ImportError as error:
        raise ImportError(
            "Install requirements/ssik-reference.txt for the optional pinned SSIK CPU reference"
        ) from error
    commit = None
    try:
        direct = importlib.metadata.distribution("ssik").read_text("direct_url.json")
        if direct:
            commit = json.loads(direct).get("vcs_info", {}).get("commit_id")
    except importlib.metadata.PackageNotFoundError:
        pass
    if commit is None:
        for parent in Path(ssik.__file__).resolve().parents:
            if (parent / ".git").exists():
                commit = subprocess.check_output(
                    ["git", "-C", str(parent), "rev-parse", "HEAD"], text=True
                ).strip()
                break
    if commit != _PIN:
        raise ImportError(
            f"SSIK reference requires pinned revision {_PIN}; install requirements/ssik-reference.txt"
        )
    return ssik


def _policy() -> Any:
    from ssik.core.tolerances import TolerancePolicy

    # The actual PK input has float32 URDF origins. Only compensate for their
    # representation error; no geometry is snapped or polished here.
    return TolerancePolicy(axis_parallel=5e-7, axis_intersect=5e-7)


def _as_kinbody(model: _SSIKModel, cfg: Any) -> Any:
    from ssik._kinbody import build_poe_kinbody

    records = [
        (
            name,
            "revolute",
            model.axes[i],
            model.offsets[i],
            tuple(model.limits[i]) if np.isfinite(model.limits[i]).all() else None,
        )
        for i, name in enumerate(model.names)
    ]
    final = np.eye(4)
    u, _, vt = np.linalg.svd(model.home_rotation)
    final[:3, :3] = u @ vt
    final[:3, 3] = model.offsets[model.dof]
    return build_poe_kinbody(
        records,
        final,
        cfg.root_link_name or "ssik_root",
        cfg.end_link_name or "ssik_end",
    )


def _manifest(model: _SSIKModel, cfg: Any) -> dict[str, Any]:
    return dict(
        ssik_commit=_PIN,
        model_fingerprint=model.fingerprint,
        joint_names=list(model.names),
        root_link_name=cfg.root_link_name,
        end_link_name=cfg.end_link_name,
    )


def _prebuilt_sidecar_name(node: ast.expr, source: Path) -> str:
    """Decode generated adjacent-file expressions without evaluating code."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _prebuilt_sidecar_name(node.left, source) + _prebuilt_sidecar_name(
            node.right, source
        )
    if (
        isinstance(node, ast.Attribute)
        and node.attr == "stem"
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "_Path"
        and len(node.value.args) == 1
        and isinstance(node.value.args[0], ast.Name)
        and node.value.args[0].id == "__file__"
    ):
        return source.stem
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "removesuffix"
        and len(node.args) == 1
    ):
        return _prebuilt_sidecar_name(node.func.value, source).removesuffix(
            _prebuilt_sidecar_name(node.args[0], source)
        )
    raise ValueError("Unsupported generated prebuilt sidecar expression")


def _prebuilt_source(name: str) -> tuple[Path, ast.Module, dict[str, Any]]:
    """Read pinned prebuilt constants and provenance without executing it."""
    prefix = "ssik.prebuilt."
    module_name = name if name.startswith(prefix) else prefix + name
    relative = module_name.removeprefix(prefix)
    if not relative or not all(part.isidentifier() for part in relative.split(".")):
        raise ValueError("prebuilt-model must identify a module under ssik.prebuilt")
    ssik = _require_ssik()
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, AttributeError, ValueError) as error:
        raise ValueError(
            f"Cannot locate pinned SSIK prebuilt model {name!r}"
        ) from error
    if spec is None or spec.origin is None:
        raise ValueError(f"Cannot locate pinned SSIK prebuilt model {name!r}")
    path = Path(spec.origin).resolve()
    package = Path(ssik.__file__).resolve().parent / "prebuilt"
    if path.suffix != ".py" or not path.is_relative_to(package.resolve()):
        raise ValueError("prebuilt-model must be source shipped under ssik.prebuilt")
    source = path.read_bytes()
    syntax = ast.parse(source, filename=str(path))
    sidecars: dict[str, str] = {}
    for node in ast.walk(syntax):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "with_name"
            and node.args
        ):
            filename = _prebuilt_sidecar_name(node.args[0], path)
            if not filename.endswith(".npz"):
                continue
            if Path(filename).name != filename:
                raise ValueError("Prebuilt sidecars must be adjacent source files")
            sidecar = path.with_name(filename)
            if not sidecar.is_file():
                raise ValueError(
                    f"Required pinned prebuilt sidecar is missing: {filename}"
                )
            sidecars[filename] = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    provenance = dict(
        module=module_name,
        source_sha256=hashlib.sha256(source).hexdigest(),
        sidecars=sidecars,
    )
    return path, syntax, provenance


def _prebuilt_literal(node: ast.expr) -> Any:
    """Decode generated numeric constants without importing artifact code."""
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_prebuilt_literal(value) for value in node.elts]
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "np"
        and node.func.attr == "array"
        and node.args
    ):
        return _prebuilt_literal(node.args[0])
    return ast.literal_eval(node)


def _prebuilt_correction(
    model: _SSIKModel, syntax: ast.Module
) -> tuple[np.ndarray, dict[str, float]]:
    """Verify joint-prefix equivalence, allowing a different rigid end frame."""
    wanted = {
        "SOLVER_NAME",
        "_JOINT_NAMES",
        "_JOINT_AXES",
        "_JOINT_T_LEFTS",
        "_JOINT_T_RIGHTS",
    }
    constants: dict[str, Any] = {}
    for node in syntax.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    constants[target.id] = _prebuilt_literal(node.value)
    if constants.get("SOLVER_NAME") != "jointlock.seven_r":
        raise ValueError(
            "Prebuilt reuse requires the verified cached joint-lock family; use ordinary artifact emission for other solvers"
        )
    if constants.keys() != wanted or tuple(constants["_JOINT_NAMES"]) != model.names:
        raise ValueError(
            "Prebuilt model joint names/order do not match the source model"
        )
    axes = np.asarray(constants["_JOINT_AXES"], dtype=np.float64)
    left = np.asarray(constants["_JOINT_T_LEFTS"], dtype=np.float64)
    right = np.asarray(constants["_JOINT_T_RIGHTS"], dtype=np.float64)
    if (
        axes.shape != (model.dof, 3)
        or left.shape != (model.dof, 4, 4)
        or right.shape != left.shape
    ):
        raise ValueError("Prebuilt model has incompatible chain constants")
    if not all(
        np.isfinite(value).all()
        for value in (axes, left, right, model.home_rotation, model.offsets)
    ):
        raise ValueError("Prebuilt/source model constants must be finite")
    identity = np.eye(3)
    if (
        not np.allclose(left[:, :3, :3], identity, atol=1e-10, rtol=0)
        or not np.allclose(right[:-1, :3, :3], identity, atol=1e-10, rtol=0)
        or not np.allclose(left[:, 3], [0, 0, 0, 1], atol=1e-10, rtol=0)
        or not np.allclose(right[:, 3], [0, 0, 0, 1], atol=1e-10, rtol=0)
    ):
        raise ValueError("Prebuilt model must use normalized POE joint frames")
    offsets = np.zeros((model.dof + 1, 3))
    offsets[:-1] = left[:, :3, 3]
    offsets[1:-1] += right[:-1, :3, 3]
    offsets[-1] = right[-1, :3, 3]
    axis_error = float(np.linalg.norm(axes - model.axes[: model.dof], axis=-1).max())
    offset_error = float(
        np.linalg.norm(offsets[:-1] - model.offsets[: model.dof], axis=-1).max()
    )
    if max(axis_error, offset_error) > _PREBUILT_PREFIX_TOLERANCE:
        raise ValueError(
            "Prebuilt model joint-axis/link prefix does not match the source model"
        )
    for rotation in (model.home_rotation, right[-1, :3, :3]):
        if (
            np.linalg.det(rotation) <= 0
            or np.max(np.abs(rotation.T @ rotation - identity)) > 1e-5
        ):
            raise ValueError("Terminal frames must have proper rigid rotations")
    actual_home = np.eye(4)
    u, _, vt = np.linalg.svd(model.home_rotation)
    actual_home[:3, :3] = u @ vt
    actual_home[:3, 3] = model.offsets[: model.dof + 1].sum(axis=0)
    prebuilt_home = np.eye(4)
    prebuilt_home[:3, :3] = right[-1, :3, :3]
    prebuilt_home[:3, 3] = offsets.sum(axis=0)
    correction = np.linalg.inv(actual_home) @ prebuilt_home
    return correction, dict(
        max_axis_error=axis_error, max_link_offset_error=offset_error
    )


def _prepare_cached_jointlock(module: Any) -> tuple[tuple[float, Any, str], ...]:
    """Validate the baked grid before any IK, without running HP builders."""
    from ssik.kinematics.poe_to_dh import poe_to_dh
    from ssik.kinematics.reverse import reverse_kinematic_chain
    from ssik.solvers.ikgeo._raghavan_roth import primed_linearity_for_dh
    from ssik.solvers.jointlock.seven_r import _RR_ELIGIBLE_INNER_SOLVERS, _lock_joint

    if not all(
        hasattr(module, name)
        for name in ("_LOCK_SAMPLES", "_DISPATCH_CACHE", "_LOCK_IDX", "_KB")
    ):
        raise ValueError(
            "Direct joint-lock artifacts require a baked sample/dispatch plan"
        )
    if len(module._LOCK_SAMPLES) != len(module._DISPATCH_CACHE):
        raise ValueError("Prepared joint-lock grid and dispatch table disagree")
    closed_form = {
        "three_parallel",
        "spherical_two_parallel",
        "spherical_two_intersecting",
    }
    plan = []
    for angle, name in zip(module._LOCK_SAMPLES, module._DISPATCH_CACHE):
        subchain = _lock_joint(module._KB, module._LOCK_IDX, float(angle))
        bare = name.removeprefix("reversed:")
        if bare in _RR_ELIGIBLE_INNER_SOLVERS:
            chain = (
                reverse_kinematic_chain(subchain)
                if name.startswith("reversed:")
                else subchain
            )
            dh = poe_to_dh(chain)
            if primed_linearity_for_dh(dh.alpha, dh.a, dh.d) is None:
                raise ValueError(
                    "Prepared joint-lock slice lacks its baked RR coefficients"
                )
        elif bare not in closed_form:
            raise ValueError(
                f"Prepared joint-lock solver {bare!r} is not a verified cached path"
            )
        plan.append((float(angle), subchain, name))
    return tuple(plan)


def _solve_cached_jointlock(
    module: Any,
    plan: tuple[tuple[float, Any, str], ...],
    target: np.ndarray,
    **kwargs: Any,
) -> list[Any]:
    """Solve only prepared slices; an empty RR sweep never launches cold HP.

    Seeds reorder the fixed sample grid. They do not introduce new lock
    coordinates whose coefficient system would require runtime derivation.
    """
    from ssik.core.solution import Solution
    from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY
    from ssik.postprocess import finalize_solutions
    from ssik.refinement import dedup_by_wrap_close
    from ssik.solvers.jointlock.seven_r import _dispatch

    if any(
        kwargs.get(name, False)
        for name in ("allow_refinement", "allow_rescue", "native")
    ):
        raise ValueError(
            "Prepared joint-lock references disable rescue/refinement/native dispatch"
        )
    policy = kwargs.get("policy", DEFAULT_TOLERANCE_POLICY)
    seed = kwargs.get("q_seed")
    order = np.arange(len(plan))
    if seed is not None:
        angles = np.array([row[0] for row in plan])
        distance = np.abs(
            (angles - float(seed[module._LOCK_IDX]) + np.pi) % (2 * np.pi) - np.pi
        )
        order = np.argsort(distance, kind="stable")
    candidates = []
    for index in order:
        angle, subchain, name = plan[index]
        rows, least_squares = _dispatch(
            name,
            subchain,
            target,
            policy,
            allow_refinement=False,
            refinement_max_iters=0,
            cached_rr_only=True,
        )
        if least_squares:
            continue
        for row in rows:
            candidates.append(
                Solution(
                    q=np.insert(row.q, module._LOCK_IDX, angle),
                    fk_residual=row.fk_residual,
                    refinement_used=row.refinement_used,
                )
            )
    unique = dedup_by_wrap_close(candidates, policy.subproblem_dedup)
    return finalize_solutions(
        unique,
        module._KB,
        T_target=target,
        respect_limits=kwargs.get("respect_limits", False),
        q_seed=seed,
        seed_metric=kwargs.get("seed_metric", "wrap_linf"),
        seed_tolerance=kwargs.get("seed_tolerance"),
        max_solutions=kwargs.get("max_solutions"),
        enumerate_windings=kwargs.get("enumerate_windings", False),
    )


def _emit_prebuilt_reference(
    model: _SSIKModel, cfg: Any, name: str, output: Path
) -> None:
    """Emit an explicitly certified end-frame adapter for a pinned artifact."""
    _, syntax, provenance = _prebuilt_source(name)
    correction, errors = _prebuilt_correction(model, syntax)
    output.parent.mkdir(parents=True, exist_ok=True)
    source = f'''"""Prepared pinned SSIK reference for source model {model.fingerprint}.

The joint prefix matches {provenance["module"]} within
{_PREBUILT_PREFIX_TOLERANCE} m/rad representation tolerance. Only the end frame
is adapted. EmbodiChain verifies every result against the original PK TCP.
"""

from __future__ import annotations

import importlib
import numpy as np
from embodichain.lab.sim.motion.solvers._ssik_reference import (
    _prepare_cached_jointlock,
    _solve_cached_jointlock,
)

_PREBUILT = importlib.import_module({provenance["module"]!r})
_SOURCE_TO_PREBUILT = np.array({correction.tolist()!r}, dtype=np.float64)
SOLVER_NAME = _PREBUILT.SOLVER_NAME
_CACHED_GRID = _prepare_cached_jointlock(_PREBUILT)


def solve(target, **kwargs):
    """Return sampled reference branches without implicit rescue/refinement."""
    kwargs.setdefault("allow_refinement", False)
    kwargs.setdefault("allow_rescue", False)
    kwargs.setdefault("native", False)
    transformed = np.asarray(target, dtype=np.float64) @ _SOURCE_TO_PREBUILT
    return _solve_cached_jointlock(_PREBUILT, _CACHED_GRID, transformed, **kwargs)
'''
    output.write_text(source)
    metadata = _manifest(model, cfg)
    metadata.update(
        source_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        prebuilt=provenance,
        prebuilt_prefix=errors,
    )
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


class _Reference:
    def __init__(self, model: _SSIKModel, cfg: Any) -> None:
        ssik = _require_ssik()
        self.model, self.cfg = model, cfg
        self.policy = _policy()
        self.artifact = None
        self._prepared_jointlock = None
        if cfg.reference_artifact is not None:
            path = Path(cfg.reference_artifact).resolve()
            metadata = json.loads(path.with_suffix(".json").read_text())
            expected = _manifest(model, cfg)
            if any(metadata.get(k) != v for k, v in expected.items()):
                raise ValueError(
                    "SSIK reference artifact does not match the model, frames, joints or pinned revision"
                )
            if hashlib.sha256(path.read_bytes()).hexdigest() != metadata.get(
                "source_sha256"
            ):
                raise ValueError(
                    "SSIK reference artifact source fingerprint does not match its manifest"
                )
            if "prebuilt" in metadata:
                recorded = metadata["prebuilt"]
                _, _, installed = _prebuilt_source(recorded.get("module", ""))
                if recorded != installed:
                    raise ValueError(
                        "Pinned prebuilt source/sidecar provenance does not match the manifest"
                    )
            spec = importlib.util.spec_from_file_location(
                "_embodichain_ssik_artifact", path
            )
            if spec is None or spec.loader is None:
                raise ValueError("Cannot load SSIK reference artifact")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.artifact = module
            self.name = module.SOLVER_NAME
            if self.name == "jointlock.seven_r" and "prebuilt" not in metadata:
                self._prepared_jointlock = _prepare_cached_jointlock(module)
        else:
            from ssik.core.dispatcher import dispatch

            kinbody = _as_kinbody(model, cfg)
            plan = dispatch(kinbody, policy=self.policy)
            if plan.solver_name in ("ikgeo.general_6r", "jointlock.seven_r"):
                raise ValueError(
                    "This reference geometry requires an explicitly built reference_artifact; runtime symbolic derivation is disabled"
                )
            self.manipulator = ssik.Manipulator(kinbody, policy=self.policy)
            self.name = self.manipulator.solver_name

    def solve(
        self,
        solver: Any,
        target: torch.Tensor,
        seed: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        weights: torch.Tensor,
        all_solutions: bool,
        max_step: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        targets = target.detach().numpy() @ solver._tcp_inverse
        # Float32 pose matrices are not exactly orthogonal. Project only this
        # roundoff before the reference's machine-precision internal FK gate;
        # original-model FK against the unmodified input remains authoritative.
        finite_rows = np.isfinite(targets).all(axis=(1, 2))
        for index in np.flatnonzero(finite_rows):
            r = targets[index, :3, :3]
            finite_rows[index] = (
                np.max(np.abs(r.T @ r - np.eye(3))) <= 1e-5
                and np.linalg.det(r) > 0
                and np.max(np.abs(targets[index, 3] - [0, 0, 0, 1])) <= 1e-6
            )
        if finite_rows.any():
            u, _, vt = np.linalg.svd(targets[finite_rows, :3, :3])
            targets[finite_rows, :3, :3] = u @ vt
        finite_seed = torch.isfinite(seed).all(dim=-1)
        seeds = torch.where(torch.isfinite(seed), seed, (lower + upper) / 2).clamp(
            min=lower, max=upper
        )
        solutions = []
        for index, (t, q) in enumerate(zip(targets, seeds.detach().numpy())):
            if (
                not finite_rows[index]
                or not finite_seed[index]
                or not np.isfinite(t).all()
                or not np.isfinite(q).all()
            ):
                solutions.append([])
                continue
            kwargs = dict(
                q_seed=q,
                respect_limits=False,
                allow_refinement=False,
                allow_rescue=False,
                enumerate_windings=False,
                policy=self.policy,
            )
            if self.name == "seven_r.srs":
                kwargs["swivel_samples"] = self.cfg.num_samples
            if self.artifact is not None:
                kwargs["native"] = False
            entry = (
                self.artifact.solve
                if self.artifact is not None
                else self.manipulator.solve
            )
            if self._prepared_jointlock is not None:
                solutions.append(
                    _solve_cached_jointlock(
                        self.artifact, self._prepared_jointlock, t, **kwargs
                    )
                )
            else:
                solutions.append(entry(t, **kwargs))
        count = max(1, max((len(s) for s in solutions), default=0))
        n = len(target)
        qpos = seeds.float()[:, None].expand(n, count, self.model.dof).clone()
        valid = torch.zeros((n, count), dtype=torch.bool)
        for i, rows in enumerate(solutions):
            for k, row in enumerate(rows):
                q = torch.tensor(row.q, dtype=torch.float64)
                first = torch.ceil((lower - 1e-6 - q) / (2 * np.pi))
                last = torch.floor((upper + 1e-6 - q) / (2 * np.pi))
                turns = torch.round((seeds[i] - q) / (2 * np.pi)).clamp(
                    min=first, max=last
                )
                candidate = (q + 2 * np.pi * turns).clamp(min=lower, max=upper).float()
                ok = (first <= last).all() and torch.isfinite(candidate).all()
                ok = ok and (candidate >= lower).all() and (candidate <= upper).all()
                ok = ok and ((candidate - seeds[i]).abs() <= max_step).all()
                if ok:
                    qpos[i, k] = candidate
                    valid[i, k] = True
        if n:
            actual = (
                solver.get_fk(qpos.reshape(-1, self.model.dof))
                .reshape(n, count, 4, 4)
                .double()
            )
            poserr = torch.linalg.vector_norm(
                actual[..., :3, 3] - target[:, None, :3, 3], dim=-1
            )
            relative = actual[..., :3, :3].transpose(-1, -2) @ target[:, None, :3, :3]
            skew = torch.stack(
                (
                    relative[..., 2, 1] - relative[..., 1, 2],
                    relative[..., 0, 2] - relative[..., 2, 0],
                    relative[..., 1, 0] - relative[..., 0, 1],
                ),
                -1,
            )
            angle = torch.atan2(
                torch.linalg.vector_norm(skew, dim=-1),
                relative.diagonal(dim1=-1, dim2=-2).sum(-1) - 1,
            )
            valid &= (poserr <= self.cfg.position_tolerance) & (
                angle <= self.cfg.rotation_tolerance
            )
        qpos = torch.where(valid[..., None], qpos, seeds.float()[:, None])
        if all_solutions:
            return valid, qpos
        distance = (
            (((qpos.double() - seeds[:, None]) * weights) ** 2)
            .sum(-1)
            .masked_fill(~valid, torch.inf)
        )
        choice = distance.argmin(dim=1)
        return valid.any(-1), qpos[torch.arange(n), choice]


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Explicitly derive a pinned SSIK reference artifact and its model manifest"
    )
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prebuilt-model",
        help="Explicit cached joint-lock ssik.prebuilt module whose verified joint prefix can be reused",
    )
    args = parser.parse_args()
    if args.output.suffix != ".py":
        parser.error(
            "--output must name a .py artifact (the adjacent .json holds its manifest)"
        )
    _require_ssik()
    from ssik.core.dispatcher import dispatch
    from ssik.core.codegen import emit_artifact
    from .ssik_solver import SSIKSolverCfg
    from embodichain.lab.sim.utility.solver_utils import create_pk_serial_chain

    cfg = SSIKSolverCfg(
        backend="reference",
        urdf_path=args.urdf,
        root_link_name=args.root,
        end_link_name=args.end,
    )
    chain = create_pk_serial_chain(
        urdf_path=args.urdf,
        root_link_name=args.root,
        end_link_name=args.end,
        device=torch.device("cpu"),
    )
    model = _extract_model(chain)
    if args.prebuilt_model:
        _emit_prebuilt_reference(model, cfg, args.prebuilt_model, args.output)
        print(args.output)
        return
    kb = _as_kinbody(model, cfg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    emit_artifact(
        kb=kb,
        plan=dispatch(kb, policy=_policy()),
        module_name=args.output.stem,
        output_path=str(args.output),
    )
    metadata = _manifest(model, cfg)
    # The pinned general-6R emitter forces Cartesian near-miss refinement.
    # Keep it opt-in without changing the separate accepted-root polishing.
    source = args.output.read_bytes()
    forced_gate = b"if not (allow_refinement or True):"
    if forced_gate in source:
        args.output.write_bytes(
            source.replace(forced_gate, b"if not allow_refinement:")
        )
        metadata["codegen_patches"] = ["respect_allow_refinement"]
        metadata["unpatched_source_sha256"] = hashlib.sha256(source).hexdigest()
    metadata["source_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    _main()
