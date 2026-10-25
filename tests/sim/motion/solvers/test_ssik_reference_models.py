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

"""Prepared ARX5 and Rizon4S references use their original model and frames."""

from __future__ import annotations

from dataclasses import replace
import importlib
import importlib.resources
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from embodichain.compute.kinematics._ssik_model import _extract_model
from embodichain.lab.sim.motion.solvers import SSIKSolver, SSIKSolverCfg
from embodichain_robots import ARX5Cfg, FlexivRizon4SCfg


@pytest.fixture(scope="module")
def pinned_reference() -> None:
    pytest.importorskip("ssik", reason="Optional pinned SSIK reference unavailable")
    from embodichain.lab.sim.motion.solvers._ssik_reference import _require_ssik

    try:
        _require_ssik()
    except ImportError:
        pytest.skip("The installed SSIK is not the pinned reference revision")


def _artifact(name: str) -> Path:
    return Path(
        str(
            importlib.resources.files("embodichain_robots").joinpath(
                "_ssik_artifacts", name + ".py"
            )
        )
    )


def _make_reference(
    name: str, artifact: Path | None = None, source: Path | None = None
) -> SSIKSolver:
    preset = (
        ARX5Cfg.from_dict({})
        if name == "arx5"
        else FlexivRizon4SCfg.from_dict({"fpath": str(source)} if source else {})
    )
    original = preset.solver_cfg["arm"]
    chain = preset.build_pk_serial_chain()["arm"]
    return SSIKSolverCfg(
        backend="reference",
        reference_artifact=str(artifact or _artifact(name)),
        urdf_path=preset.fpath,
        root_link_name=original.root_link_name,
        end_link_name=original.end_link_name,
        tcp=original.tcp,
    ).init_solver(pk_serial_chain=chain)


def _pk_fk(solver: SSIKSolver, joints: torch.Tensor) -> torch.Tensor:
    flange = solver.pk_serial_chain.forward_kinematics(joints).get_matrix()
    return flange @ torch.as_tensor(solver.tcp_xpos, dtype=flange.dtype)


def _assert_pose(target: torch.Tensor, actual: torch.Tensor) -> None:
    target_np = target.numpy().astype(np.float64)
    actual_np = actual.numpy().astype(np.float64)
    assert np.all(
        np.linalg.norm(actual_np[:, :3, 3] - target_np[:, :3, 3], axis=-1) <= 1e-5
    )
    rotation = Rotation.from_matrix(target_np[:, :3, :3]).inv() * Rotation.from_matrix(
        actual_np[:, :3, :3]
    )
    assert np.all(rotation.magnitude() <= 1e-5)


def _sample(solver: SSIKSolver, count: int = 4) -> torch.Tensor:
    fraction = 0.35 + 0.3 * torch.rand(
        count, solver.dof, generator=torch.Generator().manual_seed(2531)
    )
    return solver.lower_qpos_limits + fraction * (
        solver.upper_qpos_limits - solver.lower_qpos_limits
    )


@pytest.mark.parametrize("name", ["arx5", "flexiv_rizon4s"])
def test_packaged_reference_models_preserve_original_tcp_batch_and_candidates(
    pinned_reference: None, name: str
) -> None:
    solver = _make_reference(name)
    qpos = _sample(solver)
    tcp = np.eye(4)
    tcp[:3, :3] = Rotation.from_euler("xyz", [0.1, -0.2, 0.15]).as_matrix()
    tcp[:3, 3] = [0.03, -0.02, 0.08]
    solver.set_tcp(tcp)
    targets = _pk_fk(solver, qpos)
    seed = solver.get_default_qpos_seed()[None].expand(len(qpos), -1)
    valid, result = solver.get_ik_batch(
        targets.reshape(2, 2, 4, 4), seed.reshape(2, 2, solver.dof)
    )
    assert valid.shape == (2, 2) and bool(valid.all())
    assert result.shape == (2, 2, solver.dof)
    _assert_pose(targets, _pk_fk(solver, result.reshape(-1, solver.dof)))
    assert bool((result >= solver.lower_qpos_limits).all())
    assert bool((result <= solver.upper_qpos_limits).all())
    mask, candidates = solver.get_ik(targets[:2], seed[:2], return_all_solutions=True)
    assert mask.ndim == 2 and candidates.shape == (*mask.shape, solver.dof)
    assert bool(mask.any(-1).all())
    _assert_pose(
        targets[:2, None].expand(-1, mask.shape[1], -1, -1)[mask],
        _pk_fk(solver, candidates[mask]),
    )
    unreachable = targets[:1].clone()
    unreachable[:, :3, 3] = 100
    valid, result = solver.get_ik(unreachable, seed[:1])
    assert valid.tolist() == [False]
    torch.testing.assert_close(result, seed[:1], atol=0, rtol=0)


def test_rizon_prebuilt_cli_binds_source_and_native_sidecar_provenance(
    pinned_reference: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from embodichain.lab.sim.motion.solvers._ssik_reference import _main

    source = FlexivRizon4SCfg.from_dict({}).fpath
    output = tmp_path / "rizon4s.py"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reference",
            "--urdf",
            source,
            "--root",
            "base_link",
            "--end",
            "flange",
            "--output",
            str(output),
            "--prebuilt-model",
            "flexiv.rizon4_ik",
        ],
    )
    _main()
    metadata = json.loads(output.with_suffix(".json").read_text())
    assert metadata["prebuilt"]["module"] == "ssik.prebuilt.flexiv.rizon4_ik"
    assert len(metadata["prebuilt"]["source_sha256"]) == 64
    assert len(metadata["prebuilt"]["sidecars"]["rizon4_jl.npz"]) == 64
    assert metadata["prebuilt_prefix"]["max_axis_error"] <= 5e-7
    loaded = _make_reference("flexiv_rizon4s", output)
    qpos = _sample(loaded, 1)
    valid, result = loaded.get_ik(_pk_fk(loaded, qpos), qpos)
    assert bool(valid.all())
    _assert_pose(_pk_fk(loaded, qpos), _pk_fk(loaded, result))


@pytest.mark.parametrize("change", ["axis", "link", "joint_order"])
def test_prebuilt_reuse_rejects_a_changed_joint_prefix_before_emission(
    pinned_reference: None, tmp_path: Path, change: str
) -> None:
    from embodichain.lab.sim.motion.solvers._ssik_reference import (
        _emit_prebuilt_reference,
    )

    preset = FlexivRizon4SCfg.from_dict({})
    xml = ET.parse(preset.fpath)
    joint = xml.getroot().find("joint[@name='joint3']")
    if change == "axis":
        joint.find("axis").set("xyz", "0.001 0 1")
    elif change == "link":
        joint.find("origin").set("xyz", "0 0.036 0.205")
    else:
        joint.set("name", "different_joint_name")
    source = tmp_path / "changed.urdf"
    xml.write(source)
    changed = FlexivRizon4SCfg.from_dict({"fpath": str(source)})
    model = _extract_model(changed.build_pk_serial_chain()["arm"])
    output = tmp_path / "invalid.py"
    with pytest.raises(ValueError, match="joint|prefix"):
        _emit_prebuilt_reference(
            model, changed.solver_cfg["arm"], "flexiv.rizon4_ik", output
        )
    assert not output.exists()


def test_prebuilt_reuse_accepts_a_verified_rigid_terminal_frame_change(
    pinned_reference: None, tmp_path: Path
) -> None:
    from embodichain.lab.sim.motion.solvers._ssik_reference import (
        _emit_prebuilt_reference,
    )

    original = FlexivRizon4SCfg.from_dict({})
    xml = ET.parse(original.fpath)
    flange = next(
        j
        for j in xml.getroot().findall("joint")
        if j.find("child").get("link") == "flange"
    )
    flange.find("origin").set("xyz", "0.02 -0.01 0.17")
    flange.find("origin").set("rpy", "0.1 -0.2 0.3")
    source = tmp_path / "different_flange.urdf"
    xml.write(source)
    changed = FlexivRizon4SCfg.from_dict({"fpath": str(source)})
    model = _extract_model(changed.build_pk_serial_chain()["arm"])
    output = tmp_path / "flange_reference.py"
    _emit_prebuilt_reference(
        model, changed.solver_cfg["arm"], "flexiv.rizon4_ik", output
    )
    solver = _make_reference("flexiv_rizon4s", output, source)
    qpos = _sample(solver, 2)
    targets = _pk_fk(solver, qpos)
    valid, result = solver.get_ik(targets)
    assert bool(valid.all())
    _assert_pose(targets, _pk_fk(solver, result))


def test_prebuilt_reuse_rejects_nonrigid_terminal_rotation(
    pinned_reference: None, tmp_path: Path
) -> None:
    from embodichain.lab.sim.motion.solvers._ssik_reference import (
        _emit_prebuilt_reference,
    )

    preset = FlexivRizon4SCfg.from_dict({})
    model = _extract_model(preset.build_pk_serial_chain()["arm"])
    with pytest.raises(ValueError, match="proper rigid"):
        _emit_prebuilt_reference(
            replace(model, home_rotation=np.diag([1.0, 1.0, -1.0])),
            preset.solver_cfg["arm"],
            "flexiv.rizon4_ik",
            tmp_path / "reflection.py",
        )


def test_prebuilt_reuse_rejects_an_unverified_solver_family(
    pinned_reference: None, tmp_path: Path
) -> None:
    from embodichain.lab.sim.motion.solvers._ssik_reference import (
        _emit_prebuilt_reference,
    )

    preset = FlexivRizon4SCfg.from_dict({})
    model = _extract_model(preset.build_pk_serial_chain()["arm"])
    with pytest.raises(ValueError, match="verified cached joint-lock family"):
        _emit_prebuilt_reference(
            model,
            preset.solver_cfg["arm"],
            "kuka.iiwa14_ik",
            tmp_path / "wrong_family.py",
        )


@pytest.mark.parametrize("part", ["source_sha256", "sidecar"])
def test_prebuilt_provenance_is_checked_before_executing_the_adapter(
    pinned_reference: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, part: str
) -> None:
    original = _artifact("flexiv_rizon4s")
    output = tmp_path / original.name
    output.write_bytes(original.read_bytes())
    metadata = json.loads(original.with_suffix(".json").read_text())
    if part == "source_sha256":
        metadata["prebuilt"][part] = "0" * 64
    else:
        metadata["prebuilt"]["sidecars"]["rizon4_jl.npz"] = "0" * 64
    output.with_suffix(".json").write_text(json.dumps(metadata))
    real_import = importlib.import_module

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name == metadata["prebuilt"]["module"]:
            raise AssertionError("Mismatched prebuilt module was executed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", guarded_import)
    with pytest.raises(ValueError, match="provenance"):
        _make_reference("flexiv_rizon4s", output)


def test_prebuilt_sidecar_must_exist_during_build_and_load(
    pinned_reference: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from embodichain.lab.sim.motion.solvers._ssik_reference import (
        _emit_prebuilt_reference,
    )

    preset = FlexivRizon4SCfg.from_dict({})
    model = _extract_model(preset.build_pk_serial_chain()["arm"])
    is_file = Path.is_file

    def missing_sidecar(path: Path) -> bool:
        return False if path.name == "rizon4_jl.npz" else is_file(path)

    monkeypatch.setattr(Path, "is_file", missing_sidecar)
    with pytest.raises(ValueError, match="sidecar is missing"):
        _emit_prebuilt_reference(
            model, preset.solver_cfg["arm"], "flexiv.rizon4_ik", tmp_path / "missing.py"
        )
    with pytest.raises(ValueError, match="sidecar is missing"):
        _make_reference("flexiv_rizon4s")


def _forbid_runtime_derivation(monkeypatch: pytest.MonkeyPatch) -> None:
    from ssik.solvers.ikgeo import _raghavan_roth
    from ssik.solvers.husty_pfurner import _constraints, general_6r

    def unexpected_derivation(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "A prepared reference attempted runtime symbolic derivation"
        )

    for function in (
        "_derive_pq_for_arm",
        "_derive_pq_symbolic",
        "_cached_best_leftvar",
    ):
        monkeypatch.setattr(_raghavan_roth, function, unexpected_derivation)
    for function in dir(_constraints):
        if function.startswith("_build_") and callable(getattr(_constraints, function)):
            monkeypatch.setattr(_constraints, function, unexpected_derivation)
    monkeypatch.setattr(general_6r, "precompute_rrr_chain", unexpected_derivation)


@pytest.mark.parametrize("name", ["arx5", "flexiv_rizon4s"])
def test_prepared_models_never_derive_symbolic_systems_at_runtime(
    pinned_reference: None, name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_runtime_derivation(monkeypatch)
    solver = _make_reference(name)
    qpos = _sample(solver)
    target = _pk_fk(solver, qpos)
    seed_fraction = torch.rand(
        qpos.shape, generator=torch.Generator().manual_seed(9721)
    )
    seed = solver.lower_qpos_limits + seed_fraction * (
        solver.upper_qpos_limits - solver.lower_qpos_limits
    )
    valid, result = solver.get_ik(target, seed)
    assert bool(valid.all())
    _assert_pose(target, _pk_fk(solver, result))
    if name == "flexiv_rizon4s":
        from ssik.prebuilt.flexiv import rizon4_ik

        grid = torch.as_tensor(rizon4_ik._LOCK_SAMPLES, dtype=torch.float64)
        locked = result[:, rizon4_ik._LOCK_IDX].double()
        assert bool((torch.abs(locked[:, None] - grid).amin(-1) < 1e-6).all())
        # An out-of-workspace target empties every cached RR slice. It must
        # return failure rather than enter upstream's cold HP secondary pass.
        target[:, :3, 3] = 100
        valid, result = solver.get_ik(target, seed)
        assert not bool(valid.any())
        torch.testing.assert_close(result, seed, atol=0, rtol=0)


def test_direct_jointlock_artifact_uses_only_its_prepared_grid(
    pinned_reference: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from embodichain.lab.sim.motion.solvers._ssik_reference import (
        _manifest,
        _prebuilt_source,
    )

    # Use the upstream fixture's 81 mm end frame, so the explicitly supplied
    # direct artifact needs no end-frame adapter. Its AOT coefficients are
    # already embedded; this fixture never runs a symbolic build.
    original = FlexivRizon4SCfg.from_dict({})
    xml = ET.parse(original.fpath)
    flange = next(
        j
        for j in xml.getroot().findall("joint")
        if j.find("child").get("link") == "flange"
    )
    flange.find("origin").set("xyz", "0 0 0.081")
    source = tmp_path / "upstream_flange.urdf"
    xml.write(source)
    preset = FlexivRizon4SCfg.from_dict({"fpath": str(source)})
    model = _extract_model(preset.build_pk_serial_chain()["arm"])
    prebuilt, _, _ = _prebuilt_source("flexiv.rizon4_ik")
    artifact = tmp_path / "direct_ik.py"
    artifact.write_bytes(prebuilt.read_bytes())
    metadata = _manifest(model, preset.solver_cfg["arm"])
    metadata["source_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    artifact.with_suffix(".json").write_text(json.dumps(metadata))
    _forbid_runtime_derivation(monkeypatch)
    solver = _make_reference("flexiv_rizon4s", artifact, source)
    qpos = _sample(solver, 2)
    seed = qpos.flip(0)
    targets = _pk_fk(solver, qpos)
    valid, result = solver.get_ik(targets, seed)
    assert bool(valid.all())
    _assert_pose(targets, _pk_fk(solver, result))
    targets[:, :3, 3] = 100
    valid, result = solver.get_ik(targets, seed)
    assert not bool(valid.any())
    torch.testing.assert_close(result, seed, atol=0, rtol=0)
