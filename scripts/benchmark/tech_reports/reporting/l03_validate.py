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

"""Audit saved L-03 formal coverage, raw evidence and all-environment model/state data."""

from __future__ import annotations

from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdPhysics

__all__ = ["main"]


def main() -> None:
    """Check structural invariants and report continuous differences without accuracy gates."""
    base = Path("outputs/benchmarks/L-03/three-robots-v1")
    manifest = json.loads(
        Path("outputs/benchmarks/L-03/prepared/shared-usd-v1/manifest.json").read_text()
    )
    names = ("cartpole", "franka", "go2")
    inputs = [item for name in names for item in manifest["robots"][name]["files"]]
    bounded = {}
    for name in names:
        stage = Usd.Stage.Open(manifest["robots"][name]["usd_path"])
        bounded[name] = set()
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Joint) and not prim.IsA(UsdPhysics.FixedJoint):
                limits = [
                    prim.GetAttribute("physics:" + key).Get()
                    for key in ("lowerLimit", "upperLimit")
                ]
                if np.isfinite(limits).all():
                    bounded[name].add(prim.GetName())
    for item in inputs:
        assert (
            hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()
            == item["sha256"]
        )
    plan = json.loads((base / "plans/three_robots_main_v1.json").read_text())
    assert len(plan) == 126 and all(j["status"] == "completed" for j in plan)
    expected = set(
        itertools.product(
            names, (1, 100, 1000), ("C", "H"), ("embodichain", "isaaclab"), range(3)
        )
    )
    actual = {
        (j["robot"], j["num_envs"], j["cache_mode"], j["platform"], j["repeat"])
        for j in plan
        if j["purpose"] == "main"
    }
    assert actual == expected
    assert sum(j["purpose"] == "main" for j in plan) == 108
    assert all(j["performance_eligible"] for j in plan if j["purpose"] == "main")
    preloads = {
        (j["robot"], j["num_envs"], j["platform"]): j
        for j in plan
        if j["purpose"] == "preload"
    }
    assert len(preloads) == 18
    pairs, verified_runs = {}, []
    hardware, commits, source_hashes, adapter_hashes = set(), set(), {}, {}
    report_rows = {
        r["run_id"]: r for r in json.loads((base / "analysis/runs.json").read_text())
    }
    for run in sorted((base / "runs").iterdir()):
        assert (run / "exit.json").exists(), f"Unfinished run: {run}"
        cfg = json.loads((run / "config.json").read_text())
        finish = json.loads((run / "exit.json").read_text())
        meta = json.loads((run / "run.json").read_text())
        launch = json.loads((run / "launch.json").read_text())
        assert (
            meta["status"] == "completed"
            and finish["code"] == 0
            and not finish["timeout"]
        )
        assert meta["quaternion_order"] == "xyzw"
        for key, value in manifest["robots"][cfg["robot"]].items():
            assert cfg[key] == value, (run.name, key)
        for source, digest in launch["source_sha256"].items():
            assert (
                hashlib.sha256(
                    (run / "source" / Path(source).name).read_bytes()
                ).hexdigest()
                == digest
            )
            if Path(source).name == "l03_loading.py":
                adapter_hashes.setdefault(cfg["platform"], set()).add(digest)
        stages = list(meta["stage_times_s"].values())
        assert len(stages) == 4 and stages == sorted(stages) and stages[0] > 0
        expected_phases = dict(
            ready_s=stages[-1],
            engine_s=stages[0],
            declare_s=stages[1] - stages[0],
            prepare_s=stages[2] - stages[1],
            initialize_first_readback_s=stages[3] - stages[2],
        )
        assert all(
            report_rows[run.name][key] == value
            for key, value in expected_phases.items()
        )
        with np.load(run / "states.npz") as stream:
            states = {k: stream[k] for k in stream.files}
        with np.load(run / "model.npz") as stream:
            model = {k: stream[k] for k in stream.files}
        audit = json.loads((run / "model-audit.json").read_text())
        n = cfg["num_envs"]
        assert audit["instance_count"] == audit["num_envs"] == n
        assert (
            len(set(audit["arena_names"]))
            == len(np.unique(model["env_origins"], axis=0))
            == n
        )
        assert sorted(audit["joint_names"]) == cfg["expected_joint_names"]
        assert set(audit["link_names"]) == set(cfg["source_body_names"])
        assert model["mass"].shape == (n, len(cfg["source_body_names"]))
        assert states["first_qpos"].shape == (n, len(cfg["initial_qpos"]))
        assert states["first_link_pose"].shape == (n, len(audit["link_names"]), 7)
        assert all(np.isfinite(value).all() for value in states.values())
        target = np.asarray(
            [cfg["initial_qpos"][name] for name in audit["joint_names"]],
            dtype=states["initial_qpos"].dtype,
        )
        assert np.array_equal(
            states["initial_qpos"],
            np.broadcast_to(target, states["initial_qpos"].shape),
        )
        assert np.all(states["initial_qvel"] == 0) and np.all(
            states["initial_root_velocity"] == 0
        )
        if cfg["platform"] == "embodichain":
            assert (
                not audit["vbd_active"]
                and audit["articulation_solver"] == "SolverMuJoCo"
            )
            for field, count in (
                ("native_shape_world", cfg["colliders_per_env"]),
                ("native_body_world", len(audit["link_names"])),
            ):
                ids = model[field]
                assert np.array_equal(
                    np.bincount(ids[ids >= 0], minlength=n), np.full(n, count)
                )
        else:
            assert audit["native_shapes_per_articulation"] == cfg["colliders_per_env"]
            assert audit["effective_fixed_base"] == cfg["fixed_base"]
        if cfg["purpose"] in ("main", "preload"):
            assert not cfg["record"] and not meta["rendering"] and cfg["dt"] == 0.005
            assert launch["compute_before"] == finish["external_compute_pids"] == []
            memory = [
                json.loads(line)
                for line in (run / "memory.jsonl").read_text().splitlines()
            ]
            assert all(
                not (set(s["compute_pids"]) - set(s["tree_pids"])) for s in memory
            )
            window = [
                s
                for s in memory
                if meta["entry_epoch"] <= s["epoch"] <= meta["entry_epoch"] + stages[-1]
            ]
            assert len(window) > 1
            rss = (
                max(s["rss_bytes"] for s in window if s["rss_bytes"] is not None)
                / 2**20
            )
            vram = (
                max(
                    sum(
                        s["gpu_process_bytes"].get(str(pid), 0)
                        for pid in s["tree_pids"]
                    )
                    for s in window
                )
                / 2**20
            )
            assert vram > 0
            assert report_rows[run.name]["peak_rss_mib"] == rss
            assert report_rows[run.name]["peak_process_gpu_mib"] == vram
            hardware.add(json.dumps(launch["hardware"], sort_keys=True))
            commits.add(json.dumps(launch["commits"], sort_keys=True))
            source_hashes.setdefault(cfg["platform"], set()).add(
                json.dumps(launch["source_sha256"], sort_keys=True)
            )
            if cfg["purpose"] == "main":
                assert finish["performance_eligible"]
                if cfg["cache_mode"] == "H":
                    preload = preloads[(cfg["robot"], n, cfg["platform"])]
                    end = json.loads(
                        (base / "runs" / preload["run_id"] / "exit.json").read_text()
                    )["end_epoch"]
                    assert end < launch["start_epoch"]
        else:
            assert not finish["performance_eligible"]
        key = (cfg["purpose"], cfg["robot"], n, cfg["cache_mode"], cfg["repeat"])
        assert cfg["platform"] not in pairs.setdefault(key, {})
        pairs[key][cfg["platform"]] = (run, states, model, audit, cfg)
        verified_runs.append(run.name)
    assert len(hardware) == len(commits) == 1
    assert all(len(s) == 1 for s in source_hashes.values())
    assert all(
        len(s) == 1 for s in adapter_hashes.values()
    ), "Media, pilot and formal adapters differ"
    summary = json.loads((base / "analysis/summary.json").read_text())
    assert len(summary) == 36
    for group in summary:
        samples = [report_rows[run_id] for run_id in group["run_ids"]]
        assert len(samples) == group["repeats"] == 3
        assert all(s["performance_eligible"] for s in samples)
        for source, prefix in (
            ("ready_s", "ready"),
            ("peak_rss_mib", "ram"),
            ("peak_process_gpu_mib", "vram"),
        ):
            unit = "s" if prefix == "ready" else "mib"
            values = [s[source] for s in samples]
            assert group[f"{prefix}_median_{unit}"] == statistics.median(values)
            assert group[f"{prefix}_min_{unit}"] == min(values)
            assert group[f"{prefix}_max_{unit}"] == max(values)
        assert (
            report_rows[group["representative_run_id"]]["ready_s"]
            == group["ready_median_s"]
        )
    comparisons = []
    for (purpose, robot, n, cache, repeat), platforms in sorted(pairs.items()):
        a, sa, ma, aa, ca = platforms["isaaclab"]
        b, sb, mb, ab, cb = platforms["embodichain"]
        for key in (
            "files",
            "usd_path",
            "initial_qpos",
            "initial_position",
            "fixed_base",
            "env_spacing",
            "dt",
            "record",
        ):
            assert ca[key] == cb[key]
        links = [ab["link_names"].index(name) for name in aa["link_names"]]
        joints = [ab["joint_names"].index(name) for name in aa["joint_names"]]
        finite = [
            i for i, name in enumerate(aa["joint_names"]) if name in bounded[robot]
        ]
        rotation = (
            Rotation.from_quat(mb["com_pose"][:, links, 3:].reshape(-1, 4))
            .as_matrix()
            .reshape(n, len(links), 3, 3)
        )
        ib = (
            rotation
            @ (np.eye(3)[None, None] * mb["inertia_principal"][:, links, None, :])
            @ rotation.swapaxes(-1, -2)
        )
        ia = ma["inertia_matrix"].reshape(ib.shape)
        qa = Rotation.from_quat(sa["first_link_pose"][..., 3:].reshape(-1, 4))
        qb = Rotation.from_quat(sb["first_link_pose"][:, links, 3:].reshape(-1, 4))
        comparisons.append(
            dict(
                purpose=purpose,
                robot=robot,
                num_envs=n,
                cache_mode=cache,
                repeat=repeat,
                isaac_run=a.name,
                embodichain_run=b.name,
                max_link_position_difference_m=float(
                    np.linalg.norm(
                        sa["first_link_pose"][..., :3]
                        - sb["first_link_pose"][:, links, :3],
                        axis=-1,
                    ).max()
                ),
                max_link_rotation_difference_deg=float(
                    np.degrees((qa.inv() * qb).magnitude()).max()
                ),
                max_joint_position_difference=float(
                    np.abs(sa["first_qpos"] - sb["first_qpos"][:, joints]).max()
                ),
                max_initial_root_difference_m=float(
                    np.linalg.norm(
                        sa["initial_root_pose"][..., :3]
                        - sb["initial_root_pose"][..., :3],
                        axis=-1,
                    ).max()
                ),
                max_mass_difference_kg=float(
                    np.abs(ma["mass"] - mb["mass"][:, links]).max()
                ),
                max_inertia_component_difference_kgm2=float(np.abs(ia - ib).max()),
                max_com_component_difference_m=float(
                    np.abs(ma["com_pose"][..., :3] - mb["com_pose"][:, links, :3]).max()
                ),
                max_bounded_limit_difference=float(
                    np.abs(
                        ma["joint_limits"][:, finite]
                        - mb["joint_limits"][:, joints][:, finite]
                    ).max()
                ),
            )
        )
    assert Counter(r["purpose"] for r in comparisons) == {
        "main": 54,
        "preload": 9,
        "pilot": 9,
        "media": 3,
    }
    counts = {}
    for name, path in (
        ("current", base / "analysis/evidence.json"),
        (
            "pre_formal",
            base / "history/pre-formal-20261027_185909/analysis/evidence.json",
        ),
    ):
        evidence = json.loads(path.read_text())
        for item in evidence:
            assert (
                hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()
                == item["sha256"]
            )
        counts[name] = len(evidence)
    metrics = [key for key in comparisons[0] if key.startswith("max_")]
    formal = [r for r in comparisons if r["purpose"] == "main"]
    result = dict(
        verified_raw_files=counts["current"],
        preserved_pre_formal_raw_files=counts["pre_formal"],
        input_files_unchanged=len(inputs),
        verified_runs=len(verified_runs),
        formal_jobs=108,
        preloads=18,
        formal_pairs=54,
        all_formal_resource_isolated=True,
        one_hardware_and_commit_configuration=True,
        per_platform_execution_sources_identical=True,
        media_pilot_formal_platform_adapters_identical=True,
        formal_maxima={key: max(r[key] for r in formal) for key in metrics},
        comparisons=comparisons,
    )
    (base / "analysis/validation.json").write_text(json.dumps(result, indent=2) + "\n")
    (base / "analysis/validation-source.py").write_text(Path(__file__).read_text())
    print(json.dumps({k: v for k, v in result.items() if k != "comparisons"}, indent=2))


if __name__ == "__main__":
    main()
