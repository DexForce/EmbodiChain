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

"""Build a portable, hash-indexed resource archive for the public demo catalog.

This publisher-side tool does not upload or execute a simulation. Only pinned
source files and verified frozen inputs enter the archive, with per-file terms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat
import zipfile

import numpy as np
import trimesh

from embodichain.data_pipeline.real2sim import load_prototype
from embodichain.data_pipeline.real2sim.droid import (
    load_droid_episode,
    verify_droid_unit_evidence,
)
from examples.expansion.trajectory.real2sim_carton.prepare import (
    ROOT,
    digest,
    write_json,
)

__all__: list[str] = []


def build(
    source: Path, prepared: Path, native: Path, inputs: Path, output: Path
) -> Path:
    """Build the selected episode, prototypes, evidence and proxy resources."""
    output.mkdir(parents=True, exist_ok=False)
    staging = output / "resource"
    staging.mkdir()
    source_lock = json.loads((ROOT / "source.lock.json").read_text())
    input_lock = json.loads((inputs / "manifest.json").read_text())
    scene = json.loads((prepared / "scene.lock.json").read_text())
    qualification = json.loads((prepared / "qualification.json").read_text())
    if not qualification["admitted"] or scene["source_lock_sha256"] != digest(
        ROOT / "source.lock.json"
    ):
        raise ValueError("resource scene must come from the qualified pinned source")
    if (
        load_droid_episode(source / "trajectory.h5").sha256
        != qualification["source_revision"]
    ):
        raise ValueError("resource recording differs from source qualification")
    native_report = json.loads((native / "report.json").read_text())
    if (
        not native_report["baseline_verified"]
        or not native_report["source_artifacts_complete"]
    ):
        raise ValueError("independent synthetic reference has no verified capture")
    records = []

    def add(
        path: Path,
        name: str,
        *,
        license_id: str,
        origin: str,
        expected: dict | None = None,
    ) -> None:
        if expected and (
            path.stat().st_size != expected["size"]
            or digest(path) != expected["sha256"]
        ):
            raise ValueError(f"pinned resource differs: {name}")
        target = staging / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        records.append(
            dict(
                path=name,
                size=target.stat().st_size,
                sha256=digest(target),
                license=license_id,
                origin=origin,
            )
        )

    for entry in source_lock["files"]:
        add(
            source / entry["path"],
            "raw/" + entry["path"],
            license_id="CC-BY-4.0",
            origin="unmodified_DROID_raw_1.0.1",
            expected=entry,
        )
    for entry in source_lock["annotations"]:
        add(
            ROOT / "annotations" / entry["path"],
            "annotations/" + entry["path"],
            license_id="CC-BY-4.0",
            origin="DROID_author_calibration_or_manual_derived_annotation",
            expected=entry,
        )
    for entry in input_lock["files"]:
        real = entry["path"].startswith("measured/")
        add(
            inputs / entry["path"],
            "inputs/" + entry["path"],
            license_id="CC-BY-4.0" if real else "Apache-2.0",
            origin=(
                "measured_DROID_derived" if real else "independent_synthetic_reference"
            ),
            expected=entry,
        )
    add(
        inputs / "manifest.json",
        "inputs/manifest.json",
        license_id="Apache-2.0",
        origin="file_index",
    )
    for i in range(3):
        prototype = load_prototype(inputs / f"measured/{i}/prototype.json")
        evidence = prepared / f"measured/evidence/{i}.npz"
        verify_droid_unit_evidence(
            evidence, source / "trajectory.h5", prototype.unit.source
        )
        add(
            evidence,
            f"evidence/measured/{i}.npz",
            license_id="CC-BY-4.0",
            origin="typed_original_state_and_command_evidence",
        )
    for entry in sorted((native / "prototypes").rglob("*")):
        if entry.is_file():
            add(
                entry,
                "synthetic_native/prototypes/"
                + entry.relative_to(native / "prototypes").as_posix(),
                license_id="Apache-2.0",
                origin="independently_generated_native_design",
            )
    for filename in ("report.json", "actual_rollout.npz"):
        add(
            native / filename,
            "synthetic_native/" + filename,
            license_id="Apache-2.0",
            origin="independent_native_execution",
        )
    for filename in ("source.lock.json", "assets.lock.json"):
        add(
            ROOT / filename,
            filename,
            license_id="Apache-2.0",
            origin="source_provenance_index",
        )
    add(
        prepared / "environment.yaml",
        "scene/environment.yaml",
        license_id="Apache-2.0",
        origin="example_physical_configuration",
    )
    # Retain numerical/reference evidence, but no machine-local cache paths.
    qualification["robot"].pop("urdf_path", None)
    qualification["resource_notice"] = (
        "Frozen reference evidence with local URDF cache path omitted; live execution requalifies its own model."
    )
    write_json(output / "qualification.json", qualification)
    add(
        output / "qualification.json",
        "evidence/qualification.json",
        license_id="CC-BY-4.0",
        origin="derived_geometry_and_kinematic_qualification",
    )
    physical = {key: value for key, value in scene.items() if key != "geometry"}
    physical.update(
        robot_geometry_sha256=scene["geometry"]["robot_geometry_sha256"],
        proxy_geometry=dict(
            carton="carton_proxy.glb",
            worktop="worktop.glb",
            purpose="Display proxies in metre-scale object-local frames; physics is defined by environment.yaml, not GLB import",
        ),
        geometry_scope="Estimated cuboid; no manufacturer CAD or real dynamics qualification",
    )
    write_json(output / "physical.json", physical)
    add(
        output / "physical.json",
        "scene/physical.json",
        license_id="CC-BY-4.0",
        origin="estimated_physical_proxy",
    )
    size = scene["object_size_m"]
    carton = trimesh.creation.box(size)
    carton.visual.face_colors = np.tile([235, 240, 236, 255], (len(carton.faces), 1))
    carton.visual.face_colors[carton.face_normals[:, 0] > 0.99] = [35, 105, 185, 255]
    carton.export(output / "carton_proxy.glb")
    board = trimesh.creation.box([0.65, 0.65, scene["support_height_m"]])
    board.visual.face_colors = [167, 180, 190, 255]
    board.export(output / "worktop.glb")
    for filename in ("carton_proxy.glb", "worktop.glb"):
        add(
            output / filename,
            "scene/" + filename,
            license_id="CC-BY-4.0",
            origin="procedural_estimated_display_proxy",
        )
    license_dir = ROOT.parents[3] / "LICENSE"
    add(
        license_dir,
        "licenses/Apache-2.0.txt",
        license_id="Apache-2.0",
        origin="license_text",
    )
    attribution = """# Attribution and terms

DROID raw files and measured derivatives: CC BY 4.0.
Credit: Alexander Khazatsky, Karl Pertsch, Suraj Nair and the DROID Dataset Team,
RAIL contributors; DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset.
Project: https://droid-dataset.github.io/
Paper / data-license statement: https://arxiv.org/html/2403.12945v2 (Introduction).
License: https://creativecommons.org/licenses/by/4.0/
Original source URLs and SHA-256 values are in source.lock.json.

Selected original HDF5, metadata and MP4 files are byte-for-byte unchanged.
Measured input derivatives were cropped, calibrated and resampled; labels,
estimated cuboid geometry and conditional mount calibration were added by
DexForce's EmbodiChain demo. They are not original independent measurements.
Additional selected camera annotations: Karl Pertsch / DROID Dataset Team,
https://huggingface.co/KarlP/droid at
bcb840c3b496533e0adf548a54b51f2f00057837.

Independent synthetic reference and authored package metadata: Apache-2.0,
Copyright 2021-2026 DexForce Technology Co., Ltd. Synthetic arrays are not copies
of the measured recording and carry synthetic_reference origin.

Per-file licenses are recorded in manifest.json. The hosting repository's
default Apache-2.0 tag does not replace DROID's CC BY 4.0 terms. No endorsement
by DROID, Franka, Robotiq or STAPLES is implied. Robot URDFs and meshes are
referenced through existing asset downloads and are not duplicated here.
"""
    (output / "ATTRIBUTION.md").write_text(attribution)
    add(
        output / "ATTRIBUTION.md",
        "ATTRIBUTION.md",
        license_id="Apache-2.0",
        origin="attribution_and_change_notice",
    )
    readme = """# Real2Sim carton resource v1

One selected original DROID episode and independent synthetic motion reference
for EmbodiChain's Panda + Robotiq 2F-85 directional carton demo.

- raw/: unchanged original HDF5, metadata and three single-view MP4s.
- annotations/: pinned camera calibration, manual stages/events and geometry basis.
- inputs/: measured and independently synthetic Pick / MoveHeldObject / Place prototypes.
- evidence/: original typed streams and frozen qualification report.
- synthetic_native/: independent clean design prototypes, actual T+1 rollout and capture report.
- scene/: one carton and one worktop, metre-scale local GLB display proxies and physical configuration.

The blue carton face defines object +X. Scene appearance is procedural; it is
not a scan or manufacturer model. The original arm/Robotiq assets are already
available through EmbodiChain's model registry. A live run must requalify its
robot geometry, use source-bound IK/refinement and preserve recorder checks.
These are source resources, not confirmed generated collection episodes.

See ATTRIBUTION.md and manifest.json for per-file licenses, origins and hashes.
Reproduce with the matching SDK example:
embodichain run-example expansion/trajectory/real2sim_carton/run.py -- --output /tmp/carton-run
"""
    (output / "README.md").write_text(readme)
    add(
        output / "README.md",
        "README.md",
        license_id="Apache-2.0",
        origin="resource_usage",
    )
    manifest = dict(
        schema="real2sim-carton-resource/v1",
        revision="v1",
        source_id=source_lock["source_id"],
        source_lock_sha256=digest(ROOT / "source.lock.json"),
        files=sorted(records, key=lambda row: row["path"]),
        collection_episodes_included=False,
        robot_models_duplicated=False,
        licenses={"measured": "CC-BY-4.0", "independent_synthetic": "Apache-2.0"},
    )
    write_json(staging / "manifest.json", manifest)
    archive = output / "real2sim_carton_v1.zip"
    with zipfile.ZipFile(
        archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as zipped:
        # Explicit directories precede their files for Open3D extraction.
        for path in sorted(
            staging.rglob("*"), key=lambda p: (not p.is_dir(), p.as_posix())
        ):
            info = zipfile.ZipInfo(
                path.relative_to(staging).as_posix() + ("/" if path.is_dir() else ""),
                (2026, 11, 1, 0, 0, 0),
            )
            info.create_system = 3
            info.external_attr = (
                (stat.S_IFDIR | 0o755) if path.is_dir() else (stat.S_IFREG | 0o644)
            ) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            zipped.writestr(info, b"" if path.is_dir() else path.read_bytes())
    identity = dict(
        schema="real2sim-carton-resource-archive/v1",
        archive=archive.name,
        revision="v1",
        sha256=digest(archive),
        md5=hashlib.md5(archive.read_bytes()).hexdigest(),
        size=archive.stat().st_size,
        manifest_sha256=digest(staging / "manifest.json"),
        file_count=len(records),
        source_id=source_lock["source_id"],
    )
    write_json(output / "archive.json", identity)
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "prepared", "native", "inputs", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(
        build(
            *(
                getattr(args, name).resolve()
                for name in ("source", "prepared", "native", "inputs", "output")
            )
        )
    )


if __name__ == "__main__":
    main()
