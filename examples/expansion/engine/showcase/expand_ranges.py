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

"""Collect wider scene, motion and visual variants, then publish only after acceptance."""

from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import yaml
from ._cases import ENGINE, dump
from .__main__ import run_worker

__all__ = ["main"]


def main() -> None:
    """Extend an existing qualified showcase without replacing its frozen recipes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    R = args.output.resolve()
    import fcntl

    lock = (R / ".showcase.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def freeze(path, value):
        if path.exists():
            if yaml.safe_load(path.read_text()) != value:
                raise ValueError(f"Frozen expansion input changed: {path}")
            return
        dump(path, value)

    source = ENGINE.parent / "scene_robot_feasibility/search.ur5.yaml"
    cfg = yaml.safe_load(source.read_text())
    for v in cfg["request"]["variables"]:
        if v["kind"] == "scene":
            v["lower"] = [-0.05, -0.05, 0.0]
            v["upper"] = [0.05, 0.05, 0.0]
        else:
            v["lower"] = [-0.04, -0.03, -0.12]
            v["upper"] = [0.04, 0.03, 0.12]
    for k in ("deployment", "program"):
        reference = cfg["task_program"][k]
        resolved = (source.parent / reference).resolve()
        if (
            not resolved.exists()
            and "extensions/embodichain_tasks/configs/" in reference
        ):
            from embodichain.utils.config_paths import resolve_config_path

            resolved = resolve_config_path(
                "extensions/embodichain_tasks/configs/"
                + reference.split("extensions/embodichain_tasks/configs/", 1)[1]
            )
        cfg["task_program"][k] = str(resolved)
    search = R / "configs/search.wide.yaml"
    freeze(search, cfg)
    if run_worker(R, "atlas", job=search, timeout=750):
        raise RuntimeError("wide atlas search failed")
    original = json.loads((R / "cases.json").read_text())
    cases = original["cases"]
    added = []
    for key, base in [
        ("wide-scene", "ur5-scene"),
        ("wide-joint_residual", "ur5-joint_residual"),
        ("wide-via_points", "ur5-via_points"),
        ("wide-visual", "default-rerender-controls"),
    ]:
        c = copy.deepcopy(next(c for c in cases if c["id"] == base))
        job = yaml.safe_load((R / c["job"]).read_text())
        c.update(id=key, title="大范围 · " + c["title"], job=f"configs/job.{key}.yaml")
        job["output_dir"] = str(R / "jobs" / key)
        if key == "wide-scene":
            job["scene_atlas"] = str(R / "inputs/atlas-wide/atlas.json")
        if "motion_profile" in job:
            profile = yaml.safe_load(Path(job["motion_profile"]).read_text())
            profile["trajectory"]["spatial"]["joint_offset_scale"] = 0.008
            path = R / "configs" / f"motion.{key}.yaml"
            freeze(path, profile)
            job["motion_profile"] = str(path)
        if key == "wide-visual":
            profile = yaml.safe_load(Path(job["visual_profile_file"]).read_text())
            profile["profiles"]["camera"]["cameras"]["cam_overhead"].update(
                position_delta=[0.10, -0.04, 0.05], focal_scale=1.25
            )
            profile["profiles"]["material"]["materials"]["cube"].update(
                base_color=[0.05, 0.25, 0.95], roughness=0.12, metallic=0.35
            )
            profile["profiles"]["light"]["lights"]["main_light"].update(
                color=[1.0, 0.55, 0.25], intensity=4.0
            )
            path = R / "configs/visual.wide.yaml"
            freeze(path, profile)
            job["visual_profile_file"] = str(path)
        if not (R / c["job"]).exists():
            dump(R / c["job"], job)
        added.append(c)
    dump(
        R / "cases.json",
        {
            **original,
            "cases": cases
            + [c for c in added if c["id"] not in {old["id"] for old in cases}],
        },
    )
    for c in added:
        for op in ("collect", "replay"):
            result = run_worker(
                R,
                op,
                job=R / c["job"],
                resume=op == "collect" and (R / "jobs" / c["id"] / "job.json").exists(),
                timeout=1500,
            )
            if result:
                raise RuntimeError(f'{op} failed: {c["id"]}; previous preview retained')
        print(c["id"], "collected and replayed", flush=True)
    if run_worker(R, "audit", timeout=1200):
        raise RuntimeError("wide acceptance failed; previous preview retained")
    from examples.expansion.engine.showcase._report import build_page

    build_page(R)
    print("QUALIFIED_PREVIEW_UPDATED", flush=True)


if __name__ == "__main__":
    main()
