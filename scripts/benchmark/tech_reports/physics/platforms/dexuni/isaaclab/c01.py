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
"""Audit native Isaac Lab/Newton support for C01's sealed isothermal cavity.

Run: /home/oem/app/IsaacLab/.venv/bin/python scripts/benchmark/tech_reports/physics/platforms/dexuni/isaaclab/m05.py
The installed native VBD has no cavity pressure term. Save the evidence and
report this limitation; a plain closed cloth mesh is not an equivalent C01.
No simulation, fabricated video, custom pressure solver, or timing is produced.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import re
import sys

import newton

__all__ = ["main"]


def main() -> None:
    """Record versioned API and source evidence for this experiment's limit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[4].parents[2]
        / "outputs/benchmarks/tech_reports/dexuni/c01_supermarket_packing/analysis/isaaclab_support.json",
    )
    parser.add_argument(
        "--isaaclab-source",
        type=Path,
        default=Path("/home/oem/app/IsaacLab/source/source"),
    )
    args = parser.parse_args()
    newton_root = Path(newton.__file__).resolve().parent
    schema = (
        args.isaaclab_source
        / "isaaclab_newton/isaaclab_newton/sim/schemas/schemas_cfg.py"
    )
    source_files = [
        newton_root / "_src/sim/builder.py",
        newton_root / "_src/sim/model.py",
        *sorted((newton_root / "_src/solvers/vbd").rglob("*.py")),
        *sorted(
            (args.isaaclab_source / "isaaclab_newton/isaaclab_newton/sim").rglob("*.py")
        ),
    ]
    pattern = re.compile(
        r"\b(pneumatic|isothermal|add_inflatable_mesh|add_pneumatic_cavity|reference_absolute_pressure)\b",
        re.I,
    )
    sources = {}
    matches = []
    for path in source_files:
        text = path.read_text()
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for line, value in enumerate(text.splitlines(), 1):
            if pattern.search(value):
                matches.append(dict(path=str(path), line=line, text=value.strip()))
    builder_methods = [
        name
        for name in dir(newton.ModelBuilder)
        if any(
            part in name.lower()
            for part in ("pneumatic", "pressure", "cavity", "inflatable")
        )
    ]
    signature = inspect.signature(newton.solvers.SolverVBD)
    pressure_parameters = [
        name for name in signature.parameters if pattern.search(name)
    ]
    tree = ast.parse(schema.read_text())
    deformable_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "NewtonDeformableBodyPropertiesCfg"
    )
    schema_fields = [
        node.target.id
        for node in deformable_class.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    ]
    if matches or builder_methods or pressure_parameters:
        raise RuntimeError(
            "Pressure API candidates found; review before classifying support"
        )
    native_manifest = args.out.parents[1] / "manifest.json"
    reference = json.loads(native_manifest.read_text())
    audit = dict(
        task="C01",
        platform="Isaac Lab / native Newton VBD",
        status="unsupported",
        equivalent_scene_supported=False,
        limitation="No native sealed-cavity isothermal pressure authoring or VBD energy/force term in the installed stack.",
        required_gas=reference["scene"]["pouch"]["gas"],
        required_rest_volume_m3=reference["scene"]["pouch"]["rest_volume_m3"],
        executable=sys.executable,
        isaaclab_version=importlib.metadata.version("isaaclab"),
        newton_version=newton.__version__,
        newton_module=str(Path(newton.__file__).resolve()),
        checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        builder_cloth_mesh_supported=hasattr(newton.ModelBuilder, "add_cloth_mesh"),
        builder_pressure_methods=builder_methods,
        vbd_pressure_parameters=pressure_parameters,
        vbd_signature=str(signature),
        deformable_schema=dict(
            path=str(schema), line=deformable_class.lineno, fields=schema_fields
        ),
        scan_pattern=pattern.pattern,
        source_matches=matches,
        source_sha256=sources,
        reference_manifest_sha256=hashlib.sha256(
            native_manifest.read_bytes()
        ).hexdigest(),
        scope="Installed Isaac Lab 28.0.0 + Newton 1.6.1; no claim about all versions or custom extensions.",
        video_created=False,
        performance_measurements_started=False,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(audit, indent=2) + "\n")
    report = args.out.with_suffix(".md")
    report.write_text(
        "# C01 Isaac Lab 支持检查\n\n"
        f"当前环境：Isaac Lab {audit['isaaclab_version']} / Newton {newton.__version__}。\n\n"
        "结论：当前原生 Newton VBD 无法等价实现本任务的封闭等温充气包。"
        "可以导入闭合布料表面、机器人和四面体面包，但闭合网格本身不会产生内部气压。\n\n"
        "C01 需要 p·V 恒定的气体压力项、环境压力、体积率阻尼以及与袋膜和四面体面包的接触响应；"
        "批准场景为 101800 Pa 参考绝对压力、101325 Pa 环境压力、bulk damping 10。"
        "当前 ModelBuilder 没有腔体/充气建模方法，SolverVBD 没有气压参数，"
        "对 builder/model、全部 VBD 求解文件和 Isaac Lab Newton sim 配置/导入源码扫描"
        f" {len(sources)} 份文件未找到 pneumatic/isothermal 等气体项。\n\n"
        f"NewtonDeformableBodyPropertiesCfg 位于 {schema}:{deformable_class.lineno}，"
        "当前为预留的空原生属性组。MPM 的 yield_pressure 和 hydroelastic 接触刚度"
        "分别属于材料屈服与接触，不能替代封闭气体腔体。\n\n"
        "新增自定义气压/VBD 项可以作为单独研发任务，但将成为扩展求解器，"
        "不能标成 Isaac Lab 自带的原生 VBD。本次如实记录 unsupported，未制作替代布袋视频，"
        "未测量性能。该结论只覆盖所检查的安装版本。\n\n"
        f"[API、版本与源码哈希证据]({args.out.name})。\n"
    )
    print(
        f"C01 Isaac Lab native Newton VBD: unsupported sealed isothermal cavity; "
        f"{len(sources)} source files checked. Evidence: {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
