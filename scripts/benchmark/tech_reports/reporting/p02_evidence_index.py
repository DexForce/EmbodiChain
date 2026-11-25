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

"""Audit every P-02 attempt and write a reproducible evidence index."""

from __future__ import annotations

import os
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stack-limit", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    runs = []
    files = []
    main = set()
    errors = []
    directories = sorted((root / "runs").iterdir())
    if args.stack_limit:
        directories += sorted((root / "repeatability/runs").glob("*"))
    for directory in directories:
        if not directory.is_dir():
            continue
        p = directory / "run.json"
        if not p.exists():
            runs.append(
                dict(
                    run_id=directory.name,
                    relative_directory=str(directory.relative_to(root)),
                    status="startup_failed_before_run_json",
                    completed_episodes=0,
                )
            )
            continue
        m = json.loads(p.read_text())
        episodes = sorted(directory.glob("episode_??.npz"))
        row = {
            k: m.get(k)
            for k in [
                "run_id",
                "status",
                "phase",
                "platform",
                "solver",
                "dt_s",
                "duration_s",
                "completed_episodes",
                "initial_count",
                "failure",
            ]
        }
        row["scene"] = m["scene"]["id"]
        row["relative_directory"] = str(directory.relative_to(root))
        row["actual_episode_files"] = len(episodes)
        runs.append(row)
        if m["status"] == "completed":
            if args.stack_limit:
                assert m["duration_s"] == 10
                if m["platform"] == "embodichain":
                    vbd = m["requested_sim_config"]["physics_cfg"]["solver_cfg"][
                        "vbd_options"
                    ]
                    assert vbd["rigid_enable_sleep"] is True
                    assert vbd["rigid_contact_stick_motion_eps"] == 1e-4
                else:
                    sleeping = m["sleep_configuration"]
                    assert sleeping["requested_threshold"] is None
                    assert all(
                        v > 0 for v in sleeping["body_threshold_readback"].values()
                    )
            assert (
                len(episodes) == m["initial_count"] == m["completed_episodes"]
            ), directory
            for ep in episodes:
                with np.load(ep) as d:
                    state = d["state"]
                    t = d["t"]
                    i = int(ep.stem.split("_")[-1])
                    size = d["body_size"]
                    pose = d["requested_initial_poses"]
                    assert np.isfinite(state).all()
                    assert len(state) == round(m["duration_s"] / m["dt_s"]) + 1
                    np.testing.assert_allclose(t[-1], m["duration_s"], atol=1e-12)
                    np.testing.assert_allclose(state[0, :, :3], pose[:, :3], atol=1e-6)
                    assert np.abs(state[0, :, 7:]).max() < 1e-7
                    memmap = np.load(
                        ep.with_name(ep.stem + "_state.npy"), mmap_mode="r"
                    )
                    np.testing.assert_array_equal(memmap, state)
                    if m["phase"] == "independent" or (
                        m["phase"] == "main"
                        and m["scene"]["id"] not in ("stack_05", "stack_10")
                    ):
                        key = (m["platform"], m["scene"]["id"], i)
                        assert key not in main, key
                        main.add(key)
                    if "contacts" in d:
                        assert d["contacts"].shape[1] == 11
                        assert (
                            d["contact_counts"].max()
                            < m["contact_export"]["capacity_per_sensor"]
                        )
                        assert d["contact_counts"].sum() == len(d["contacts"])
        for source, digest in m["source_sha256"].items():
            snap = directory / "config" / Path(source).name
            if hashlib.sha256(snap.read_bytes()).hexdigest() != digest:
                errors.append(str(snap))
        capture_sources = directory / "config" / "capture-sources.json"
        if capture_sources.exists():
            for source, digest in json.loads(capture_sources.read_text()).items():
                snap = directory / "config" / Path(source).name
                if hashlib.sha256(snap.read_bytes()).hexdigest() != digest:
                    errors.append(str(snap))
    assert not errors, errors
    expected = 0 if args.stack_limit else 160
    assert len(main) == expected, f"Incomplete primary matrix: {len(main)}/{expected}"
    raw_paths = [p for directory in directories for p in directory.rglob("*")]
    for p in sorted(raw_paths):
        if p.is_file():
            files.append(
                dict(
                    path=str(p.relative_to(root)),
                    bytes=p.stat().st_size,
                    sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                )
            )
    project = Path(
        os.environ.get("SIM_REPORT_ROOT", Path(__file__).resolve().parents[4])
    ).resolve()
    source_files = sorted(
        (project / "scripts/benchmark/tech_reports").rglob("p02_*.py")
    )
    source_files += [
        project / "scripts/__init__.py",
        project / "tests/benchmark/tech_reports/reporting/test_p02_analysis.py",
    ]
    source_snapshots = []
    for source in source_files:
        target = root / "analysis/source" / source.relative_to(project)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        source_snapshots.append(
            dict(
                original_path=str(source),
                snapshot=str(target.relative_to(root)),
                sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
            )
        )
    result = dict(
        main_trajectories=len(main),
        main_expected=expected,
        source_snapshots_valid=True,
        final_source_snapshots=source_snapshots,
        analysis_environment=dict(
            python=platform.python_version(),
            packages={
                name: importlib.metadata.version(name)
                for name in ["numpy", "scipy", "matplotlib", "pillow"]
            },
        ),
        runs=runs,
        files=files,
    )
    (root / "analysis" / "evidence-manifest.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    lines = [
        "# P-02 数据出处与复现",
        "",
        "原始文件保留在相邻的 `../runs/`；堆叠重复性补测另存 `../repeatability/runs/`。每次运行都有独立编号；失败记录不覆盖。完整逐文件 SHA-256、大小和覆盖核验见 [evidence-manifest.json](../analysis/evidence-manifest.json)。",
        "",
        "| run_id | 阶段 | 平台／求解器 | 场景 | dt / ms | 完成初态数 | 状态 |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for r in runs:
        target = (
            "run.json"
            if (root / r["relative_directory"] / "run.json").exists()
            else "console.log"
        )
        lines.append(
            f"| [{r['run_id']}](../{r['relative_directory']}/{target}) | {r.get('phase','启动')} | {r.get('solver',r.get('platform','未初始化'))} | {r.get('scene','—')} | {r.get('dt_s',0)*1000:g} | {r.get('completed_episodes',0)} | {r['status']} |"
        )
    run_lines = lines.copy()
    lines += [
        "",
        "## 原始数组",
        "",
        "`episode_XX.npz`：`t` 单位秒；`state[T,B,13]` 为 xyz、xyzw、线速度、角速度，单位 m、m/s、rad/s。`requested_initial_poses`、箱体尺寸／惯量、支撑位姿和斜面坐标基一并保存。`episode_XX_state.npy` 是逐步刷新到磁盘的同一状态数组，已与 NPZ 逐项核对。底层 float32，存储为 float64 不增加仿真精度。",
        "",
        "PhysX 另有 `normal_forces`、`friction_forces`、`contact_counts` 和 `contacts`；首个 t=0 力未采样，保留 NaN。`contacts` 列名见 `run.json`，只保存有效行；冲量以 N·s、法向／切向力以 N 计。DexUni 公开接口未提供对应力，未填零补齐。",
        "",
        "`config/` 冻结入口源码、共享协议代码、参数、初态和运行时 Spec；`run.json` 保存解释器、完整命令、源提交、安装包版本、导入路径、实际属性与完成状态。视频另有 `native_camera/capture.json`、原始 PNG 及录制代码快照。",
        "",
        "本次最终采样／分析／绘图／报告脚本及数值测试另存于 `../analysis/source/`；清单中的 `final_source_snapshots` 记录路径与哈希，`analysis_environment` 记录离线依赖版本。运行时快照和最终分析快照分别保存，不覆盖历史版本。",
        "",
        "## 离线重建",
        "",
        "在 `/home/oem/code/simulation-report` 下运行（不初始化仿真器）：",
        "",
        "```bash",
        "/home/oem/miniconda3/envs/embodichain/bin/python -m scripts.benchmark.tech_reports.reporting.p02_analysis --runs outputs/benchmarks/P-02/runs --output outputs/benchmarks/P-02/analysis/main --phases main supplement independent independent_supplement",
        "/home/oem/miniconda3/envs/embodichain/bin/python -m scripts.benchmark.tech_reports.reporting.p02_figures --analysis outputs/benchmarks/P-02/analysis/main --output outputs/benchmarks/P-02/figures",
        "/home/oem/miniconda3/envs/embodichain/bin/python -m scripts.benchmark.tech_reports.reporting.p02_analysis --runs outputs/benchmarks/P-02/runs --output outputs/benchmarks/P-02/analysis/capture --phases capture --min-duration 2",
        "/home/oem/miniconda3/envs/embodichain/bin/python -m scripts.benchmark.tech_reports.reporting.p02_media_review --root outputs/benchmarks/P-02",
        "/home/oem/miniconda3/envs/embodichain/bin/python -m scripts.benchmark.tech_reports.reporting.p02_report --root outputs/benchmarks/P-02",
        "/home/oem/miniconda3/envs/embodichain/bin/python -m scripts.benchmark.tech_reports.reporting.p02_benchmark_summary --root outputs/benchmarks/P-02",
        "```",
        "",
        "重跑物理使用各运行 `run.json` 的 `command`，只把 `--output` 换成新目录。两平台进程串行运行。初态、碰撞尺寸和材料不能随展示修改。录制采用 `p02_batch --phase capture --count 1`，逐帧物理时刻与相机路径保持一致。",
        "",
        "录像轨迹另在 `analysis/capture/` 计算指标；`--min-duration 2` 排除两条仅 0.1 s 的相机接线试录，它们没有完整的稳定／观察窗口，原始文件仍保留。正式录制的状态数组均覆盖完整 10／12 s，视频截取时段另见视频索引。",
        "",
        "## 运行问题记录",
        "",
        "预检保留了本地包命名遮蔽、颜色 RGB/RGBA 和 Isaac 启动环境问题；多层 Isaac 场景保留了 SciPy 角度数组维度、ContactSensor 通配符匹配数量问题。均为采样脚本／环境适配，未修改引擎求解器。修复后以新 run_id 重跑。Kit 退出过程可能返回 0，即使 `run.json` 为 failed；本索引按实际状态和完整数组核对。较早 ledger 的退出码不能单独用于统计。",
        "",
        "早期 `pilot_physx_slope15_02` 虽完成了状态采样，静态接触筛选尚未指向实际碰撞 shape，力数据不作为校准或对比依据。修正后的试跑和正式运行另存，试跑不计入主测。",
    ]
    if args.stack_limit:
        lines = run_lines + [
            "",
            "## 堆叠扩展的数组与复现",
            "",
            "本索引是保留睡眠、10 s 稳定堆叠边界扩展；原 300 条 P-02 主测／参数研究和旧索引保留。[扩展报告](stack-limit.md)给出全部扫描点；[逐初态 CSV](../analysis/episode-metrics.csv)含原始数组路径与 SHA-256。[录像索引](../videos/index.md)区分测量与录制。",
            "同初态的 4 条无录像复跑也纳入本索引及哈希核验，见[重复性报告](repeatability.md)；这些轨迹及录像不重复计入原来的 10 个独立初态。",
            "",
            "`episode_XX.npz` 保存逐步 `t` 和 `state[T,B,13]`：xyz、xyzw、线速度、角速度，单位 m、m/s、rad/s；初态、尺寸、惯量、支撑位姿也在同一文件。底层 float32；存储为 float64 不增加精度。未压缩 `_state.npy` 与 NPZ 已逐项核对。本次两端均不导出接触力，未填零。",
            "",
            "每次运行 `config/` 保存当时入口、协议、配置、初态和 Spec；`run.json` 记录命令、环境、源码提交、实际参数与状态。当前显式开启 DexUni 刚体睡眠，PhysX 保留默认睡眠阈值；DexUni 三项接触微动抑制阈值明确保留默认值。旧无睡眠长时扫描另存，不混入本次统计。",
            "",
            "旧协议首次分组单层初始化失败：接口要求 5–704 个非单体自由刚体。新扫描从 5 层开始；旧失败与被用户叫停的长时运行保留在相邻 stack-limit/。",
            "",
            "在项目目录、embodichain 环境中运行 `python -m scripts.benchmark.tech_reports.physics.experiments.p02_stack_sweep` 可补齐未完成测量；程序保留已有结果并逐平台串行运行。离线分析运行 `python -m scripts.benchmark.tech_reports.reporting.p02_stack_report`。每个已完成原始命令亦在 `run.json`；重放时必须换新的输出目录。",
            "",
            "完整源码快照在 `../analysis/source/`，源码、全部运行文件及环境版本在本页所链接的 evidence-manifest.json。扩展前的原报告、Spec 与脚本留在 `../../history/20261022_before_stack_limit/`；原主测文件未被替换。",
        ]
    (root / "reports" / "data-index.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            dict(
                main_trajectories=len(main),
                attempts=len(runs),
                files=len(files),
                failed=sum(r["status"] != "completed" for r in runs),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
