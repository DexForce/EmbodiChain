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
"""Rebuild the EC-only C01 five-process report without executing simulation.

Run: .venv/bin/python -m scripts.benchmark.tech_reports.reporting.dexuni.report_c01
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from scripts.benchmark.tech_reports.physics.experiments.dexuni.c01_common import OUTPUT
from scripts.benchmark.tech_reports.physics.experiments.dexuni.c01_replay import (
    sha256,
    write_json,
)
from scripts.benchmark.tech_reports.physics.experiments.dexuni.c01_supermarket_packing.tools.analyze import (
    _analyze,
)

__all__ = ["main"]


def main() -> None:
    """Validate every attempt, aggregate process means and retain physical data."""
    root = OUTPUT / "measurement"
    protocol = json.loads((root / "protocol.json").read_text())
    for rel, digest in protocol["protected_sha256"].items():
        assert sha256(OUTPUT / rel) == digest, rel
    for rel, digest in protocol["source_sha256"].items():
        assert sha256(OUTPUT.parents[1] / rel) == digest, rel
        assert sha256(root / "source" / rel) == digest, rel
    attempts = json.loads((root / "attempts.json").read_text())
    assert len(attempts) == 5 and all(
        x["status"] == "completed" and x["exit_code"] == 0 for x in attempts
    )
    records, curves = [], {}
    with np.load(
        OUTPUT / "platforms/embodichain/analysis/initial_model.npz"
    ) as reference:
        for repeat in range(1, 6):
            name = f"r{repeat:02d}"
            run = root / "embodichain" / name
            manifest = json.loads((run / "manifest.json").read_text())
            summary = json.loads((run / "summary.json").read_text())
            assert manifest["status"] == "completed" and manifest["formal_measurement"]
            assert manifest["protocol_sha256"] == sha256(root / "protocol.json")
            assert summary["performance_eligible"] and summary["formal_measurement"]
            for group, folder in [
                ("input_sha256", "inputs"),
                ("execution_source_sha256", "source"),
            ]:
                for rel, digest in manifest[group].items():
                    assert sha256(run / folder / rel) == digest, (name, group, rel)
            with (run / "steps.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            assert len(rows) == 1686
            assert [int(r["frame"]) for r in rows] == list(range(1, 1687))
            assert [r["steady"] == "True" for r in rows] == [
                n > 30 for n in range(1, 1687)
            ]
            elapsed = np.array([float(r["completed_step_s"]) for r in rows])
            assert np.isfinite(elapsed).all() and (elapsed > 0).all()
            for key, value in dict(
                cost_time_ms=elapsed[30:].mean() * 1000,
                median_ms=np.median(elapsed[30:]) * 1000,
                p95_ms=np.percentile(elapsed[30:], 95) * 1000,
            ).items():
                assert np.isclose(summary[key], value, rtol=1e-12, atol=1e-12), (
                    name,
                    key,
                )
            physical, arrays = _analyze(run)
            with (
                np.load(run / "preview/states.npz") as states,
                np.load(run / "inputs/joint_commands.npz") as commands,
            ):
                assert np.max(np.abs(states["time"] - commands["time"])) < 1e-7
                assert np.max(np.abs(states["qpos"] - commands["targets"])) < 2e-5
                assert (
                    np.linalg.norm(states["tcp"] - commands["tcp_baked"], axis=-1).max()
                    < 2e-5
                )
                assert np.allclose(
                    [float(row["actual_time_s"]) for row in rows],
                    states["time"][1:],
                    rtol=0,
                    atol=1e-7,
                )
            assert physical["full_finite_states"] and physical["samples"] == 1687
            assert abs(physical["actual_duration_s"] - 28.1) < 1e-7
            assert (
                physical["actual_outer_steps"] == 1686
                and physical["graph_status"] == "captured"
            )
            assert (
                physical["max_joint_replay_error"] < 2e-5
                and physical["max_tcp_replay_error_m"] < 2e-5
            )
            actual = manifest["effective"]
            assert actual["num_substeps"] == 16 and actual["vbd"]["iterations"] == 24
            with np.load(run / "analysis/initial_model.npz") as initial:
                assert set(initial.files) == set(reference.files)
                assert all(
                    np.array_equal(initial[k], reference[k]) for k in initial.files
                )
            write_json(run / "analysis/current_metrics.json", physical)
            np.savez_compressed(run / "analysis/current_plot_data.npz", **arrays)
            records.append(
                dict(
                    repeat=repeat,
                    **summary,
                    physical=physical,
                    states_sha256=sha256(run / "preview/states.npz"),
                    csv_sha256=sha256(run / "steps.csv"),
                )
            )
            curves[name + "_actual_time_s"] = np.array(
                [float(r["actual_time_s"]) for r in rows]
            )
            curves[name + "_completed_step_ms"] = elapsed * 1000
    means = np.array([r["cost_time_ms"] for r in records])
    result = dict(
        platform="embodichain",
        task="C01",
        formal_measurement=True,
        independent_processes=5,
        mean_of_process_means_ms=float(means.mean()),
        sample_sd_process_means_ms=float(means.std(ddof=1)),
        protocol_sha256=sha256(root / "protocol.json"),
        records=records,
        no_other_platform_comparison=True,
    )
    write_json(root / "results.json", result)
    np.savez_compressed(root / "timing_plot_data.npz", **curves)
    lines = [
        "# C01 双商品装袋：EmbodiChain 正式性能记录",
        "",
        f"按用户最新要求仅测 EmbodiChain。固定前五个独立无相机进程，均值为 **{means.mean():.6f} ± {means.std(ddof=1):.6f} ms/外步**（五个进程均值及其样本标准差）。Isaac Lab 无本场景正式结果；历史单面包摸测保留在[此前记录](../pilot_performance/REPORT.md)，不作加速比或质量排名。",
        "",
        "当前左面包后右牛奶、28.1秒完整轨迹不变，普通VBD24次×16子步、1/60秒外步、kinematic/one_way、真实等温气腔、原有FilmDrag及接触设置，CUDA Graph captured。各进程1686外步，排除前30步预热，剩余1656步取算术平均，再对五个进程等权汇总。",
        "",
        "计时是实际公共外步及最终Warp CUDA设备同步等待，计时前也同步；命令更新在既有闭包内，读回、检查、显存轮询、序列化、初始化、相机和在线IK在计时外。不同动作阶段的状态连续变化，不把这些步当成独立进程重复。",
        "",
        "## Time & Memory",
        "",
        "| 进程 | 平均外步 ms | 中位 ms | P95 ms | 显存采样最大 MiB | RSS采样最大 MiB | 读回平均 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in records:
        lines.append(
            f"| r{r['repeat']:02d} | {r['cost_time_ms']:.6f} | {r['median_ms']:.6f} | {r['p95_ms']:.6f} | {r['sampled_process_gpu_max_mib']:.0f} | {r['sampled_cpu_rss_max_mib']:.1f} | {r['readback_mean_ms']:.3f} |"
        )
    lines += [
        f"| 五个进程均值 ± 样本SD | {means.mean():.6f} ± {means.std(ddof=1):.6f} | N/A | N/A | N/A | N/A | N/A |",
        "",
        "显存是每120步及起止时刻nvidia-smi对当前进程的采样最大值，RSS同样为采样值，均在计时外。不是单次调用增量、瞬时峰值或整卡占用，cpu_delta/gpu_delta/peak_gpu均为N/A；全部轮询记录保留。",
        "",
        "## Success & Other Metrics",
        "",
        "| 进程 | 实际秒 / 外步 / 有限状态 | Graph | 商品中心留存 | 倒置tet数 / 最小有符号体积比 | 面包持握漂移 mm | 奶包封边漂移 mm | 末秒袋速度P95 mm/s |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for r in records:
        p = r["physical"]
        bread, milk = p["goods"]["bread"], p["goods"]["pouch"]
        retained = sum(p["final_centre_in_bag"].values())
        lines.append(
            f"| r{r['repeat']:02d} | 28.1 / 1686 / 1687 | captured | {retained}/2 | {bread['inverted_tet_pose_count']} / {bread['minimum_signed_tet_volume_ratio']:.6f} | {bread['max_centre_to_tcp_drift_during_hold_m']*1000:.3f} | {milk['max_seam_marker_to_tcp_drift_during_hold_m']*1000:.3f} | {p['windows']['final']['bag_node_speed_p95_m_s']*1000:.3f} |"
        )
    lines += [
        "",
        "全程实际时钟、具名关节/TCP、有限状态、预算和图捕获逐次核验；持握、体积和留存仅描述，不按最终形状筛选或重测。中心留存采用冻结袋口/袋底边界，不证明完整包含。本批不重复FCL离散表面检查，不将录像阶段的检查冒充本次检查。",
        "",
        "## Leaderboard / coverage",
        "",
        "| 当前正式范围 | 完整执行 / 尝试 | 统计 | 排名 / 平台加速比 |",
        "| --- | --- | --- | --- |",
        f"| EmbodiChain，双商品含气腔 | 5/5 | {means.mean():.6f} ± {means.std(ddof=1):.6f} ms/外步 | N/A，仅此平台 |",
        "",
        "5/5是本批完成记录，不能外推鲁棒性或物理质量。设备RTX5090Dv2、Warp1.17、float32/cuda:0；EC使用Python3.11、SDK0.5.1+dfceb00、Newton1.6.0。完整版本、模块路径、线程数和CPU亲和性在各manifest；场景/源码/输入在[冻结协议](protocol.json)及source目录。各进程原始状态、逐步CSV、摘要、采样内存、初始化数组和物理绘图原数组均在embodichain/r01～r05。",
        "",
        "[全部尝试](attempts.json) · [批次日志](batch.log) · [精确汇总](results.json) · [最终核验](validation.json) · [原始耗时曲线数据](timing_plot_data.npz)。五次都保留，不修改此前有效预览、历史摸测或其他任务的测量。这里只扩展独立benchmark，不涉及生产API/路由，无项目上下文更新。",
    ]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    write_json(
        root / "validation.json",
        dict(
            status="passed",
            protected_files_unchanged=len(protocol["protected_sha256"]),
            frozen_source_files_unchanged=len(protocol["source_sha256"]),
            complete_processes=5,
            actual_seconds_each=28.1,
            actual_steps_each=1686,
            warmup_excluded_each=30,
            measured_steps_each=1656,
            cuda_graph_each="captured",
            source_and_input_hashes_verified=True,
            csv_statistics_independently_recomputed=True,
            initialized_model_arrays_equal_current_preview=True,
            no_shape_filtering=True,
        ),
    )
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
