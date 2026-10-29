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
"""Build the existing-task comparison from retained measurements only."""

from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any

TECH = Path(__file__).resolve().parents[2]
RUNS = TECH.parents[2] / "outputs/benchmarks/tech_reports/dexuni"
__all__ = ["main"]


def _read(path: Path) -> Any:
    """Read one retained result document."""
    return json.loads(path.read_text())


def _stat(values: list[float]) -> dict[str, Any]:
    """Summarize independent process means."""
    return dict(
        mean_ms=statistics.mean(values),
        sd_ms=statistics.stdev(values) if len(values) > 1 else None,
        processes=len(values),
    )


def main() -> None:
    """Build the task index and comparison without rerunning simulation."""
    tasks = []

    def add(
        code: str,
        name: str,
        folder: str,
        report: str,
        video: str,
        config: str,
        outcome: str,
        stats: dict,
        source: str,
        scope: str = "各5次正式测量",
    ) -> None:
        path = RUNS / folder
        for rel in (report, video, source):
            assert (path / rel).is_file(), path / rel
        tasks.append(
            dict(
                code=code,
                name=name,
                folder=folder,
                report=f"{folder}/{report}",
                video=f"{folder}/{video}",
                configuration=config,
                outcome=outcome,
                stats=stats,
                scope=scope,
                source=f"{folder}/{source}",
                source_sha256=hashlib.sha256((path / source).read_bytes()).hexdigest(),
            )
        )

    for code, task, name, folder, video in [
        ("S01-C", "s01c", "软块压缩", "s01_fixed_five", "videos/s01c_comparison.mp4"),
        ("S01-T", "s01t", "软块拉伸", "s01_fixed_five", "videos/s01t_comparison.mp4"),
        ("S02", "s02", "大变形抓取搬运", "s02_fixed_five", "videos/s02_comparison.mp4"),
    ]:
        raw = _read(RUNS / folder / "attempts.json")
        stats = {}
        for platform in ("embodichain", "isaaclab"):
            rows = [
                r
                for r in raw
                if r["task"] == task
                and r["platform"] == platform
                and r["mode"] == "measurement"
            ]
            assert len(rows) == 5 and all(
                r["returncode"] == 0 and r["metrics"]["success"] for r in rows
            )
            stats[platform] = _stat([float(r["metrics"]["step_ms"]) for r in rows])
        add(
            code,
            name,
            folder,
            "REPORT.md",
            video,
            "普通VBD；S01两侧40×10；S02两侧10子步",
            "两侧当前5次均通过既定任务检查；固定初态，不代表普遍成功率",
            stats,
            "attempts.json",
        )
    for code, name, folder, video, config, outcome in [
        (
            "S03",
            "布角提起与释放",
            "s03_cloth_corner_lift",
            "videos/s03_comparison.mp4",
            "两侧10×10；SDK优化后有效批次",
            "两侧5次完整有限执行；不以协议完成率代替布形精度",
        ),
        (
            "S04",
            "刚块/软块双侧夹持",
            "s04_rigid_soft_squeeze_grasp",
            "videos/s04_comparison.mp4",
            "两侧12×10；SDK优化后有效批次",
            "两侧5次完整有限执行；旧错误步长批次不计入",
        ),
    ]:
        raw = list(
            csv.DictReader((RUNS / folder / "measurement/results/runs.csv").open())
        )
        stats = {}
        for p in ("embodichain", "isaaclab"):
            values = [float(r["cost_time_ms"]) for r in raw if r["platform"] == p]
            assert len(values) == 5
            stats[p] = _stat(values)
        add(
            code,
            name,
            folder,
            "REPORT.md",
            video,
            config,
            outcome,
            stats,
            "measurement/results/runs.csv",
        )
    for code, name, folder, video, config, outcome in [
        (
            "M01",
            "双臂软梁弯曲搬运",
            "m01_soft_beam_bend",
            "videos/m01_comparison.mp4",
            "两侧12×10，two-way动态关节；Isaac CouplerProxy",
            "下垂/端距响应接近；未设置物理质量通过阈值",
        ),
        (
            "M02",
            "方布单次对折",
            "m02_cloth_single_fold",
            "videos/m02_comparison.mp4",
            "EC surface-fast8；Isaac普通10；均10子步",
            "两侧完成流程；左右角对齐各有优劣，未设折叠通过线",
        ),
    ]:
        raw = _read(RUNS / folder / "measurement/results/summary.json")
        stats = {
            p: dict(
                mean_ms=raw[p]["mean_ms"],
                sd_ms=raw[p]["sd_ms"],
                processes=raw[p]["episodes"],
            )
            for p in ("embodichain", "isaaclab")
        }
        add(
            code,
            name,
            folder,
            "REPORT.md",
            video,
            config,
            outcome,
            stats,
            "measurement/results/summary.json",
        )
    folder = "m03_four_layer_cloth_pick_place"
    raw = _read(RUNS / folder / "measurement/results.json")["results"]
    stats = {
        p: dict(
            mean_ms=raw[p]["latency_ms"]["mean"],
            sd_ms=raw[p]["latency_ms"]["std"],
            processes=5,
        )
        for p in ("embodichain", "isaaclab")
    }
    add(
        "M03",
        "四层布夹取铺放",
        folder,
        "REPORT.md",
        "preview/comparison.mp4",
        "EC fast8+自适应；Isaac普通20；均10子步",
        "抬升均5/5；20mm放置线EC5/5、Isaac0/5；全部样本计时均保留",
        stats,
        "measurement/results.json",
    )
    folder = "m04_bimanual_tshirt_single_fold"
    raw = _read(RUNS / folder / "measurement/results.json")["results"]
    stats = {
        p: dict(mean_ms=raw[p]["mean_ms"], sd_ms=raw[p]["sample_sd_ms"], processes=5)
        for p in ("embodichain", "isaaclab")
    }
    add(
        "M04",
        "双臂T恤对折",
        folder,
        "measurement/REPORT.md",
        "preview/comparison.mp4",
        "EC fast16+自适应；Isaac普通20；均10子步",
        "两侧原6项检查均5/5；最终下摆XY误差EC约26mm、Isaac约46mm",
        stats,
        "measurement/results.json",
    )
    folder = "m05_inflatable_pouch_load"
    add(
        "M05",
        "充气包与刚块压载",
        folder,
        "REPORT.md",
        "preview/comparison.mp4",
        "EC普通24×8，封闭等温气腔",
        "EC已演示压载、恢复和搬运；本次Isaac版本缺少所需气腔，视频为DexSim/EC",
        {},
        "REPORT.md",
        "仅功能演示，未测性能",
    )
    folder = "m06_franka_tshirt_fold"
    raw = _read(RUNS / folder / "measurement/results/summary.json")["platforms"]
    stats = {
        p: dict(
            mean_ms=raw[p]["mean_ms"]["mean"],
            sd_ms=raw[p]["mean_ms"]["sample_std"],
            processes=5,
        )
        for p in ("embodichain", "isaaclab")
    }
    add(
        "M06",
        "Franka单臂T恤折叠",
        folder,
        "REPORT.md",
        "preview/comparison.mp4",
        "EC fast8+自适应；Isaac普通15；均10子步",
        "当前保留样本两侧折叠检查5/5；EC第5次按用户要求重测替换",
        stats,
        "measurement/results/summary.json",
        "每侧保留5次，EC有1次指定替换",
    )
    folder = "c01_supermarket_packing"
    raw = _read(RUNS / folder / "measurement/results.json")
    stats = {
        "embodichain": dict(
            mean_ms=raw["mean_of_process_means_ms"],
            sd_ms=raw["sample_sd_process_means_ms"],
            processes=5,
        )
    }
    add(
        "C01",
        "面包/奶包双商品装袋",
        folder,
        "measurement/REPORT.md",
        "platforms/embodichain/preview/video.mp4",
        "EC普通24×16；左面包后右奶包",
        "EC5/5两件留存且无倒置；Isaac仅单面包，不能与双商品算速度比",
        stats,
        "measurement/results.json",
        "EC正式5次；Isaac完整场景不适用",
    )
    folder = "c02_bag_frame_packing"
    raw = _read(RUNS / folder / "performance_comparison/results.json")["configurations"]
    stats = {}
    for row in raw:
        if "complete_64s" in row:
            p = "embodichain" if row["label"].startswith("Embodi") else "isaaclab"
            stats[p] = dict(
                mean_ms=row["complete_64s"]["physics_s"]["mean_ms"],
                sd_ms=None,
                processes=1,
            )
    add(
        "C02",
        "袋体扶正、双载荷装填提袋",
        folder,
        "performance_comparison/REPORT.md",
        "budget_reduction/iteration_comparison.mp4",
        "EC DexUni12×6；Isaac普通96×24；另保留24×12失败版",
        "EC与96×24录像完成载荷提袋；24×12倒袋失败。96×24独立计时终态纸盒超界0.191mm",
        stats,
        "performance_comparison/results.json",
        "单次64s摸测；低配另有12s短测",
    )
    assert len(tasks) == 13
    lines = [
        "# 已有任务对比总览（2026-10-27）",
        "",
        "本页仅汇总已有结果，没有新增仿真或重跑测量。共13个任务项（S01-C/T分别计），保留当前正式数据、视频、原始状态、复现输入与源码。清理范围及证据见[清理记录](../maintenance/2026-10-27/README.md)。",
        "",
        "主要口径为无相机、GPU同步完成的物理外步耗时；越小越快，不包含渲染，不是整应用FPS。±为5个进程均值的样本标准差；单次摸测不附虚假的SD。不同任务的计时边界、控制、SDK版本与求解策略以各任务原报告为准，不能跨任务合成总榜。硬件为RTX5090 D v2。",
        "",
        "## 性能概览",
        "",
        "| 任务 | 场景 | EmbodiChain ms/步 | Isaac Lab ms/步 | EC耗时降低 | 数据级别 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for t in tasks:

        def fmt(p: str) -> str:
            s = t["stats"].get(p)
            if not s:
                return (
                    "未测"
                    if t["code"] == "M05" and p == "embodichain"
                    else "不适用/无同场景数据"
                )
            return f"{s['mean_ms']:.3f}" + (
                f" ± {s['sd_ms']:.3f}" if s["sd_ms"] is not None else "（单次）"
            )

        ratio = "—"
        if len(t["stats"]) == 2:
            e = t["stats"]["embodichain"]["mean_ms"]
            i = t["stats"]["isaaclab"]["mean_ms"]
            ratio = f"{100*(1-e/i):.1f}%"
            t["ec_latency_reduction_percent"] = 100 * (1 - e / i)
        lines.append(
            f"| [{t['code']}]({t['report']}) | {t['name']} | {fmt('embodichain')} | {fmt('isaaclab')} | {ratio} | {t['scope']} |"
        )
    lines += [
        "",
        "上述百分比只描述该行已测配置的物理推进时间。M02/M03/M04/M06的surface-fast与普通VBD属于统一任务条件下的求解策略比较；基础迭代、自适应回退、Jacobi/Chebyshev/多层校正均是策略组成，一次迭代不等价。C02还存在平台机器人轨迹差异，不能把巨大耗时差单独归因于求解器。",
        "",
        "## 效果、配置与视频",
        "",
        "参数×号按“迭代次数×子步数”书写，fast自适应与普通VBD不能按乘积等价换算。",
        "",
        "| 任务 | 当前配置 | 已观察到的效果及限制 | 视频 |",
        "| --- | --- | --- | --- |",
    ]
    for t in tasks:
        lines.append(
            f"| {t['code']} | {t['configuration']} | {t['outcome']} | [查看]({t['video']}) |"
        )
    lines += [
        "",
        "M05本次安装的Isaac原生Newton缺少封闭等温气腔，[支持证据](m05_inflatable_pouch_load/analysis/isaaclab_support.md)仍保留。C01另保留[Isaac单面包视频](c01_supermarket_packing/platforms/isaaclab/preview/video.mp4)及[异负载摸测](c01_supermarket_packing/pilot_performance/REPORT.md)，不冒充双商品等价比较。M06的指定重测记录和全部正式批次原始样本保留。",
        "",
        "## C02低配与成功效果的取舍",
        "",
        "当前录到成功装袋/载荷提袋的Isaac配置为96×24，24×12倒袋后未完成任务；尚未证明96×24是最低所需预算。两档录像与EC均为实际64秒，无冻结补帧。以下单独比较相同前12秒、剔除前30步的690个稳态样本，不能与上表64秒均值混用。",
        "",
        "| 配置 | 前12秒物理推进均值 ms/步 | 计时来源 | 完整录像结果 |",
        "| --- | --- | --- | --- |",
    ]
    for x in raw:
        label = x["label"]
        s = x["common_initial_12s"]["physics_s"]["mean_ms"]
        low = "24x12" in label
        lines.append(
            f"| {label} | {s:.3f} | {'新测一次12s' if low else '复用已有64s计时前缀'} | {'倒袋，装袋/提袋失败' if low else '当前录像完成装袋和载荷提袋'} |"
        )
    lines += [
        "",
        "C02高配视频的原严格袋口弯曲限值未全过（51.285mm>30mm）；独立64秒性能轨迹在悬停阶段装载成立，但终态纸盒X边界超0.191mm。保形失败及失败计时均照实保留，不把录像时长完成当作任务成功。",
        "",
        "## 当前结论",
        "",
        "- 简单软体/布角任务S01–S04与M01的物理步耗时差距较小，约2%–22%；读回与端到端开销需另看。",
        "- 布料优化组M03/M04/M06的当前配置分别约3.67×、3.30×、2.22×逆耗时比；效果按各任务检查描述，不外推通用物理精度。",
        "- M05及C01完整含气腔场景缺少本次Isaac等价能力；不生成虚构的平台速度比。",
        "- C02展示了明显的速度/任务效果取舍，目前仅有摸测，不能包装为五次正式统计。",
        "",
        "完整机器可读数据与来源哈希见[comparison.json](comparison.json)，可分享表格见[comparison.csv](comparison.csv)。各任务原始协议、运行日志、状态、时序和源码快照继续在原目录，旧调参档案的精简证据与删除账本见清理记录。",
    ]
    (RUNS / "COMPARISON.md").write_text("\n".join(lines) + "\n")
    (RUNS / "comparison.json").write_text(
        json.dumps(
            dict(
                generated_date="2026-10-27",
                new_simulations=0,
                tasks=tasks,
            ),
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    with (RUNS / "comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "task",
                "name",
                "ec_mean_ms",
                "ec_sample_sd_ms",
                "isaac_mean_ms",
                "isaac_sample_sd_ms",
                "scope",
                "configuration",
                "outcome",
                "video",
                "report",
            ]
        )
        for t in tasks:
            e = t["stats"].get("embodichain", {})
            i = t["stats"].get("isaaclab", {})
            writer.writerow(
                [
                    t["code"],
                    t["name"],
                    e.get("mean_ms"),
                    e.get("sd_ms"),
                    i.get("mean_ms"),
                    i.get("sd_ms"),
                    t["scope"],
                    t["configuration"],
                    t["outcome"],
                    t["video"],
                    t["report"],
                ]
            )
    (RUNS / "README.md").write_text(
        "# 当前结果入口\n\n本轮整理已有结果。\n\n[13个任务项完整对比](COMPARISON.md) · [数据CSV](comparison.csv) · [来源JSON](comparison.json) · [清理与保留记录](../maintenance/2026-10-27/README.md)\n\n"
        + "\n".join(
            f"- **{t['code']} {t['name']}**：[报告]({t['report']}) · [视频]({t['video']})。{t['scope']}。"
            for t in tasks
        )
        + "\n"
    )
    print("Built 13-task comparison, CSV, JSON and current index.")


if __name__ == "__main__":
    main()
