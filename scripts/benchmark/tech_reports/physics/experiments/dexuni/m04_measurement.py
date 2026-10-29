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
"""Freeze, run and report five independent M04 episodes per platform.

Run: python -m scripts.benchmark.tech_reports.physics.experiments.dexuni.m04_measurement --prepare
Then use --run and --report. Each mode preserves accepted artifacts and prior attempts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np

from scripts.benchmark.tech_reports.physics.experiments.dexuni.m04_common import (
    OUTPUT,
    digest,
    write_json,
)

__all__ = ["measurement_directory", "main"]
ROOT = OUTPUT / "measurement"
TECH = OUTPUT.parents[1]
REPO = TECH.parents[2]
PLATFORMS = ("embodichain", "isaaclab")
PYTHONS = {
    "embodichain": REPO / ".venv/bin/python",
    "isaaclab": Path("/home/oem/app/IsaacLab/.venv/bin/python"),
}


def _foreign_processes() -> list[str]:
    """Read current compute processes other than this process."""
    result = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader",
        ],
        text=True,
    )
    return [
        line
        for line in result.splitlines()
        if line.strip() and int(line.split(",", 1)[0]) != os.getpid()
    ]


def _verify(protocol: dict[str, Any]) -> None:
    """Require identical accepted artifacts and frozen execution sources."""
    for name, expected in protocol["protected_sha256"].items():
        if digest(OUTPUT / name) != expected:
            raise ValueError(f"Protected artifact changed: {name}")
    for name, expected in protocol["source_sha256"].items():
        if (
            digest(TECH / name) != expected
            or digest(ROOT / "source" / name) != expected
        ):
            raise ValueError(f"Frozen measurement source changed: {name}")


def measurement_directory(platform: str, repeat: int) -> Path:
    """Reserve a formal sample after checking authorization, hashes and GPU idleness."""
    if platform not in PLATFORMS or repeat not in range(1, 6):
        raise ValueError("M04 formal measurement requires platforms and repeats1..5")
    protocol = json.loads((ROOT / "protocol.json").read_text())
    if not protocol["authorized"] or protocol["episodes_per_platform"] != 5:
        raise ValueError("Expected explicit fixed-five formal authorization")
    _verify(protocol)
    if _foreign_processes():
        raise RuntimeError("Wait for foreign GPU compute processes before measurement")
    output = ROOT / platform / f"r{repeat:02d}"
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "manifest.json",
        dict(status="initializing", platform=platform, repeat=repeat),
    )
    return output


def _prepare() -> None:
    """Freeze the accepted preview, the completed pilot and current measurement sources."""
    pilot = json.loads((OUTPUT / "pilot_performance/protocol.json").read_text())
    for platform in PLATFORMS:
        manifest = json.loads(
            (OUTPUT / "pilot_performance" / platform / "manifest.json").read_text()
        )
        if manifest["status"] != "completed" or not all(manifest["checks"].values()):
            raise ValueError("Expected both completed pilots")
    ROOT.mkdir(exist_ok=False)
    protected = {
        str(p.relative_to(OUTPUT)): digest(p)
        for p in sorted(OUTPUT.rglob("*"))
        if p.is_file() and ROOT not in p.parents and "__pycache__" not in p.parts
    }
    sources = set(pilot["source_sha256"])
    sources.add(str(Path(__file__).resolve().relative_to(TECH)))
    source_hashes = {}
    for name in sorted(sources):
        target = ROOT / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(TECH / name, target)
        source_hashes[name] = digest(target)
    protocol = {
        k: v
        for k, v in pilot.items()
        if k
        not in (
            "protected_sha256",
            "source_sha256",
            "scope",
            "authorization",
            "frozen_utc",
            "episodes_per_platform",
            "limitations",
            "order",
            "formal_measurement",
        )
    }
    protocol.update(
        scope="fixed five formal episodes per platform",
        authorized=True,
        formal_measurement=True,
        authorization="Fixed-configuration performance measurement after preview acceptance",
        frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        episodes_per_platform=5,
        process_policy="fresh independent process for each sample; fixed first five; all attempts retained; no quality-based filtering or best-of selection",
        order=[
            dict(repeat=i, platform=p)
            for i in range(1, 6)
            for p in (PLATFORMS if i % 2 else PLATFORMS[::-1])
        ],
        statistics="mean and sample SD (ddof=1) of five process means; all1890 steady outer steps per process retained",
        limitations="solver-level strategy comparison, not isolated equal-iteration kernels or end-to-end application superiority; process framebuffer sampling is not transient peak; foreign compute sampled every120steps, not continuously",
        protected_sha256=protected,
        source_sha256=source_hashes,
    )
    write_json(ROOT / "protocol.json", protocol)
    print(
        f"Frozen {len(protected)} protected files and {len(source_hashes)} execution sources",
        flush=True,
    )


def _run() -> None:
    """Run the fixed alternating batch once, stopping and retaining any failed attempt."""
    protocol = json.loads((ROOT / "protocol.json").read_text())
    _verify(protocol)
    with (ROOT / "attempts.json").open("x") as stream:
        json.dump([], stream)
    attempts = []
    for sample in protocol["order"]:
        platform, repeat = sample["platform"], sample["repeat"]
        while foreign := _foreign_processes():
            print(f"Waiting for GPU compute: {foreign}", flush=True)
            time.sleep(10)
        command = [
            str(PYTHONS[platform]),
            str(TECH / f"physics/platforms/dexuni/{platform}/m04.py"),
            "--measure",
            "--repeat",
            str(repeat),
        ]
        attempt = dict(
            sample,
            command=command,
            status="running",
            started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        )
        attempts.append(attempt)
        write_json(ROOT / "attempts.json", attempts)
        print(f"Starting {platform} r{repeat:02d}", flush=True)
        with (ROOT / f"{platform}_r{repeat:02d}.log").open("x") as log:
            result = subprocess.run(
                command,
                cwd=REPO,
                env=dict(os.environ, PYTHONUNBUFFERED="1"),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        attempt.update(
            returncode=result.returncode,
            ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        )
        manifest = (
            json.loads((ROOT / platform / f"r{repeat:02d}/manifest.json").read_text())
            if (ROOT / platform / f"r{repeat:02d}/manifest.json").exists()
            else {}
        )
        valid = (
            result.returncode == 0
            and manifest.get("status") == "completed"
            and all(manifest.get("checks", {"missing": False}).values())
        )
        attempt["status"] = "completed" if valid else "failed"
        write_json(ROOT / "attempts.json", attempts)
        if not valid:
            raise RuntimeError(
                f"Retained failed attempt: {platform} r{repeat:02d}; inspect log"
            )
        print(
            f"Completed {platform} r{repeat:02d}: {manifest['summary']['cost_time_ms']:.6f} ms",
            flush=True,
        )
    _verify(protocol)
    print(
        "All ten independent episodes completed; sources and protected artifacts unchanged",
        flush=True,
    )


def _report() -> None:
    """Analyze every retained trajectory and summarize all five process means per side."""
    protocol = json.loads((ROOT / "protocol.json").read_text())
    _verify(protocol)
    attempts = json.loads((ROOT / "attempts.json").read_text())
    assert len(attempts) == 10 and all(a["status"] == "completed" for a in attempts)
    results = {}
    for platform in PLATFORMS:
        samples = []
        for repeat in range(1, 6):
            output = ROOT / platform / f"r{repeat:02d}"
            manifest = json.loads((output / "manifest.json").read_text())
            assert manifest["status"] == "completed" and manifest["formal_measurement"]
            assert manifest["protocol_sha256"] == digest(ROOT / "protocol.json")
            assert (
                all(manifest["checks"].values())
                and manifest["no_foreign_compute_sampled"]
            )
            timings = np.genfromtxt(
                output / "timing.csv",
                delimiter=",",
                names=True,
                dtype=None,
                encoding=None,
            )
            seconds = timings["step_s"][30:]
            assert len(seconds) == 1890 and np.isclose(
                1000 * seconds.mean(),
                manifest["summary"]["cost_time_ms"],
                rtol=0,
                atol=1e-10,
            )
            with tempfile.TemporaryDirectory(
                prefix=f"m04_formal_{platform}_r{repeat:02d}_"
            ) as scratch:
                scene = Path(scratch)
                (scene / "inputs").symlink_to(
                    OUTPUT / "inputs", target_is_directory=True
                )
                (scene / "preview").symlink_to(output, target_is_directory=True)
                with (output / "analysis.log").open("w") as log:
                    subprocess.run(
                        [
                            sys.executable,
                            str(
                                TECH
                                / "physics/experiments/dexuni/m04_bimanual_tshirt_single_fold/tools/analyze.py"
                            ),
                            "--run",
                            str(scene),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                shutil.copytree(
                    scene / "analysis", output / "analysis", dirs_exist_ok=True
                )
            physical = json.loads(
                (output / "analysis/physical_checks.json").read_text()
            )
            samples.append(
                dict(
                    repeat=repeat,
                    summary=manifest["summary"],
                    readback_mean_ms=manifest["readback_mean_ms"],
                    effective=manifest["effective"],
                    checks=manifest["checks"],
                    environment=manifest["environment"],
                    gpu_memory=manifest["gpu_memory"],
                    physical=physical,
                    files_sha256={
                        name: digest(output / name)
                        for name in (
                            "manifest.json",
                            "timing.csv",
                            "states.npz",
                            "initial_model.npz",
                            "analysis/physical_checks.json",
                        )
                    },
                )
            )
        means = np.array([s["summary"]["cost_time_ms"] for s in samples])
        results[platform] = dict(
            samples=samples,
            process_mean_ms=means.tolist(),
            mean_ms=float(means.mean()),
            sample_sd_ms=float(means.std(ddof=1)),
            readback_mean_ms=float(np.mean([s["readback_mean_ms"] for s in samples])),
            physical_successes=sum(
                all(s["physical"]["candidate_checks"].values()) for s in samples
            ),
            total_steady_steps=9450,
            gpu_sampled_max_mib=max(
                s["gpu_memory"]["gpu_process_peak_mib"]
                for s in samples
                if s["gpu_memory"]["gpu_tracking_available"]
            ),
        )
    ec, il = (results[p]["mean_ms"] for p in PLATFORMS)
    summary = dict(
        task="M04",
        formal_measurement=True,
        generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        protocol_sha256=digest(ROOT / "protocol.json"),
        protected_artifacts_verified=len(protocol["protected_sha256"]),
        execution_sources_verified=len(protocol["source_sha256"]),
        results=results,
        ec_latency_reduction_percent=(1 - ec / il) * 100,
        inverse_latency_ratio=il / ec,
    )
    write_json(ROOT / "results.json", summary)
    lines = [
        "# M04 两平台正式性能测量",
        "",
        "当前认可版T恤对折任务；每侧五个独立无相机进程，按轮次交替先后顺序。每次实际32秒、1920外步/19200子步、1921完整有限状态；前30步预热剔除，各统计1890步，每侧总计9450个稳态样本。保留固定前五次全部尝试，未按最终布形筛选或替换。",
        "",
        "RTX 5090 D v2；两侧统一6436节点/12736面T恤、W1/Pika、材料、冻结关节与松爪指令、60Hz/10子步、one_way/kinematic和CUDA Graph。EC为surface-fast基础16+自适应、polish1、multilevel关闭；IsaacLab为上游普通VBD20。这是在共同任务条件下比较求解器优化策略，迭代次数及自适应回退属于策略组成。EC DexSim dfceb00e/Newton1.7.0，IsaacLab28.0.0/isaaclab-newton9.1.4/Newton1.6.1，Warp均1.17.0；详见有效配置与版本记录。",
        "",
        "## Time & Memory",
        "",
        "| 平台 | cost_time_ms 均值±样本SD | 五次进程均值ms | 读回均值ms | CPU Δ MB | GPU Δ MB | 瞬态peak GPU MB | 采样进程显存最大MiB |",
        "|---|---:|---|---:|---|---|---|---:|",
    ]
    for p, r in results.items():
        means = ", ".join(f"{x:.3f}" for x in r["process_mean_ms"])
        lines.append(
            f"| {p} | {r['mean_ms']:.3f} ± {r['sample_sd_ms']:.3f} | {means} | {r['readback_mean_ms']:.3f} | 未测 | 未测 | 未测 | {r['gpu_sampled_max_mib']:.1f} |"
        )
    lines += [
        "",
        "均值及SD来自五个进程各自的稳态均值，SD使用ddof=1，不把9450步当独立进程样本。每步用time.perf_counter和所属Warp CUDA设备前后同步：EC sim.update(physics_dt=1/60,step=1)，IsaacLab write_data_to_sim/sim.step/scene.update。读回、正确性检查、日志、显存采样、初始化与序列化不计入步耗时。",
        "",
        "显存为定期采样的进程占用最大值，不是瞬态峰值或增量分配；未测CPU及每次调用的分配增量。所有十次Graph captured、实际时钟/命名关节/TCP/冻结求解策略检查通过；每120步采样未见其他GPU计算进程，该监测并非连续。",
        "",
        "## Success & Other Metrics",
        "",
        "| 平台 | success_rate 本任务完整判据 | 完整有限轨迹 | 原六项判据通过数（逐次） | 最终下摆XY误差mm（逐次） |",
        "|---|---|---|---|---|",
    ]
    for p, r in results.items():
        counts = ", ".join(
            str(sum(s["physical"]["candidate_checks"].values())) + "/6"
            for s in r["samples"]
        )
        errors = ", ".join(
            f"{1000*s['physical']['final_hem_to_release_center_xy_error_m']:.2f}"
            for s in r["samples"]
        )
        lines.append(
            f"| {p} | {r['physical_successes']}/5 | 5/5 | {counts} | {errors} |"
        )
    lines += [
        "",
        "使用原配置判据，未为结果修改阈值。完整物理数组及离线分析全部保留，布形用于任务描述；本轮未重复完整FCL碰撞扫描，移植预览的离散FCL结果见PORTS.md。",
        "",
        "## Leaderboard",
        "",
        "| 任务判据排序 | 平台 | 全判据通过/尝试 |",
        "|---|---|---|",
    ]
    previous = None
    rank = 0
    for index, (p, r) in enumerate(
        sorted(results.items(), key=lambda item: -item[1]["physical_successes"]), 1
    ):
        if previous != r["physical_successes"]:
            rank = index
        previous = r["physical_successes"]
        lines.append(f"| {rank} | {p} | {r['physical_successes']}/5 |")
    lines += [
        "",
        "此表仅按当前任务判据列示全部平台，不推断通用布料精度或物理质量排名。",
        "",
        f"本任务中EC平均完成一步的耗时低{(1-ec/il)*100:.2f}%，逆耗时比为{il/ec:.3f}×。该结论限定当前场景、版本、控制及求解策略；未测异步吞吐和完整应用端到端性能。",
        "",
        f"全部{len(protocol['protected_sha256'])}个原输入、认可媒体、原状态/分析/源码及预试文件哈希保持不变；{len(protocol['source_sha256'])}个执行源码单独冻结。没有SDK改动。原始数据见每侧r01–r05的timing.csv、states.npz、manifest.json和analysis；protocol.json冻结口径，attempts.json保留进程顺序和完整命令。",
        "",
        "本次只涉及tech_reports实验脚本，无生产契约或MAP匹配主题变化，项目上下文无需修改。",
        "",
    ]
    (ROOT / "REPORT.md").write_text("\n".join(lines))
    print(
        json.dumps(
            {
                p: {k: v for k, v in r.items() if k != "samples"}
                for p, r in results.items()
            },
            indent=2,
        )
    )
    print("Markdown report saved:", ROOT / "REPORT.md")


def main() -> None:
    """Select one explicit preparation, batch execution or report-only action."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        _prepare()
    elif args.run:
        _run()
    else:
        _report()


if __name__ == "__main__":
    main()
