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
"""Validate completed M04 platform recordings and build their comparison.

Run: .venv/bin/python scripts/benchmark/tech_reports/physics/experiments/dexuni/m04_bimanual_tshirt_single_fold/tools/validate_ports.py
Reads approved inputs and raw states; native artifacts remain unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from typing import Any
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[8]))
from scripts.benchmark.tech_reports.physics.experiments.dexuni.m04_common import (
    OUTPUT,
    digest,
    write_json,
)

__all__ = ["main"]
PLATFORMS = ("embodichain", "isaaclab")
GEOMETRY_PYTHON = "/home/oem/code/engine/dexsim/.env-3.12/bin/python"


def _expanded_links(
    states: dict[str, np.ndarray], manifest: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Expand only fixed imported links from measured parent/child transforms."""
    names = manifest["link_pose_order"]
    transforms = dict(zip(names, np.moveaxis(states["link_poses"], 1, 0), strict=True))
    urdf = ET.parse(OUTPUT / "inputs/w1_v030/robot.urdf").getroot()
    all_names = [link.get("name") for link in urdf.findall("link")]
    if set(all_names) == set(names):
        return states, manifest

    fixed = []
    for joint in urdf.findall("joint"):
        if joint.get("type") != "fixed":
            continue
        origin = joint.find("origin")
        matrix = np.eye(4)
        if origin is not None:
            matrix[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            matrix[:3, :3] = Rotation.from_euler(
                "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            ).as_matrix()
        fixed.append(
            (joint.find("parent").get("link"), joint.find("child").get("link"), matrix)
        )
    while set(all_names) - set(transforms):
        before = len(transforms)
        for parent, child, matrix in fixed:
            if parent in transforms and child not in transforms:
                transforms[child] = transforms[parent] @ matrix
            if child in transforms and parent not in transforms:
                transforms[parent] = transforms[child] @ np.linalg.inv(matrix)
        if before == len(transforms):
            raise ValueError("Missing non-fixed measured link poses")
    manifest = dict(
        manifest,
        link_pose_order=all_names,
        fixed_link_expansion_from_recorded_links=True,
        originally_measured_links=names,
    )
    states = dict(
        states, link_poses=np.stack([transforms[n] for n in all_names], axis=1)
    )
    return states, manifest


def _analyze(platform: str) -> dict[str, Any]:
    preview = OUTPUT / "preview" / platform
    analysis = OUTPUT / "analysis/ports" / platform
    analysis.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((preview / "manifest.json").read_text())
    assert manifest["status"] == "completed" and manifest["outer_steps"] == 1920
    assert (
        manifest["video_frames"] in (840, 841)
        and manifest["effective"]["cuda_graph_status"] == "captured"
    )
    state_hash = digest(preview / "states.npz")
    with np.load(preview / "states.npz") as data:
        states = {k: data[k] for k in data.files}
    assert states["time"].shape == (1921,) and abs(states["time"][-1] - 32) < 1e-7
    if platform == "isaaclab":
        assert manifest["effective"]["actual_substeps"] == 19200
    with tempfile.TemporaryDirectory(prefix=f"m04_{platform}_analysis_") as tmp:
        root = Path(tmp)
        (root / "inputs").symlink_to(OUTPUT / "inputs", target_is_directory=True)
        (root / "preview").mkdir()
        # Imported fixed links can be welded; retain measured raw states unchanged.
        expanded, geometric_manifest = _expanded_links(states, manifest)
        geometric_manifest["source_states_sha256"] = state_hash
        write_json(root / "manifest.json", geometric_manifest)
        if len(manifest["link_pose_order"]) == len(
            geometric_manifest["link_pose_order"]
        ):
            (root / "preview/states.npz").symlink_to(preview / "states.npz")
        else:
            np.savez_compressed(root / "preview/states.npz", **expanded)
        tool = Path(__file__).parent
        with (analysis / "physical_checks.log").open("w") as log:
            subprocess.run(
                [sys.executable, str(tool / "analyze.py"), "--run", str(root)],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        for p in (root / "analysis").iterdir():
            shutil.copy2(p, analysis / p.name)
        clearance = analysis / "actual_clearance.json"
        previous = json.loads(clearance.read_text()) if clearance.exists() else {}
        if previous.get("source_states_sha256") != state_hash:
            with (analysis / "clearance.log").open("w") as log:
                subprocess.run(
                    [
                        GEOMETRY_PYTHON,
                        str(tool / "check_clearance.py"),
                        "--run",
                        str(root),
                        "--out",
                        str(clearance),
                        "--stride",
                        "5",
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            d = json.loads(clearance.read_text())
            d.update(
                source_states_sha256=state_hash,
                measured_link_count=len(manifest["link_pose_order"]),
                fixed_link_expansion_count=len(geometric_manifest["link_pose_order"])
                - len(manifest["link_pose_order"]),
            )
            write_json(clearance, d)
    c = json.loads(clearance.read_text())
    clear = not (
        c["intersections"]
        or c["cloth_robot_check"]["intersections"]
        or c["cloth_support_check"]["intersections"]
    )
    physical = json.loads((analysis / "physical_checks.json").read_text())
    assert (
        physical["finite"]
        and physical["max_actual_named_qpos_error"] < 2e-5
        and physical["max_actual_tcp_error_m"] < 2e-5
    )
    write_json(
        analysis / "state_provenance.json",
        dict(
            raw_states_sha256=state_hash,
            measured_links=manifest["link_pose_order"],
            fixed_link_expansion_count=c["fixed_link_expansion_count"],
            method="Welded fixed links expanded from recorded neighbouring body poses and unchanged URDF fixed transforms for geometry checks only; raw recording untouched.",
        ),
    )
    return dict(
        manifest=manifest,
        physical=physical,
        clearance=clear,
        clearance_samples=c["checked_frames"],
        states_sha256=state_hash,
        video_sha256=digest(preview / "video.mp4"),
    )


def _media() -> dict[str, Any]:
    import imageio.v2 as imageio
    from PIL import Image

    videos = [OUTPUT / "preview" / p / "video.mp4" for p in PLATFORMS]
    checks = {}
    source_frames = {}
    for p, video in zip(PLATFORMS, videos):
        reader = imageio.get_reader(str(video))
        meta = reader.get_meta_data()
        frames = sum(1 for _ in reader)
        reader.close()
        with np.load(OUTPUT / "preview" / p / "states.npz") as states:
            source_frames[p] = np.flatnonzero(
                (np.arange(1921) % 2 == 0) & (states["time"] >= 4.0)
            )
        assert frames == len(source_frames[p]) and abs(meta["fps"] - 30) < 1e-6
        checks[p] = dict(
            frames=frames,
            fps=meta["fps"],
            size=meta["size"],
            sha256=digest(video),
            full_decode_passed=True,
        )
    common = np.intersect1d(source_frames[PLATFORMS[0]], source_frames[PLATFORMS[1]])
    indices = {p: np.searchsorted(source_frames[p], common) for p in PLATFORMS}
    count = len(common)
    assert count >= 840
    output = OUTPUT / "preview/comparison.mp4"
    readers = [imageio.get_reader(str(video)) for video in videos]
    try:
        with imageio.get_writer(
            str(output), fps=30, codec="libx264", quality=8, macro_block_size=2
        ) as writer:
            for frame in range(count):
                views = []
                for reader, platform in zip(readers, PLATFORMS):
                    source = reader.get_data(int(indices[platform][frame]))
                    views.append(
                        np.asarray(
                            Image.fromarray(source).resize(
                                (960, 540), Image.Resampling.LANCZOS
                            )
                        )
                    )
                writer.append_data(np.hstack(views))
    finally:
        for reader in readers:
            reader.close()
    reader = imageio.get_reader(str(output))
    output_meta = reader.get_meta_data()
    frames = sum(1 for _ in reader)
    reader.close()
    assert frames == count and abs(output_meta["fps"] - 30) < 1e-6
    np.savez_compressed(
        OUTPUT / "preview/comparison_frames.npz",
        output_frame=np.arange(count),
        video_time_s=np.arange(count) / 30,
        simulation_time_s=common / 60,
        embodichain_source_frame=indices["embodichain"],
        isaaclab_source_frame=indices["isaaclab"],
    )
    checks["comparison"] = dict(
        frames=frames,
        fps=30,
        size=[1920, 540],
        sha256=digest(output),
        full_decode_passed=True,
        method="intersection of actual recorded 30Hz frame timestamps after settling; only spatial scaling and side-by-side composition; no temporal smoothing or splicing",
        first_physical_time_s=float(common[0] / 60),
        last_physical_time_s=float(common[-1] / 60),
    )
    return checks


def _plot() -> None:
    """Plot both actual recordings and retain the exact source arrays."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    arrays = {}
    for platform in PLATFORMS:
        with np.load(OUTPUT / "analysis/ports" / platform / "plot_data.npz") as data:
            t = data["time"]
            mask = t >= 4
            axes[0, 0].plot(t[mask], data["hem_center"][mask, 0], label=platform)
            axes[0, 1].plot(
                t[mask], (data["edge_length_ratio_p95"][mask] - 1) * 100, label=platform
            )
            axes[1, 0].plot(t[mask], data["rms_speed"][mask] * 1000, label=platform)
            axes[1, 1].plot(
                t[mask], data["highpass_rms_m"][mask] * 1000, label=platform
            )
            arrays.update({f"{platform}_{key}": data[key] for key in data.files})
    labels = (
        "Hem centre X (m)",
        "Edge length p95 extension (%)",
        "Node RMS speed (mm/s)",
        "Position RMS above 5 Hz (mm)",
    )
    for ax, label in zip(axes.flat, labels):
        ax.set(xlabel="Actual simulation time (s)", ylabel=label)
        ax.axvspan(26, 28, color="grey", alpha=0.12)
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "analysis/ports/physical_comparison.png", dpi=150)
    plt.close(fig)
    np.savez_compressed(OUTPUT / "analysis/ports/physical_comparison.npz", **arrays)


def main() -> None:
    """Check both complete recordings and publish one honest scene comparison."""
    result = {p: _analyze(p) for p in PLATFORMS}
    media = _media()
    _plot()
    baseline = json.loads(
        (OUTPUT / "analysis/ports/approved_native_hashes.json").read_text()
    )
    assert all(
        digest(OUTPUT / p) == h for p, h in baseline.items()
    ), "Approved native artifacts changed"
    write_json(
        OUTPUT / "analysis/ports/validation.json",
        dict(
            approved_native_artifacts_unchanged=True,
            protected_files=len(baseline),
            platform_results=result,
            media=media,
            performance_measurements_started=False,
        ),
    )
    rows = []
    for platform in PLATFORMS:
        p = result[platform]["physical"]
        rows.append(
            f"| {platform} | {max(p['max_grip_relative_drift_m'])*1000:.2f} | {p['lift_far_end_slide_toward_gripper_m']*1000:.2f} | {(p['max_lift_edge_length_ratio_p95']-1)*100:.2f} | {p['final_hem_to_release_center_xy_error_m']*1000:.2f} | {sum(p['candidate_checks'].values())}/6 |"
        )
    text = (
        """# M04：EmbodiChain / IsaacLab 移植预览

用户已认可原生最终松爪版本并授权两平台移植。本轮保留原生输入、视频、状态、分析和源码快照不变，没有性能测量。

[并排视频](preview/comparison.mp4) · [EmbodiChain](preview/embodichain/video.mp4) · [IsaacLab](preview/isaaclab/video.mp4) · [全部验证数据](analysis/ports/validation.json) · [实际响应曲线](analysis/ports/physical_comparison.png)

两侧共用6436节点/12736面完整T恤、原始W1/Pika资产、固定桌面、24个命名关节轨迹、32秒物理时间、60Hz/10子步和CUDA Graph。0–26秒指令及最终26–28秒张开、1.6秒摩擦平滑过渡、28–29.5秒退让均来自认可输入。密度、膜/边刚度与阻尼、粒子半径、接触材料及8/s空气阻力保持相同；初始化节点坐标、质量、无pin均经核对。浮点变换造成质量相对误差约2.52e-5，两移植侧彼此一致。

EmbodiChain经公开SimulationManager/配置/运行控制接口，使用DexUni surface-fast基础16次及自适应策略、Jacobi polish1、multilevel关闭。IsaacLab经InteractiveScene/NewtonVBDManager，使用上游普通VBD20次；仅适配冻结运动学、分阶段摩擦与相同被动空气阻力，没有替换VBD物理求解器。两者均one_way/kinematic/self-contact。该版本数值行为保留求解策略差异，不据预览推断通用物理精度或性能排名。

原生DexSim版本7cfe364b/Newton1.7.0；EC使用安装的DexSim dfceb00e/Newton1.7.0；IsaacLab28.0.0/isaaclab-newton9.1.4/Newton1.6.1，Warp均1.17.0。SDK未修改。原始URDF和全部源网格哈希匹配后，IsaacLab复用已验证的USD转换；所有机器人碰撞网格请求SDF。IsaacLab导入焊合11个固定link，33个实际body姿态保留在原始状态中，几何检查只通过原URDF固定变换展开到44个link。EC记录44个link。空场景地面为平台默认Z0，未增加工作台高度附近的支撑。

两侧均录制1921状态、1920外步/19200子步，实际32秒；关节和TCP回放逐帧校验，Graph captured。视频从4秒静置后首个被录制的30Hz帧开始（浮点时钟使EC首帧为4.0333秒），完整帧数与映射见验证JSON；同一42度水平视野、相机位置和FabricGray颜色贴图。两渲染器光照/法线及材质实现不同，画面不作物理质量标定；视频未做轨迹平滑或拼接。IsaacLab补齐颜色贴图repeat设置后，使用原始录制状态逐帧重新渲染：没有物理步进，节点/关节/link姿态写入均核对到2e-6以内，原始状态哈希未变，详见preview/isaaclab/render_replay.json。

| 平台 | 最大夹持漂移mm | 提起时远端滑移mm | 提起时边长p95最大伸长% | 最终下摆XY误差mm | 原判据通过数 |
|---|---:|---:|---:|---:|---:|
"""
        + "\n".join(rows)
        + """

各项判据阈值来自原冻结配置，未为移植结果修改。失败项与有效求解器配置见每侧physical_checks.json和manifest.json，布形差异按实际结果保留。

每侧按实际录制状态每5帧采样385个姿态，以FCL检查机器人/支撑、非相邻机器人自碰撞、衣服/非手指机器人和衣服/支撑；有意手指接触除外。结果见[EC](analysis/ports/embodichain/actual_clearance.json)与[IL](analysis/ports/isaaclab/actual_clearance.json)，只描述离散采样，不能证明连续无穿透。目标轨迹沿用原先19201个子步姿态检查。

复现：
```bash
.venv/bin/python scripts/benchmark/tech_reports/physics/platforms/dexuni/embodichain/m04.py
/home/oem/app/IsaacLab/.venv/bin/python scripts/benchmark/tech_reports/physics/platforms/dexuni/isaaclab/m04.py
.venv/bin/python scripts/benchmark/tech_reports/physics/experiments/dexuni/m04_bimanual_tshirt_single_fold/tools/validate_ports.py
```

源码保存在source/ports，批准的原生source目录不变；每侧状态、模型审计、日志和分析数组均保留。当前两平台对比视频待用户查看。

本次仅涉及tech_reports实验脚本，MAP没有匹配的生产契约变更；已复核项目上下文，无需更新，证据见analysis/ports/context_affected.log。
"""
    )
    (OUTPUT / "PORTS.md").write_text(text)
    print(
        json.dumps(
            {
                p: dict(
                    checks=result[p]["physical"]["candidate_checks"],
                    clearance=result[p]["clearance"],
                )
                for p in PLATFORMS
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
