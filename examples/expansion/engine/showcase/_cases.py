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

"""Explicit cases and frozen configuration for the two expert motion sources."""

from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
import yaml

__all__: list[str] = []
HERE = Path(__file__).resolve().parent
ENGINE = HERE.parent


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".yaml":
        content = yaml.safe_dump(value, sort_keys=False, allow_unicode=True)
    else:
        content = (
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def absolute_deployment(path: Path) -> dict:
    from embodichain.utils.config_paths import resolve_config_path

    def resolve(reference):
        candidate = (path.parent / reference).resolve()
        if not candidate.exists():
            marker = "extensions/embodichain_tasks/configs/"
            if marker not in reference:
                raise FileNotFoundError(candidate)
            candidate = resolve_config_path(marker + reference.split(marker, 1)[1])
        return str(candidate)

    value = yaml.safe_load(path.read_text())
    for key in ("environment", "embodiment"):
        if key in value:
            value[key]["component"] = resolve(value[key]["component"])
    for key in ("program", "integration", "execution_policy"):
        if key in value.get("task_program", {}):
            value["task_program"][key] = resolve(value["task_program"][key])
    if "real2sim" in value:
        value["real2sim"]["profile"] = resolve(value["real2sim"]["profile"])

    return value


def build_cases(root: Path, mode: str, *, preserve: bool = False) -> list[dict]:
    """Write a deterministic case inventory; each physical run still uses Engine."""
    from embodichain.utils.config_paths import resolve_config_path

    configs = root / "configs"
    configs.mkdir(exist_ok=True)

    def write(path, value):
        if preserve and path.exists():
            return
        dump(path, value)

    prepared = root / "inputs" / "carton"
    for backend in ("default", "newton"):
        write(
            configs / f"ur5.{backend}.yaml",
            absolute_deployment(ENGINE / f"task.{backend}.yaml"),
        )
    task = absolute_deployment(prepared / "task.measured.yaml")
    task["embodiment"]["component"] = str(
        resolve_config_path(
            "extensions/embodichain_tasks/configs/components/embodiments/franka_panda_robotiq85_rgbd.yaml"
        )
    )
    task["renderer"] = "hybrid"
    task["enable_sensor"] = True
    write(configs / "carton.default.yaml", task)
    newton = deepcopy(task)
    env = yaml.safe_load((prepared / "environment.yaml").read_text())
    env["physics"] = "newton"
    env["physics_config"] = {
        "physics_dt": env.get("physics_config", {}).get("physics_dt", 1.0 / 240.0)
    }
    write(configs / "carton.environment.newton.yaml", env)
    newton["environment"]["component"] = str(configs / "carton.environment.newton.yaml")
    write(configs / "carton.newton.yaml", newton)
    profiles = {}
    for source, uid in (("ur5", "cube"), ("carton", "carton")):
        profiles[source] = {
            "material": {
                "materials": {
                    uid: {
                        "base_color": [0.12, 0.48, 0.90],
                        "roughness": 0.25,
                        "metallic": 0.15,
                        "ior": 1.4,
                    }
                }
            },
            "light": {
                "lights": {"main_light": {"color": [1.0, 0.75, 0.45], "intensity": 3.0}}
            },
            "camera": {
                "cameras": {
                    "cam_overhead": {
                        "position_delta": [0.04, 0.0, 0.02],
                        "focal_scale": 1.08,
                    }
                }
            },
        }
        profiles[source]["combined"] = {
            k: v for p in profiles[source].values() for k, v in p.items()
        }
        profiles[source]["capture"] = {
            "materials": {
                uid: {
                    "base_color": [0.75, 0.2, 0.55],
                    "roughness": 0.65,
                    "metallic": 0.0,
                    "ior": 1.3,
                }
            }
        }
        write(configs / f"visual.{source}.yaml", {"profiles": profiles[source]})
    motion = yaml.safe_load((ENGINE / "motion.yaml").read_text())
    for operator in ("joint_residual", "via_points", "affordance", "combined"):
        m = deepcopy(motion)
        if operator == "affordance":
            m["affordance"].update(enabled=True, branches_per_family=2, max_proposals=8)
            m["trajectory"]["spatial"]["enabled"] = False
            m["trajectory"]["variants_per_family"] = 1
        else:
            m["trajectory"]["spatial"]["operators"] = (
                [operator]
                if operator != "combined"
                else ["joint_residual", "via_points"]
            )
        write(configs / f"motion.{operator}.yaml", m)
    profile = yaml.safe_load((prepared / "profile.measured.single.yaml").read_text())
    recipes = ("nominal", "joint_7", "joint_11", "task_space", "time_115", "time_130")
    for recipe in (*recipes, "combined"):
        p = deepcopy(profile)
        p["recipes"] = list(recipes) if recipe == "combined" else [recipe]
        write(configs / f"prototype.{recipe}.yaml", p)
    cases = []

    def add(
        id,
        title,
        source="ur5",
        backend="default",
        tags=(),
        required=True,
        quick=False,
        **overrides,
    ):
        job = {
            "deployment": str(configs / f"{source}.{backend}.yaml"),
            "motion_source": "native" if source == "ur5" else "prototype",
            "output_dir": str(root / "jobs" / id),
            "batch_size": 2 if mode == "quick" else 8,
            "target_episodes": 2,
            "max_attempts": 8,
            "max_steps": 2000 if source == "ur5" else 2400,
            "max_wall_time_s": 600.0,
            "seed": 7,
            "settling_steps": 5,
            "capture_media": True,
            "media_buffer_bytes": 256 * 1024 * 1024,
        }
        if source == "ur5":
            job["collision_profile"] = str(
                ENGINE.parent / "scene_robot_feasibility/search.ur5.yaml"
            )
        else:
            job["source_profile"] = str(configs / "prototype.nominal.yaml")
        job.update(overrides)
        saved = configs / f"job.{id}.yaml"
        if preserve and saved.exists():
            job = yaml.safe_load(saved.read_text())
        job["output_dir"] = str(root / "jobs" / id)
        dump(saved, job)
        cases.append(
            {
                "id": id,
                "title": title,
                "source": source,
                "backend": backend,
                "tags": list(tags),
                "required": required,
                "selected": mode == "full" or quick,
                "job": f"configs/job.{id}.yaml",
                "target": job["target_episodes"],
                "batch_size": job["batch_size"],
                "timeout_s": int(job["max_wall_time_s"])
                + (
                    900
                    if job.get("render_profiles") or job.get("export_lerobot")
                    else 300
                ),
            }
        )

    add("ur5-baseline", "规划生成 · 基准专家", tags=("baseline",), quick=True)
    add(
        "carton-baseline",
        "真实示范驱动 · 基准专家",
        source="carton",
        tags=("baseline", "prototype"),
        quick=True,
    )
    add(
        "ur5-scene",
        "场景 × 机器人安装位置",
        tags=("scene", "workspace"),
        scene_atlas=str(root / "inputs/atlas/atlas.json"),
        target_episodes=4,
        max_attempts=16,
    )
    for op, title in (
        ("joint_residual", "关节残差"),
        ("via_points", "途经点"),
        ("affordance", "抓取分支"),
    ):
        add(
            "ur5-" + op,
            title,
            tags=("motion", op),
            motion_profile=str(configs / f"motion.{op}.yaml"),
        )
    physics = {
        "mass": {"mass_scale": {"cube": [0.98, 1.02]}},
        "friction": {"friction": {"cube": [0.95, 1.0]}},
        "com": {
            "center_of_mass": {"cube": [[-0.001, 0.001], [-0.001, 0.001], [0.0, 0.0]]}
        },
        "stiffness": {
            "joint_stiffness_scale": {"gripper_finger1_joint_1": [0.98, 1.02]}
        },
        "damping": {"joint_damping_scale": {"gripper_finger1_joint_1": [0.98, 1.02]}},
    }
    for backend in ("default", "newton"):
        if backend == "newton":
            add(
                "newton-baseline",
                "Newton · 基准专家",
                backend=backend,
                tags=("baseline", "backend"),
            )
        for key, values in physics.items():
            add(
                f"{backend}-{key}",
                f"{backend} · {key}",
                backend=backend,
                tags=("physics", key),
                **values,
            )
        add(
            f"{backend}-visual",
            "原生视觉采集与观测重渲染",
            backend=backend,
            tags=("visual", "rerender", "export"),
            visual_profile_file=str(configs / "visual.ur5.yaml"),
            visual_profile="capture",
            render_profiles=["material", "light", "camera", "combined"],
            observation_independent=True,
            export_lerobot=True,
        )
        add(
            f"{backend}-rerender-controls",
            "规范基线 · 视觉单项对照",
            backend=backend,
            tags=("visual", "rerender", "controlled_visual"),
            visual_profile_file=str(configs / "visual.ur5.yaml"),
            render_profiles=["material", "light", "camera"],
            observation_independent=True,
            target_episodes=1,
        )
        params = {k: v for p in physics.values() for k, v in p.items()}
        add(
            f"{backend}-combined",
            "多领域组合 · " + backend,
            backend=backend,
            tags=("combined", "scene", "motion", "physics", "visual", "rerender"),
            quick=backend == "default",
            scene_atlas=str(root / "inputs/atlas/atlas.json"),
            motion_profile=str(configs / "motion.combined.yaml"),
            visual_profile_file=str(configs / "visual.ur5.yaml"),
            render_profiles=["material", "light", "camera", "combined"],
            observation_independent=True,
            target_episodes=2 if mode == "quick" else 8,
            max_attempts=32,
            max_wall_time_s=1200.0,
            **params,
        )
    for recipe in recipes[1:]:
        add(
            "carton-" + recipe,
            "真实示范 · " + recipe,
            source="carton",
            tags=(
                "prototype",
                "motion",
                "retiming" if recipe.startswith("time") else "spatial",
            ),
            source_profile=str(configs / f"prototype.{recipe}.yaml"),
        )
    add(
        "carton-combined",
        "真实示范 · 运动／物理／视觉组合",
        source="carton",
        tags=("prototype", "combined", "physics", "visual", "rerender", "export"),
        quick=True,
        source_profile=str(configs / "prototype.combined.yaml"),
        mass_scale={"carton": [0.99, 1.01]},
        visual_profile_file=str(configs / "visual.carton.yaml"),
        render_profiles=["material", "camera"],
        observation_independent=True,
        export_lerobot=True,
        target_episodes=2 if mode == "quick" else 8,
        max_attempts=32,
        max_wall_time_s=1200.0,
    )
    add(
        "newton-prototype",
        "Newton 原型来源 · 资格诊断",
        source="carton",
        backend="newton",
        tags=("prototype", "diagnostic"),
        required=False,
        max_attempts=2,
        capture_media=False,
    )
    for batch in (1, 8, 32):
        add(
            f"batch-{batch}",
            f"同 World B={batch} 与尾批",
            tags=("batch",),
            batch_size=batch,
            target_episodes=batch + 1,
            max_attempts=2 * (batch + 1),
            capture_media=False,
            mass_scale={"cube": [0.98, 1.02]},
            max_wall_time_s=1200.0,
        )
    # Preserve separately qualified extensions such as wider-range experiments.
    inventory = root / "cases.json"
    if preserve and inventory.exists():
        known = {case["id"] for case in cases}
        cases.extend(
            case
            for case in json.loads(inventory.read_text())["cases"]
            if case["id"] not in known
        )
    dump(inventory, {"schema_version": 1, "mode": mode, "cases": cases})
    return cases


def recover_portability(root: Path) -> None:
    """Restore the two owned renames if an audit process was interrupted."""
    intent = root / "diagnostics/portability-move.json"
    if not intent.exists():
        return
    record = json.loads(intent.read_text())
    paths = {}
    for key in ("job", "dataset", "destination", "hidden"):
        relative = Path(record[key])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid relocation recovery path")
        paths[key] = root / relative
    if paths["hidden"].exists():
        if paths["job"].exists():
            raise RuntimeError("both visible and hidden job directories exist")
        paths["hidden"].rename(paths["job"])
    if paths["destination"].exists():
        if paths["dataset"].exists():
            raise RuntimeError("both relocated and original datasets exist")
        paths["destination"].rename(paths["dataset"])
    if not paths["job"].exists() or not paths["dataset"].exists():
        raise RuntimeError("relocation audit lost an owned directory")
    intent.unlink()


def worker_environment() -> dict[str, str]:
    """Keep wheel-installed example data importable outside the repository cwd."""
    import os

    parents = [str(HERE.parents[3])]
    if os.environ.get("PYTHONPATH"):
        parents.append(os.environ["PYTHONPATH"])
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(parents),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "EMBODICHAIN_SIM_EXIT_PROCESS": "0",
    }


def stop_process_tree(process, *, force: bool = False) -> None:
    """Reap a subprocess and its owned descendants on diagnostic cancellation."""
    import psutil

    try:
        parent = psutil.Process(process.pid)
        targets = parent.children(recursive=True) + [parent]
    except psutil.NoSuchProcess:
        targets = []
    for target in targets:
        try:
            target.kill() if force else target.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(targets, timeout=3)
    for target in alive:
        try:
            target.kill()
        except psutil.NoSuchProcess:
            pass
    process.wait(timeout=5)


class ProcessMetrics:
    """Sample the current worker's process tree without retaining frame buffers."""

    def __init__(self, root: Path, name: str) -> None:
        import os
        import threading
        import time

        self.pid = os.getpid()
        self.started = time.monotonic()
        self.path = root / "telemetry" / name / (str(time.time_ns()) + ".json")
        self.values = {
            "pid": self.pid,
            "sampled_peak_rss_bytes": 0,
            "sampled_peak_gpu_bytes": None,
            "sample_period_s": 1.0,
            "gpu_measurement": "unavailable",
        }
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def _sample(self) -> None:
        import time
        import psutil

        nvml = None
        try:
            import pynvml

            pynvml.nvmlInit()
            nvml = pynvml
            devices = [
                nvml.nvmlDeviceGetHandleByIndex(i)
                for i in range(nvml.nvmlDeviceGetCount())
            ]
            self.values["gpu_measurement"] = "NVML process memory, sampled"
        except Exception as error:
            self.values["gpu_measurement"] = str(error)
        try:
            while not self.stop_event.is_set():
                try:
                    process = psutil.Process(self.pid)
                    family = [process, *process.children(recursive=True)]
                    rss = sum(p.memory_info().rss for p in family if p.is_running())
                    self.values["sampled_peak_rss_bytes"] = max(
                        self.values["sampled_peak_rss_bytes"], rss
                    )
                    if nvml is not None:
                        pids = {p.pid for p in family}
                        total = 0
                        for device in devices:
                            memory = {}
                            for getter in (
                                nvml.nvmlDeviceGetComputeRunningProcesses,
                                nvml.nvmlDeviceGetGraphicsRunningProcesses,
                            ):
                                for item in getter(device):
                                    if (
                                        item.pid in pids
                                        and 0 <= item.usedGpuMemory < 2**63
                                    ):
                                        memory[item.pid] = max(
                                            memory.get(item.pid, 0),
                                            int(item.usedGpuMemory),
                                        )
                            total += sum(memory.values())
                        self.values["sampled_peak_gpu_bytes"] = max(
                            self.values["sampled_peak_gpu_bytes"] or 0, total
                        )
                    dump(
                        self.path,
                        {
                            **self.values,
                            "wall_time_s": time.monotonic() - self.started,
                            "state": "running",
                        },
                    )
                except psutil.Error:
                    pass
                except Exception as error:
                    self.values["measurement_error"] = str(error)
                self.stop_event.wait(1.0)
        finally:
            if nvml is not None:
                nvml.nvmlShutdown()

    def close(self) -> dict:
        import time

        self.stop_event.set()
        self.thread.join(timeout=5)
        result = {
            **self.values,
            "wall_time_s": time.monotonic() - self.started,
            "state": "closed",
        }
        if not self.thread.is_alive():
            dump(self.path, result)
        return result


def ensure_prepared(destination: Path, prepare, validate) -> dict:
    """Validate unmarked preparation before reuse; preserve damaged qualified inputs."""
    import time

    marker = destination / "showcase-prepared.json"
    if marker.exists():
        qualification = validate()
        pinned = json.loads(marker.read_text())
        if pinned["source_revision"] != qualification["source_revision"]:
            raise ValueError("qualified source revision changed")
        return qualification
    try:
        qualification = validate()
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError):
        if destination.exists():
            destination.rename(
                destination.with_name(
                    destination.name + ".interrupted-" + str(time.time_ns())
                )
            )
        prepare()
        qualification = validate()
    dump(
        marker, {"complete": True, "source_revision": qualification["source_revision"]}
    )
    return qualification
