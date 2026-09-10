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

"""Budgeted local coding sessions and fresh-process candidate replay."""

from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import replace
from collections.abc import Callable
import json
import hashlib
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
from uuid import uuid4

import psutil

from .catalog import Task, write_json

__all__ = ["create_run", "execute_script", "solve"]

_DEFAULT_MODEL = "gpt-6-astra"
_DEFAULT_REASONING_EFFORT = "xhigh"


def _run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6]


def _environment(repo: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(repo), env.get("PYTHONPATH", "")]).rstrip(
        os.pathsep
    )
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _process(
    command: list[str],
    *,
    cwd: Path,
    output: Path,
    timeout: float,
    env: dict[str, str],
    prompt: str | None = None,
    service: Callable[[], None] | None = None,
    interactive: bool = False,
) -> dict:
    started = time.monotonic()
    started_at = time.time()
    timed_out = False
    interrupted = False
    with (
        (output / "stdout.log").open("w") as stdout,
        (output / "stderr.log").open("w") as stderr,
    ):
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=(
                None
                if interactive
                else subprocess.PIPE if prompt is not None else subprocess.DEVNULL
            ),
            stdout=None if interactive else stdout,
            stderr=None if interactive else stderr,
            start_new_session=not interactive,
            text=True,
        )
        identity = {
            "pid": process.pid,
            "created_at": psutil.Process(process.pid).create_time(),
            "command": command,
            "started_at": started_at,
        }
        write_json(output / "process.json", identity)
        try:
            if service is None:
                process.communicate(input=prompt, timeout=timeout)
            else:
                if not interactive:
                    process.stdin.write(prompt or "")
                    process.stdin.close()
                while process.poll() is None:
                    if time.monotonic() - started >= timeout:
                        raise subprocess.TimeoutExpired(command, timeout)
                    service()
                    time.sleep(0.2)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            timed_out = isinstance(exc, subprocess.TimeoutExpired)
            interrupted = not timed_out
            _stop_tree(process)
    elapsed = time.monotonic() - started
    record = {
        **identity,
        "returncode": process.returncode,
        "timed_out": timed_out,
        "interrupted": interrupted,
        "wall_seconds": elapsed,
        "ended_at": started_at + elapsed,
    }
    write_json(output / "process.json", record)
    if interrupted:
        raise KeyboardInterrupt
    return record


def _stop_tree(process: subprocess.Popen) -> None:
    # Solver workers create their own sessions. A process-group signal alone
    # would leave those GPU processes alive after a Codex search deadline.
    parent = psutil.Process(process.pid)
    children = parent.children(recursive=True)
    family = children + [parent]
    for child in family:
        try:
            if "ffmpeg" not in child.name():
                child.send_signal(signal.SIGTERM)
        except psutil.NoSuchProcess:
            pass
    # Only Popen may reap its direct child, otherwise wait() can lose its status.
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    _, alive = psutil.wait_procs(children, timeout=5)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass


def create_run(
    task: Task,
    *,
    repo: Path,
    output_root: Path,
    objective: str = "solve",
    robot_component: Path | None = None,
    instruction: str | None = None,
    reuse_policy: str | None = None,
) -> Path:
    """Create a workspace, optionally replacing the goal on unchanged source assets.

    An instruction override also replaces acceptance with the user's new goal
    and clears inherited difficulty. The source task remains provenance only.
    """
    source_task = task.to_dict()
    reuse_policy = reuse_policy or (
        "isolated" if objective == "cold_start" else "research"
    )
    if reuse_policy not in {"research", "isolated"}:
        raise ValueError("reuse_policy must be research or isolated")
    if instruction is not None:
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(
                "instruction must be non-empty; omit it to keep the original task"
            )
        task = replace(task, instruction=instruction, acceptance=instruction, level="")
    root = output_root.resolve() / task.task_id / _run_id()
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    subprocess.run(["git", "init", "--quiet", str(workspace)], check=True)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    write_json(
        root / "run.json",
        {
            "task": task.to_dict(),
            "task_variant": (
                "custom_instruction" if instruction is not None else "default"
            ),
            "acceptance_source": "instruction" if instruction is not None else "task",
            "source_task": source_task if instruction is not None else None,
            "repo": str(repo.resolve()),
            "repo_revision": revision,
            "python": sys.executable,
            "codex_version": subprocess.check_output(
                ["codex", "--version"], text=True
            ).strip(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mode": "B",
            "objective": objective,
            "reuse_policy": reuse_policy,
        },
    )
    (root / "source.diff").write_bytes(
        subprocess.check_output(["git", "diff", "HEAD"], cwd=repo)
    )
    write_json(
        workspace / "lab.json",
        {
            "scene": task.scene_config or task.source_dir,
            "robot_component": str(
                robot_component
                or repo
                / "embodichain_tasks/configs/components/embodiments"
                / "dual_franka_robotiq_arg2f_140.yaml"
            ),
            "seed": 0,
        },
    )
    from ._workspace import _write_workspace

    _write_workspace(root, json.loads((root / "run.json").read_text()))
    return root


def execute_script(
    root: Path,
    script: Path,
    config: Path | None,
    *,
    timeout: float = 300,
    label: str = "attempt",
    request_path: Path | None = None,
) -> dict:
    """Snapshot and run one candidate, retaining failed and timed-out attempts."""
    if os.environ.get("GENSIM_LAB_BROKER") == str(root):
        return _request_execution(root, script, config, timeout=timeout)
    manifest = json.loads((root / "run.json").read_text())
    attempt = root / "attempts" / f"{label}_{_run_id()}"
    attempt.mkdir(parents=True)
    write_json(
        attempt / "experiment.json",
        {
            "mode": manifest.get("mode", "B"),
            "approved_asset_changes": manifest.get("approved_asset_changes", []),
        },
    )
    if request_path is not None:
        write_json(request_path.with_suffix(".active.json"), {"attempt": str(attempt)})
    runtime = attempt / "runtime_source"
    shutil.copytree(
        Path(__file__).parent,
        runtime,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    write_json(
        attempt / "runtime_hashes.json",
        {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in runtime.glob("*.py")
        },
    )
    code = attempt / "code"

    def ignore_outputs(directory: str, names: list[str]) -> list[str]:
        if Path(directory).resolve() in {
            root / "attempts",
            root / "codex",
            root / "requests",
        }:
            return names
        return [name for name in names if name in {".git", "__pycache__"}]

    shutil.copytree(script.resolve().parent, code, ignore=ignore_outputs)
    frozen_script = code / script.name
    command = [
        manifest["python"],
        "-B",
        "-m",
        "embodichain.gen_sim.agent_lab.worker",
        "--script",
        str(frozen_script),
        "--output",
        str(attempt),
    ]
    if config is not None:
        frozen_config = attempt / "lab.json"
        shutil.copy2(config, frozen_config)
        command += ["--config", str(frozen_config)]
    from .delivery import _hash

    input_files = [path for path in code.rglob("*") if path.is_file()]
    if config is not None:
        input_files.append(frozen_config)
    write_json(
        attempt / "inputs.json",
        {
            "script": str(frozen_script.relative_to(attempt)),
            "files_sha256": {
                str(path.relative_to(attempt)): _hash(path)
                for path in sorted(input_files)
            },
        },
    )
    env = _environment(Path(manifest["repo"]))
    env["PYTHONPATH"] = str(code) + os.pathsep + env["PYTHONPATH"]
    process = _process(command, cwd=code, output=attempt, timeout=timeout, env=env)
    result_path = attempt / "result.json"
    worker = (
        json.loads(result_path.read_text())
        if result_path.exists()
        else {
            "execution_completed": False,
            "task_success": None,
            "error": "Worker ended without result.json; inspect native stdout/stderr.",
        }
    )
    metrics_path = attempt / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    report = {
        "attempt": str(attempt),
        "process": process,
        "worker": worker,
        "metrics": metrics,
    }
    write_json(root / "latest_attempt.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def _request_execution(
    root: Path, script: Path, config: Path | None, *, timeout: float
) -> dict:
    # Only simulation runs leave the coding sandbox; ordinary shell/code tools
    # remain there. The host executes the same unrestricted Python solve(lab).
    queue = root / "requests"
    queue.mkdir(exist_ok=True)
    request = queue / f"{_run_id()}.json"
    reply = request.with_suffix(".reply.json")
    staging = request.with_suffix(".tmp")
    write_json(
        staging,
        {
            "script": str(script),
            "config": str(config) if config else None,
            "timeout": timeout,
        },
    )
    staging.replace(request)
    deadline = min(
        time.time() + timeout + 30, float(os.environ["GENSIM_LAB_DEADLINE"]) + 20
    )
    while not reply.exists():
        if time.time() >= deadline:
            raise TimeoutError(f"Host simulation request did not finish: {request}")
        time.sleep(0.2)
    result = json.loads(reply.read_text())
    if "host_error" in result:
        raise RuntimeError(result["host_error"])
    notes = root / "host_notes.md"
    if notes.is_file():
        result["host_notes"] = notes.read_text()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def _serve_requests(root: Path, deadline: float, episode_host=None) -> None:
    for request in sorted((root / "requests").glob("*.json")):
        if request.name.endswith((".reply.json", ".active.json")):
            continue
        reply = request.with_suffix(".reply.json")
        if reply.exists():
            continue
        active = request.with_suffix(".active.json")
        if active.exists():
            active_record = json.loads(active.read_text())
            if "episode_operation" in active_record:
                write_json(
                    reply,
                    {
                        "host_error": "Interrupted episode request; no state or command was automatically replayed",
                        **active_record,
                    },
                )
                continue
            attempt = Path(active_record["attempt"])
            process_path = attempt / "process.json"
            process = (
                json.loads(process_path.read_text()) if process_path.exists() else {}
            )
            if process.get("pid") is not None:
                try:
                    live = psutil.Process(process["pid"])
                    if (
                        live.create_time() == process["created_at"]
                        and live.status() != psutil.STATUS_ZOMBIE
                    ):
                        continue
                except psutil.NoSuchProcess:
                    pass
            result_path = attempt / "result.json"
            worker = (
                json.loads(result_path.read_text())
                if result_path.exists()
                else {
                    "execution_completed": False,
                    "error": "Interrupted host run has no worker result; artifacts retained.",
                }
            )
            process.setdefault("returncode", None)
            process["recovered"] = True
            write_json(
                reply, {"attempt": str(attempt), "process": process, "worker": worker}
            )
            continue
        job = json.loads(request.read_text())
        try:
            if "episode_operation" in job:
                if episode_host is None:
                    raise RuntimeError(
                        "Persistent episodes require the launch/pipeline host"
                    )
                write_json(
                    active,
                    {
                        "episode_operation": job["episode_operation"],
                        "attempt": (
                            str(episode_host.attempt) if episode_host.attempt else None
                        ),
                    },
                )
                result = episode_host.service(job)
            else:
                if (
                    episode_host is not None
                    and episode_host.process is not None
                    and episode_host.process.poll() is None
                ):
                    raise RuntimeError(
                        "Close the persistent episode before starting an independent run"
                    )
                result = execute_script(
                    root,
                    Path(job["script"]),
                    Path(job["config"]) if job.get("config") else None,
                    timeout=max(1, min(job["timeout"], deadline - time.time())),
                    request_path=request,
                )
        except Exception as exc:
            result = {"host_error": f"{type(exc).__name__}: {exc}"}
        staging = reply.with_suffix(".tmp")
        write_json(staging, result)
        staging.replace(reply)


def _prompt(root: Path, manifest: dict, deadline: float) -> str:
    task = manifest["task"]
    mode = manifest.get("mode", "B")
    approved_changes = json.dumps(
        manifest.get("approved_asset_changes", []), ensure_ascii=False
    )
    python = shlex.quote(manifest["python"])
    run_command = (
        f"GENSIM_LAB_BROKER={shlex.quote(str(root))} "
        f"GENSIM_LAB_DEADLINE={deadline:.0f} "
        f"{python} -B -m embodichain.gen_sim.agent_lab run "
        f"--run-dir {shlex.quote(str(root))} --script solution.py --config lab.json"
    )
    return f"""Complete this real robot simulation task as an autonomous coding researcher.

Task: {task['instruction']}
Acceptance: {task['acceptance']}
Difficulty: {task.get('level', 'unspecified')}
Task variant: {manifest.get('task_variant', 'default')}
Acceptance source: {manifest.get('acceptance_source', 'task')}
Source assets: {task['source_dir']}
Reference image: {task['image']}
Asset preparation state: {task['asset_status']}
Experiment mode: {mode}
Explicit user-approved asset changes: {approved_changes}
Read the current project at {manifest['repo']}; start with agent_context/MAP.yaml
and gen_sim/agent_lab/README.md. Read relevant implementations and examples, not
the entire repository mechanically. This is open-book engineering, not blind testing.

Agent Lab owns launch, execution, recording, accounting and delivery only.
You own the task method and its implementation. Reuse existing project APIs;
keep controllers and temporary compatibility adapters in this experiment's
workspace or copy. Shared-library fixes require a separate review, not an
automatic expansion of the experiment launcher.

Use ANY useful approach: existing Atomic Skills, Task Program, MotionGenerator,
direct IK, custom joint trajectories, feedback control, repeated grasp searches,
controller tuning, or your own Python code. Do not wait to implement a general
framework. Do not stop after proposing a plan. Run experiments, examine images
and telemetry, change your hypothesis, and try again. Failed IK candidates are
normal. You may change robot placement before execution. There is no action DSL.

Inspect the reference and actual scene before choosing a method. Record observed
scene suitability and blockers in scene_review.json. Derive goals from the task
and observations, not answer-hint columns or personal memory.
Reuse policy: {manifest.get('reuse_policy', 'isolated')}. Research permits qualified
general adapters and asset calibration with reuse_log.json provenance, fingerprints
and current-scene revalidation. Isolated forbids other experiment outputs and old
same-task answers. Shared project infrastructure is allowed under either policy.
Preserve every stage of the task's acceptance; partial progress is not completion.
For a custom_instruction task, the supplied instruction is the goal and acceptance
basis. Record measurable checks in task_interpretation.json before commanded
control. Do not reuse the source task's acceptance or difficulty, or weaken the
user's goal after a failed attempt. Source-task metadata is provenance only.

Your writable code workspace is {root / 'workspace'}. All run artifacts belong
under {root}. Original project code and assets are reference inputs: experiment
with local copies or overrides, never modify originals, credentials, user settings,
or services. Do not read credential files. Do not install packages globally.
Physics policy: real contact-driven object motion, original object mass/friction,
gravity and collisions. No weld/attachment cheats, object teleports, setting an
object's joint target to solve the task, state resets inside the final episode,
or relaxing acceptance to claim success. Only the explicitly approved asset
changes listed above are authorized. Any further source asset repair requires
the user's approval; document asset blockers instead of silently changing them.
Preserve task-relevant initial states and objects. Native project assumptions
about grasp transforms are planning data, not physical attachment permission.

Use the active embodichain040 Python: {manifest['python']}.
Prefer an ordinary solution.py with def solve(lab). lab.sim, lab.robot,
lab.objects, lab.articulations expose the actual APIs. Helpers are optional.
lab.step(), move_joints(), move_tcp(), capture(), event(), snapshot() are available.
The recorder observes sim.update and saves a streaming video, images, telemetry,
physical setup and errors for every attempt, including failures.
Run: {run_command}
For long composite episodes, pass --timeout 900 to this run command when needed;
the host still enforces the remaining search budget. Do not truncate the task
to fit a short probe timeout.
This run command requests execution from the host GPU supervisor. Run it exactly
as shown: the Codex shell itself has no GPU access. Do not launch worker.py
directly, and do not switch to CPU physics to work around the shell sandbox.
For a genuine native GPU backend failure in the HOST, a documented CPU PhysX
compatibility experiment is allowed: preserve contact dynamics, assets and task
state, and record the backend difference. This is not permission to disable physics.
Modify lab.json to configure robot_overrides, sim, camera_eye/camera_target,
control_dt, or seed. Use repo source to understand frame conventions and limits.
Record observations and named measured milestones. Your return value is a solver
claim, not authoritative task success. Inspect the actual latest.png and event
frames after trials. Do not infer motion from target qpos or exit code alone.

Hard search deadline (Unix seconds): {deadline:.0f}. Check the current time.
Reserve the last two minutes for a clean successful full episode and summary.
Read {root / 'usage.json'} for host-measured cumulative time and reported token
usage. State its scope, as_of and completeness in your handoff. Never estimate
total tokens from response length, add cached/reasoning subsets again, or count
only the final successful attempt. Do not edit usage.json. The host refreshes
the final totals after you stop; your own copy is only a progress snapshot.
When you have a reproducible candidate, write candidate.json with keys script,
config, rationale (paths relative to this workspace). Otherwise preserve your
best attempt and write a concrete failure diagnosis. Keep all failed videos.
Write brief experiment notes to notes.md so subsequent turns can resume.
"""


def _resolve_agent_settings(
    root: Path, model: str | None, effort: str | None
) -> tuple[str, str]:
    """Resolve each omitted value independently, without changing global config."""
    saved = {}
    if model is None or effort is None:
        paths = list((root / "codex").glob("*/agent_settings.json")) + list(
            (root / "codex").glob("*/launch.json")
        )

        def started(path: Path) -> float:
            process = path.parent / "process.json"
            record = json.loads(process.read_text()) if process.is_file() else {}
            return record.get(
                "started_at", record.get("created_at", path.stat().st_mtime)
            )

        if paths:
            saved = json.loads(max(paths, key=started).read_text())
        else:
            records = [
                p for p in (root / "search.json", root / "launch.json") if p.is_file()
            ]
            if records:
                saved = json.loads(
                    max(records, key=lambda p: p.stat().st_mtime).read_text()
                )
    model = model if model is not None else saved.get("model") or _DEFAULT_MODEL
    effort = (
        effort
        if effort is not None
        else saved.get("reasoning_effort") or _DEFAULT_REASONING_EFFORT
    )
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty model identifier")
    if not isinstance(effort, str) or not effort.strip():
        raise ValueError(
            "reasoning-effort must be a non-empty value supported by the Codex backend"
        )
    return model, effort


def solve(
    root: Path,
    *,
    minutes: float,
    model: str | None = None,
    reasoning_effort: str | None = None,
    sandbox: str = "workspace-write",
) -> dict:
    """Let Codex iterate freely, then replay its candidate outside the agent.

    Omitted model/effort values inherit the most recent recorded invocation
    independently; runs without prior settings use the project defaults.
    """
    from .delivery import finalize_run
    from .usage import UsageMeter

    model, reasoning_effort = _resolve_agent_settings(root, model, reasoning_effort)
    execution = {
        "entrypoint": "solve",
        "status": "running",
        "pid": os.getpid(),
        "created_at": psutil.Process().create_time(),
        "model": model or _DEFAULT_MODEL,
        "reasoning_effort": reasoning_effort,
    }
    write_json(root / "search.json", execution)
    meter = UsageMeter(root, "solve")
    try:
        return _solve(
            root,
            minutes=minutes,
            model=model,
            reasoning_effort=reasoning_effort,
            sandbox=sandbox,
            execution=execution,
        )
    except KeyboardInterrupt:
        execution["status"] = "interrupted"
        raise
    except Exception as exc:
        execution.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if execution["status"] == "running":
            execution["status"] = "completed"
        write_json(root / "search.json", execution)
        try:
            finalize_run(root, usage_meter=meter)
        finally:
            meter.close()
        print(f"[Agent Lab] Final report: {root / 'final/report.md'}", flush=True)


def _solve(
    root: Path,
    *,
    minutes: float,
    model: str | None,
    reasoning_effort: str,
    sandbox: str,
    execution: dict,
) -> dict:
    model = model or _DEFAULT_MODEL
    manifest = json.loads((root / "run.json").read_text())
    total_deadline = time.time() + minutes * 60
    deadline = total_deadline - min(300, minutes * 60 * 0.15)
    prompt = _prompt(root, manifest, deadline)
    (root / "prompt.txt").write_text(prompt)
    session_path = root / "session.json"
    previous = json.loads(session_path.read_text()) if session_path.exists() else {}
    thread_id = previous.get("thread_id")
    turn = previous.get("turns", 0)
    existing_turns = sorted((root / "codex").glob("turn_*"), reverse=True)
    if existing_turns:
        turn = max(turn, int(existing_turns[0].name.removeprefix("turn_")))
    if thread_id is None:
        # An interrupted parent may not have written session.json yet; the CLI's
        # thread.started event is already durable and identifies the same session.
        for directory in existing_turns:
            log = directory / "stdout.log"
            if not log.exists():
                continue
            for line in log.read_text().splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "thread.started":
                    thread_id = event["thread_id"]
                    break
            if thread_id is not None:
                break
    while time.time() < deadline:
        turn += 1
        output = root / "codex" / f"turn_{turn:03d}"
        output.mkdir(parents=True)
        write_json(
            output / "agent_settings.json",
            {
                "model": model,
                "reasoning_effort": reasoning_effort,
            },
        )
        command = ["codex", "-a", "never", "exec"]
        if thread_id is None:
            command += ["--sandbox", sandbox, "-C", str(root / "workspace")]
            command += ["--add-dir", str(root)]
            command += ["--color", "never"]
        else:
            command += ["resume", thread_id]
        command += ["--json", "-o", str(output / "final.txt")]
        command += [
            "--model",
            model,
            "-c",
            f"model_reasoning_effort={json.dumps(reasoning_effort)}",
        ]
        command += ["-"]
        print(f"[Agent Lab] Codex turn {turn}: {output}", flush=True)
        env = _environment(Path(manifest["repo"]))
        env["GENSIM_LAB_BROKER"] = str(root)
        env["GENSIM_LAB_DEADLINE"] = str(deadline)
        process = _process(
            command,
            cwd=root / "workspace",
            output=output,
            timeout=max(1, deadline - time.time()),
            env=env,
            prompt=prompt,
            service=lambda: _serve_requests(root, deadline),
        )
        execution.update(
            status=(
                "timed_out"
                if process["timed_out"]
                else "completed" if process["returncode"] == 0 else "failed"
            ),
            process=process,
        )
        if process["returncode"] != 0 and not process["timed_out"]:
            execution["agent_error"] = {
                "returncode": process["returncode"],
                "stderr": str(output / "stderr.log"),
                "message": "Codex failed; no model or effort fallback was applied.",
            }
        for line in (output / "stdout.log").read_text().splitlines():
            if line.startswith("{"):
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    # A killed CLI can leave an incomplete final JSONL record.
                    continue
                if event.get("type") == "thread.started":
                    thread_id = event["thread_id"]
        write_json(root / "session.json", {"thread_id": thread_id, "turns": turn})
        if execution.get("agent_error"):
            break
        candidate = root / "workspace" / "candidate.json"
        if candidate.exists():
            value = json.loads(candidate.read_text())
            replay = execute_script(
                root,
                root / "workspace" / value["script"],
                root / "workspace" / value["config"],
                label="replay",
                timeout=max(1, total_deadline - time.time()),
            )
            result = {
                "status": "review_required",
                "task_success": None,
                "candidate": value,
                "replay": replay,
                "thread_id": thread_id,
            }
            write_json(root / "summary.json", result)
            return result
        if process["timed_out"] or process["returncode"] != 0:
            break
        prompt = (
            "Continue from notes.md and current artifacts. There is still time before "
            f"the original deadline {deadline:.0f}. Run the next concrete experiment, "
            "inspect frames and telemetry, and produce candidate.json only after a "
            "complete execution. Do not merely repeat the previous status."
        )
    result = {
        "status": "no_candidate",
        "task_success": None,
        "thread_id": thread_id,
        "turns": turn,
        "agent_error": execution.get("agent_error"),
    }
    write_json(root / "summary.json", result)
    return result
