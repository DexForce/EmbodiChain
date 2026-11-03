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

"""Isolated native physics and deliberate lifecycle failures, never success-data substitutes."""

from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
import time
import yaml
from ._cases import dump, worker_environment, stop_process_tree

__all__: list[str] = []


def physics_probe(backend: str) -> dict:
    import torch
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import RigidObjectCfg, physics_cfg_for_backend
    from embodichain.lab.sim.shapes import CubeCfg
    from embodichain.lab.sim.scene import (
        SceneView,
        capture_state,
        restore_state,
        compare_states,
    )
    from embodichain.lab.sim.scene.physics import (
        capture_rigid_physics,
        apply_rigid_physics,
        restore_rigid_physics,
    )

    cfg = SimulationManagerCfg(
        headless=True,
        device="cpu",
        num_envs=8,
        physics_cfg=physics_cfg_for_backend(backend),
    )
    cfg.render_cfg.renderer = "no-render"
    sim = SimulationManager(cfg)
    try:
        cube = sim.add_rigid_object(
            RigidObjectCfg(
                uid="cube", shape=CubeCfg(size=(0.05, 0.05, 0.05)), init_pos=(0, 0, 0.5)
            )
        )
        sim.prepare()
        cube = sim.get_rigid_object("cube")
        rows = [7, 1, 5]
        view = SceneView(sim, env_ids=tuple(rows))
        initial = capture_state(view)
        baseline = capture_rigid_physics(view, ("cube",))
        untouched = cube.get_mass()[[0, 2, 3, 4, 6]].clone()
        apply_rigid_physics(
            view, baseline, mass_scale={"cube": torch.tensor([0.5, 2.0, 1.0])}
        )
        mass = cube.get_mass(env_ids=rows).clone()
        torch.testing.assert_close(
            cube.get_mass()[[0, 2, 3, 4, 6]], untouched, atol=0, rtol=0
        )
        cube.clear_dynamics(env_ids=list(range(8)))
        cube.add_force_torque(
            force=torch.tensor([[1.0, 0.0, 0.0]]).repeat(3, 1), env_ids=rows
        )
        sim.update(step=1)
        speed = cube.body_state[rows, 7].clone()
        momentum = mass * speed
        assert bool((speed > 0).all())
        torch.testing.assert_close(
            momentum, momentum[0].expand_as(momentum), rtol=0.02, atol=1e-6
        )
        restore_rigid_physics(view, baseline)
        assert capture_rigid_physics(view, ("cube",)) == baseline
        contact = cube.get_local_pose()[rows].clone()
        contact[:, 2] = 0.025
        cube.set_local_pose(contact, env_ids=rows)
        cube.clear_dynamics(env_ids=rows)
        sim.update(step=20)
        apply_rigid_physics(
            view, baseline, friction={"cube": torch.tensor([0.0, 1.0, 0.5])}
        )
        cube.set_velocity(
            lin_vel=torch.tensor([[3.0, 0.0, 0.0]]).repeat(3, 1),
            ang_vel=torch.zeros(3, 3),
            env_ids=rows,
        )
        sim.update(step=15)
        sliding = cube.body_state[rows, 7].clone()
        assert float(sliding[0] - sliding[1]) > 0.1
        restore_rigid_physics(view, baseline)
        restore_state(view, initial)
        assert compare_states(initial, capture_state(view)).passed
        return {
            "ok": True,
            "backend": backend,
            "kind": "controlled_parameter_probe",
            "physical_episode_credit": 0,
            "env_ids": rows,
            "mass": mass.tolist(),
            "velocity_after_equal_impulse": speed.tolist(),
            "momentum": momentum.tolist(),
            "friction": [0.0, 1.0, 0.5],
            "sliding_velocity": sliding.tolist(),
            "baseline_restored": True,
            "untouched_rows_preserved": True,
        }
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


def install_fault(name: str, root: Path) -> None:
    """Install a process-local adapter only for explicit diagnostic worker runs."""
    if name == "after_commit":
        import embodichain.lab.expansion.workflow as workflow

        original = workflow.ArtifactEpisodeSink

        class LostReceiptSink(original):
            def submit(self, episode, *, submission_id):
                receipt = super().submit(episode, submission_id=submission_id)
                marker = root / "injected-after-commit.json"
                if not marker.exists():
                    dump(
                        marker,
                        {
                            "commit_id": receipt.commit_id,
                            "storage_id": receipt.storage_id,
                            "submission_id": receipt.submission_id,
                        },
                    )
                    raise RuntimeError(
                        "showcase injected lost acknowledgement after durable commit"
                    )
                return receipt

        workflow.ArtifactEpisodeSink = LostReceiptSink
    elif name == "render":
        import embodichain.lab.expansion.visual as visual

        def fail(*args, **kwargs):
            raise RuntimeError("showcase injected renderer failure")

        visual.render_observation_variants = fail
    else:
        raise ValueError(name)


def fault_checks(root: Path) -> dict:
    from embodichain.lab.expansion import (
        ArtifactEpisodeSink,
        inspect_expansion_job,
        load_expansion_episode,
        resume_expansion_export,
    )
    from embodichain.data_pipeline.recording.media import _digest

    destination = root / "diagnostics/lifecycle"
    destination.mkdir(parents=True, exist_ok=True)
    template = yaml.safe_load((root / "configs/job.ur5-baseline.yaml").read_text())
    previous = root / "results/faults.json"
    if previous.exists():
        saved = json.loads(previous.read_text())
        if saved.get("ok"):
            for name in ("interrupted", "lost-receipt", "render-failure"):
                if not inspect_expansion_job(destination / name)["complete"]:
                    raise ValueError("completed diagnostic evidence no longer verifies")
            return {**saved, "reused": True}
    checks = []

    def run(
        name, job, *, resume=False, fault=None, interrupt=False, operation="collect"
    ):
        command = [
            sys.executable,
            "-m",
            "examples.expansion.engine.showcase._worker",
            operation,
            str(destination),
            "--job",
            str(job),
        ]
        if resume:
            command.append("--resume")
        if fault:
            command.extend(("--fault", fault))
        with (destination / (name + ".log")).open("a") as log:
            # Inherit the outer worker's process group so its deadline also
            # reaches native workers and media encoders.
            proc = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, env=worker_environment()
            )
            try:
                if not interrupt:
                    return proc.wait(timeout=600)
                jobroot = Path(yaml.safe_load(job.read_text())["output_dir"])
                deadline = time.monotonic() + 300
                trigger = None
                while proc.poll() is None and time.monotonic() < deadline:
                    for path in sorted((jobroot / "attempts").glob("*/attempt.json")):
                        record = json.loads(path.read_text())
                        if (
                            record["sequence"] > 0
                            and record["status"] == "running"
                            and list((path.parent / "media").glob("chunk-*.npz"))
                        ):
                            parents = ArtifactEpisodeSink(jobroot).inspect()
                            if parents:
                                trigger = {
                                    **record,
                                    "confirmed_before_interrupt": [
                                        r.commit_id for r in parents
                                    ],
                                }
                                break
                    if trigger:
                        break
                    time.sleep(0.1)
                if trigger is None:
                    raise RuntimeError(
                        "did not observe confirmed parent plus active media before interruption"
                    )
                stop_process_tree(proc, force=True)
                return trigger
            finally:
                if proc.poll() is None:
                    stop_process_tree(proc)

    def config(name, **overrides):
        values = {
            **template,
            "output_dir": str(destination / name),
            "batch_size": 2,
            "target_episodes": 2,
            "max_attempts": 8,
            "max_wall_time_s": 600.0,
            **overrides,
        }
        path = destination / (name + ".yaml")
        dump(path, values)
        return path

    mixed_result = destination / "results/ur5.default.json"
    if not mixed_result.exists() or not json.loads(mixed_result.read_text()).get("ok"):
        assert (
            run("mixed-rows", root / "configs/ur5.default.yaml", operation="mixed") == 0
        )
    mixed = json.loads(mixed_result.read_text())
    assert _digest(destination / mixed["trace"]["path"]) == mixed["trace"]["sha256"]
    checks.append({"id": "mixed_row_termination", "ok": True, "evidence": mixed})
    # Interrupted physics must generate a new attempt of the same frozen recipe.
    p = config(
        "interrupted",
        batch_size=1,
        media_buffer_bytes=4 * 1024 * 1024,
        mass_scale={"cube": [0.98, 1.02]},
    )
    if not (destination / "interruption.json").exists():
        trigger = run("interrupted-first", p, interrupt=True)
        dump(destination / "interruption.json", trigger)
    else:
        trigger = json.loads((destination / "interruption.json").read_text())
    before = {
        r.commit_id: _digest(Path(r.storage_id))
        for r in ArtifactEpisodeSink(destination / "interrupted").inspect()
    }
    assert before, "interruption requires at least one verified pre-existing commit"
    values = yaml.safe_load(p.read_text())
    values["batch_size"] = 8
    values["media_buffer_bytes"] = 32 * 1024 * 1024
    dump(p, values)
    assert run("interrupted-resume", p, resume=True) == 0
    records = [
        json.loads(v.read_text())
        for v in (destination / "interrupted/attempts").glob("*/attempt.json")
    ]
    assert any(
        r["recipe_id"] == trigger["recipe_id"]
        and r["attempt_id"] == trigger["attempt_id"] + 1
        and r["status"] == "committed"
        for r in records
    )
    after = {
        r.commit_id: _digest(Path(r.storage_id))
        for r in ArtifactEpisodeSink(destination / "interrupted").inspect()
    }
    assert all(after[k] == v for k, v in before.items())
    checks.append(
        {
            "id": "interrupted_same_recipe_new_attempt",
            "ok": True,
            "old_recipe": trigger["recipe_id"],
            "capacity_before": 1,
            "capacity_after": 8,
            "preserved_commits": len(before),
        }
    )
    # Actual data is durably written; only its acknowledgement is lost.
    p = config("lost-receipt")
    if not (destination / "injected-after-commit.json").exists():
        assert run("lost-receipt-first", p, fault="after_commit") != 0
    count = len(list((destination / "lost-receipt/attempts").glob("*/attempt.json")))
    assert run("lost-receipt-resume", p, resume=True) == 0
    assert (
        len(list((destination / "lost-receipt/attempts").glob("*/attempt.json")))
        == count
    )
    sink = ArtifactEpisodeSink(destination / "lost-receipt")
    receipts = sink.inspect()
    for receipt in receipts:
        episode = load_expansion_episode(receipt.storage_id)
        repeated = sink.submit(episode, submission_id=receipt.submission_id)
        assert repeated.commit_id == receipt.commit_id and repeated.confirmed
    assert len(sink.inspect()) == 2
    checks.append(
        {
            "id": "lost_receipt_and_duplicate_ack",
            "ok": True,
            "new_physical_attempts": 0,
            "unique_commits": 2,
        }
    )
    p = config(
        "render-failure",
        target_episodes=1,
        export_lerobot=True,
        visual_profile_file=str(root / "configs/visual.ur5.yaml"),
        render_profiles=["camera"],
        observation_independent=True,
    )
    if not (destination / "render-failure/job.json").exists():
        assert run("render-first", p, fault="render") != 0
    isolation = destination / "render-isolation.json"
    if not isolation.exists():
        summary = inspect_expansion_job(destination / "render-failure")
        assert (
            summary["stages"]["export"]["complete"]
            and not summary["stages"]["render"]["complete"]
        )
        exported = resume_expansion_export(destination / "render-failure")
        assert exported["stage_complete"] and not exported["complete"]
        dump(
            isolation,
            {
                "parent_exported_before_render_retry": True,
                "export_only_complete": exported["stage_complete"],
            },
        )
    assert run("render-resume", p, resume=True) == 0
    checks.append(
        {
            "id": "render_failure_isolated_and_export_only_recovery",
            "ok": True,
            "parent_exported_before_render_retry": True,
        }
    )
    return {
        "ok": all(c["ok"] for c in checks),
        "checks": checks,
        "kind": "native_lifecycle_diagnostics",
        "physical_episode_credit": 0,
    }


def mixed_row_probe(root: Path, deployment: Path) -> dict:
    """Reject one executing arena through the public row-runtime port; keep its peer running."""
    import numpy as np
    import torch
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg
    from embodichain.lab.gym.utils.registration import REGISTERED_ENVS
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.expansion.program import TaskProgramBatchExecutor
    from embodichain.utils.utility import load_config
    from embodichain.data_pipeline.recording.media import _digest

    raw = load_config(deployment)
    cfg = config_to_cfg(raw, source_path=deployment)
    cfg.num_envs = cfg.sim_cfg.num_envs = 2
    cfg.sim_cfg.headless = True
    cfg.sim_cfg.render_cfg.renderer = "no-render"
    cfg.enable_sensor = False
    cfg.dataset = None
    cfg.init_rollout_buffer = cfg.record_trajectory = cfg.trajectory_auto_save = False
    cfg.seed = 7
    env = REGISTERED_ENVS[raw["id"]].make(cfg=cfg).unwrapped
    try:
        env.reset(options={"save_data": False})
        program = env.compile_task_program(cfg.task_program)
        bridges = {
            i: env.task_program_adapter.for_env_ids((i,)).create_bridge(program)
            for i in (0, 1)
        }
        frames = [0, 0]
        qpos = [[env.robot.get_qpos()[i].cpu().clone()] for i in (0, 1)]
        commands = [[], []]

        def observe(ids, action, obs):
            values = (
                action.value
                if isinstance(action.value, torch.Tensor)
                else action.value["qpos"]
            )
            assert values.shape[0] == 2
            measured = env.robot.get_qpos().cpu()
            for i in ids:
                frames[i] += 1
                commands[i].append(values[i].cpu().clone())
                qpos[i].append(measured[i].clone())

        result = TaskProgramBatchExecutor(env, program, row_bridges=bridges).run(
            (0, 1),
            max_steps=2000,
            observe_step=observe,
            reject_row=lambda row: row == 0 and frames[row] >= 10,
        )
        assert result.accepted == (False, True), result
        assert frames[0] == 11 and frames[1] > frames[0], frames
        assert tuple(frames) == result.lengths
        path = root / "mixed-row-trace.npz"
        np.savez_compressed(
            path,
            **{
                **{f"qpos_{i}": torch.stack(qpos[i]).numpy() for i in (0, 1)},
                **{f"actions_{i}": torch.stack(commands[i]).numpy() for i in (0, 1)},
            },
        )
        return {
            "ok": True,
            "kind": "native_mixed_row_guard",
            "physical_episode_credit": 0,
            "accepted": list(result.accepted),
            "action_lengths": frames,
            "observation_lengths": [len(v) for v in qpos],
            "reasons": list(result.reasons),
            "trace": {"path": path.name, "sha256": _digest(path)},
        }
    finally:
        env.close(exit_process=False)
        SimulationManager.flush_cleanup_queue()
