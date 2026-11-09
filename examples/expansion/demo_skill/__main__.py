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

"""Migrate the verified real-source carton example to whole and mixed DemoSkills.

The original raw HDF5, imported measured units, preparation reports and actual
submitted nominal commands remain distinct evidence. Reusing a baseline never
credits a new physical episode; both new deployments execute through the Engine.
"""

from __future__ import annotations

import argparse
from dataclasses import replace, asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from embodichain.data import get_data_path
from embodichain.data_pipeline.real2sim import (
    RobotDemonstration,
    DemoSkillReference,
    DemoSkillSpec,
    DemoSkillRole,
    DemoSkillPhase,
    DemoSkillConstraint,
    DemoSkillEvent,
    TrajectoryTimeMap,
    build_demo_skill,
    load_prototype,
    save_demo_skill,
    load_demo_skill,
)

ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / "extensions/embodichain_tasks/configs/tasks/manipulation/carton_stocking"
DT = 1 / 30


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def _prepare_asset(baseline, directory, selected):
    report_path = baseline / "meta/real2sim/collection-report.json"
    report = json.loads(report_path.read_text())
    if report.get("attachments_verified") is not True or not report["cases"]:
        raise ValueError("baseline must have verified real-source collection evidence")
    case = report["cases"][0]
    if (
        report.get("journal_verified") is not True
        or not report.get("confirmed_episodes")
        or any(check.get("status") != "passed" for check in case.get("checks", ()))
    ):
        raise ValueError(
            "reference preparation requires a receipt-confirmed, physically validated baseline"
        )
    rawroot = Path(get_data_path("Real2SimCarton"))
    original = rawroot / "raw/trajectory.h5"
    units = [
        load_prototype(rawroot / f"inputs/measured/{index}/prototype.json")
        for index in selected
    ]
    episode = baseline / "meta/real2sim/episodes" / f"{case['episode_uuid']}.npz"
    trace_path = episode.with_name(episode.stem + "-trace.npz")
    observed_path = episode.with_name(episode.stem + "-observed.npz")
    with np.load(episode, allow_pickle=False) as data:
        commands = np.array(data["actions"])
        journal = json.loads(
            (
                baseline / "meta/embodichain_commits" / f"{case['episode_uuid']}.json"
            ).read_text()
        )
        observation_names = journal["sidecar"]["real2sim_episode_evidence"][
            "observation_names"
        ]
        observed = np.array(data[f"observation_{observation_names.index('qpos')}"])
    with np.load(trace_path, allow_pickle=False) as data:
        trace = {key: np.array(data[key]) for key in data.files}
    with np.load(observed_path, allow_pickle=False) as data:
        object_poses = np.array(data["object_pose"])
        target_pose = np.array(data["target_pose"])
    raw_positions = []
    raw_times = []
    raw_phases = []
    raw_events = []
    raw_offsets = {}
    for index, unit in zip(selected, units):
        source = unit.unit.source
        skip = bool(raw_positions)
        if skip and not np.array_equal(np.asarray(raw_positions[-1]), source.qpos[0]):
            raise ValueError(
                "selected real-source units must share their observed boundary sample"
            )
        offset = len(raw_positions) - (1 if skip else 0)
        raw_offsets[index] = offset
        start = raw_times[-1] if skip else 0.0
        raw_positions.extend(source.qpos[1:] if skip else source.qpos)
        raw_times.extend(
            start
            + source.timestamps[1 if skip else 0 :]
            * (0.001 if source.time_unit == "ms" else 1)
        )
        if skip:
            raw_phases[-1] = replace(raw_phases[-1], stop=offset)
        for phase in source.phases:
            name = f"source_{index}_{phase.name}"
            kind = "hold" if phase.kind in {"hold", "contact"} else "free"
            raw_phases.append(
                DemoSkillPhase(name, offset + phase.start, offset + phase.stop, kind)
            )
            raw_events.append(
                DemoSkillEvent(name + "_start", offset + phase.start, "checkpoint")
            )
    raw_times = np.asarray(raw_times)
    # Every copied sample is still an observed state. Clock concatenation is
    # explicit; prepared nominal commands are stored only in DemoSkillReference.
    first = units[0].unit.source
    names = first.joint_names
    robot_digest = first.robot_geometry_sha256
    from embodichain.data_pipeline.real2sim._geometry import (
        _canonical_robot_geometry_sha256,
    )

    robot_digest = _canonical_robot_geometry_sha256(robot_digest)
    source = RobotDemonstration(
        first.source_id,
        first.source_revision,
        "carton-" + "-".join(map(str, selected)),
        names,
        first.joint_units,
        np.array(raw_positions),
        raw_times,
        "s",
        robot_digest,
        _hash(original),
        calibration_revision=first.calibration_revision,
        annotation_revision=first.annotation_revision,
        source_sample_offset=json.loads(
            (
                ROOT
                / "examples/expansion/trajectory/real2sim_carton/annotations/staples.json"
            ).read_text()
        )["phases"][selected[0]]["start"],
        commands=np.concatenate([unit.unit.source.commands for unit in units]),
        command_representation=first.command_representation,
        command_alignment=first.command_alignment,
        qvel=(
            None
            if first.qvel is None
            else np.concatenate(
                [
                    unit.unit.source.qvel[(1 if index else 0) :]
                    for index, unit in enumerate(units)
                ]
            )
        ),
        initial_observed_qpos=first.initial_observed_qpos,
        eef_poses=(
            None
            if first.eef_poses is None
            else np.concatenate(
                [
                    unit.unit.source.eef_poses[(1 if index else 0) :]
                    for index, unit in enumerate(units)
                ]
            )
        ),
        pose_length_unit=first.pose_length_unit,
        quaternion_order=first.quaternion_order,
        world_from_pose_frame=first.world_from_pose_frame,
        observed_link_to_tcp=first.observed_link_to_tcp,
        pose_origin=first.pose_origin,
    )
    carton = units[0].unit.source.object_pose
    relation = np.eye(4)
    relation[2, 3] = 0.043592974023364456 / 2 + 0.0365978079862995 / 2
    if selected == [1]:
        # A transport-only skill retains the measured grasp and checks its final
        # rigid relation. It makes no claim of acquiring or releasing the object.
        active = np.flatnonzero(trace["step_call_index"] == 1)
        start, stop = int(active[0]), int(active[-1]) + 1
        # Terminal TCP relation is supplied by the task's calibrated grasp.
        integration = yaml.safe_load(
            (TASK / "task_program/integration.yaml").read_text()
        )
        grasp = tuple(
            integration["profile"]["action_options"]["pick"]["fixed_object_to_eef"]
        )
        constraints = (
            DemoSkillConstraint(
                "retain_tool",
                "relative_pose",
                "throughout",
                subject="tcp",
                other="carton",
                reference_pose=grasp,
                position_tolerance_m=0.015,
                rotation_tolerance_rad=0.15,
            ),
            DemoSkillConstraint(
                "retained_at_exit",
                "relative_pose",
                "terminal",
                subject="tcp",
                other="carton",
                reference_pose=grasp,
                position_tolerance_m=0.015,
                rotation_tolerance_rad=0.15,
            ),
        )
        # qpos authority freezes scene roles at the recorded entry snapshot.
    else:
        start, stop = 0, len(commands)
        constraints = (
            DemoSkillConstraint(
                "placed",
                "relative_pose",
                "terminal",
                subject="carton",
                other="worktop",
                reference_pose=tuple(relation.flat),
                position_tolerance_m=0.025,
                rotation_tolerance_rad=np.deg2rad(5),
            ),
        )
    spec = DemoSkillSpec(
        asset_id="carton." + ("transport" if selected == [1] else "whole"),
        instruction="Move the carton with the real expert demonstration",
        phases=tuple(raw_phases),
        events=tuple(raw_events),
        constraints=constraints,
        roles=(
            DemoSkillRole(
                "carton", first.object_geometry_sha256, tuple(carton.flat), "held"
            ),
            DemoSkillRole(
                "worktop",
                first.target_geometry_sha256
                or units[-1].unit.source.target_geometry_sha256,
                tuple(target_pose.flat),
            ),
        ),
        initial_held_role="carton" if selected == [1] else None,
        max_translation_m=0.0,
        max_yaw_rad=0.0,
    )
    # The original robot declaration supplies exact mimic and unit contracts.
    original_manifest = json.loads(
        (rawroot / f"inputs/measured/{selected[0]}/prototype.json").read_text()
    )
    from embodichain.data_pipeline.real2sim import JointSpec

    joints = tuple(JointSpec(**value) for value in original_manifest["joints"])
    asset = build_demo_skill(
        source, spec, joints=joints, control_dt=DT, event_tolerance_s=DT / 2
    )
    # The bridge records acknowledgement parking frames as actual submitted
    # holds. They are not additional nominal plan samples. Recover the nominal
    # clock from the explicit call/plan trace, retaining repeated gated samples
    # and separate post-policy holds with control-grid rounding.
    selected_steps = []
    hold_run = 0
    for step in range(start, stop):
        valid = (
            int(trace["step_call_index"][step]) >= 0
            and int(trace["step_plan_index"][step]) >= 0
        )
        if valid:
            selected_steps.append(step)
            hold_run = 0
        elif (
            step > start
            and int(trace["step_call_index"][step - 1]) >= 0
            and int(trace["step_plan_index"][step - 1]) >= 0
        ):
            hold_run = 0
        else:
            if hold_run % 2 == 0:
                selected_steps.append(step)
            hold_run += 1
    if selected != [1]:
        last_command = max(
            step for step in selected_steps if int(trace["step_plan_index"][step]) >= 0
        )
        # Final settling remains a Task Program post-policy. Converting a measured
        # controller hold into a trajectory stop would invent a velocity jump.
        selected_steps = [step for step in selected_steps if step <= last_command]
    labels = []
    left = []
    right = []
    weight = []
    mask = []
    events = []
    last_raw = 0
    for step in selected_steps:
        call = int(trace["step_call_index"][step])
        point = int(trace["step_plan_index"][step])
        unit = units[selected.index(call)] if call in selected else None
        call_record = next(
            (item for item in case["calls"] if item["call_index"] == call), None
        )
        source_index = point - (
            call_record["source_sample_offset"] if call_record else 0
        )
        if unit is not None and point >= 0 and 0 <= source_index < len(unit.unit.qpos):
            phase = next(
                p for p in unit.unit.phases if p.start <= source_index < p.stop
            )
            label = f"source_{call}_{phase.name}"
            rawmap = unit.unit.report.time_map
            l = int(rawmap.left[source_index]) + raw_offsets[call]
            r = int(rawmap.right[source_index]) + raw_offsets[call]
            w = float(rawmap.weight[source_index])
            is_source = True
            last_raw = l
        else:
            label = (
                "generated"
                if unit is not None and point >= 0 and source_index < 0
                else (labels[-1] if labels else "generated")
            )
            l = r = last_raw
            w = 0.0
            is_source = False
        labels.append(label)
        left.append(l)
        right.append(r)
        weight.append(w)
        mask.append(is_source)
    # Generated connections and post-policy holds are represented explicitly.
    # They never gain trajectory edit permission from a neighboring source phase.
    phases = []
    offset = 0
    generated = 0
    for end in range(1, len(labels) + 1):
        if end < len(labels) and labels[end] == labels[offset]:
            continue
        name = labels[offset]
        if name == "generated":
            name = f"generated_{generated}"
            generated += 1
            kind = "free"
        else:
            kind = next(p.kind for p in raw_phases if p.name == name)
        phases.append(DemoSkillPhase(name, offset, end, kind))
        offset = end
    for event in raw_events:
        phase = next(
            p for p in phases if p.name == event.event_id.removesuffix("_start")
        )
        events.append(replace(event, sample_index=phase.start))
    event_pairs = sorted(
        {
            (
                event.sample_index,
                next(
                    original.sample_index
                    for original in raw_events
                    if original.event_id == event.event_id
                ),
            )
            for event in events
        }
    )
    # Generated first frames map to the nearest raw ancestor; source_mask keeps
    # them distinguishable. Source brackets remain monotone across source units.
    left = np.maximum.accumulate(left)
    right = np.maximum(np.maximum.accumulate(right), left)
    event_raw = np.array([item[1] for item in event_pairs], dtype=np.int64)
    event_exec = np.array([item[0] for item in event_pairs], dtype=np.int64)
    mapping = TrajectoryTimeMap(left, right, weight, event_raw, event_exec, 0.0)
    actual = object_poses[start]
    reference_carton = np.eye(4)
    reference_carton[:3, :3] = Rotation.from_quat(actual[3:7]).as_matrix()
    reference_carton[:3, 3] = actual[:3]
    dataset_info = json.loads((baseline / "meta/info.json").read_text())
    action_names = dataset_info["features"]["action"]["names"]
    state_names = dataset_info["features"]["observation.state"]["names"]
    if set(action_names) != set(asset.joint_names) or set(state_names) != set(
        asset.joint_names
    ):
        raise ValueError("reference data must cover the exact full robot joint layout")
    commands = commands[:, [action_names.index(name) for name in asset.joint_names]]
    observed = observed[:, [state_names.index(name) for name in asset.joint_names]]
    input_commands = commands[selected_steps]
    reference_commands = np.array(input_commands, copy=True)
    hold_projection = []
    for index, step in enumerate(selected_steps):
        if index and int(trace["step_call_index"][step]) < 0:
            reference_commands[index] = reference_commands[index - 1]
            hold_projection.append(index)
    projection = {}
    names_to_id = {name: index for index, name in enumerate(asset.joint_names)}
    for joint in joints:
        if joint.mimic_driver is None:
            continue
        child = names_to_id[joint.name]
        parent = names_to_id[joint.mimic_driver]
        expected = (
            reference_commands[:, parent] * joint.mimic_multiplier + joint.mimic_offset
        )
        error = np.abs(reference_commands[:, child] - expected)
        projection[joint.name] = {
            "max_abs_correction": float(error.max()),
            "changed_samples": int(np.count_nonzero(error > 1e-7)),
        }
        reference_commands[:, child] = expected
    import torch
    from embodichain.lab.sim.motion.expansion.operators import validate_motion_samples
    from embodichain.lab.task_program.integrations.simulation.real2sim_collection import (
        load_real2sim_tabletop_profile,
    )

    motion_profile = load_real2sim_tabletop_profile(TASK / "real2sim/collection.yaml")
    intervals = torch.full((len(reference_commands),), DT, dtype=torch.float64)
    intervals[0] = 0
    motion_report = validate_motion_samples(
        torch.as_tensor(reference_commands),
        intervals,
        velocity_limits=torch.tensor(motion_profile.velocity_limits),
        acceleration_limits=torch.tensor(motion_profile.acceleration_limits),
    )
    if not motion_report.accepted:
        raise ValueError(
            f"prepared nominal reference violates its declared motion limits: {motion_report}"
        )
    reference = DemoSkillReference(
        reference_commands,
        np.arange(len(selected_steps)) * DT,
        tuple(phases),
        tuple(events),
        mapping,
        np.asarray(mask),
        json.dumps(
            {
                "origin": "adapted_nominal",
                "baseline_report_sha256": _hash(report_path),
                "baseline_episode_sha256": _hash(episode),
                "source_recording_sha256": _hash(original),
                "derivation": "submitted commands from a verified DROID-source nominal execution",
                "calls": case["calls"],
                "mimic_target_projection": projection,
                "motion_validation": [
                    dict(check_id=c.check_id, status=c.status, metrics=dict(c.metrics))
                    for c in motion_report.checks
                ],
                "post_policy_hold_projection": hold_projection,
                "input_action_joint_names": action_names,
                "reference_joint_names": asset.joint_names,
                "baseline_action_indices": selected_steps,
                "omitted_acknowledgement_waits": (stop - start) - len(selected_steps),
            },
            sort_keys=True,
        ),
        initial_qpos=observed[start],
        input_qpos=input_commands,
        role_poses=(
            ("carton", tuple(reference_carton.flat)),
            ("worktop", tuple(target_pose.flat)),
        ),
    )
    asset = replace(asset, reference=reference)
    return asset, save_demo_skill(asset, directory, original_recording=original)


def _deployment(directory, asset, manifest, mixed):
    from embodichain.lab.task_program.integrations.composition import (
        load_configured_task_program_deployment,
    )
    from embodichain.lab.task_program.integrations._program_identity import (
        task_program_fingerprint,
    )

    directory.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load((TASK / "task.panda.yaml").read_text())
    base.pop("real2sim", None)
    base.pop("collection", None)
    base["id"] = "DemoSkillCartonMixed-v1" if mixed else "DemoSkillCartonWhole-v1"
    base["environment"]["component"] = str(TASK / "env.yaml")
    base["embodiment"]["component"] = str(
        (TASK / base["embodiment"]["component"]).resolve()
    )
    base["task_program"]["execution_policy"] = str(
        (TASK / base["task_program"]["execution_policy"]).resolve()
    )
    integration = yaml.safe_load((TASK / "task_program/integration.yaml").read_text())
    program = yaml.safe_load((TASK / "task_program/program.yaml").read_text())
    callid = "task.carton_transport" if mixed else "task.carton_whole"
    program["program_id"] = (
        "demo_skill_carton_mixed" if mixed else "demo_skill_carton_whole"
    )
    integration["program_id"] = program["program_id"]
    integration["integration_id"] = (
        "demo_" + integration["integration_id"] + ("_mixed" if mixed else "_whole")
    )
    call = {
        "kind": "registered",
        "call_id": callid,
        "arguments": {"carton": "carton", "worktop": "worktop"},
    }
    if mixed:
        program["program"]["items"][1]["steps"]["call"] = call
    else:
        program["program"] = {
            "kind": "segment",
            "name": "whole_carton",
            "steps": {"kind": "invoke", "call": call},
            "post": [{"kind": "wait_stable", "entity": "carton"}],
            "validators": [{"kind": "object_upright", "object": "carton"}],
        }
    integration["profile"]["defaults"]["execute_demo_skill"] = {
        "primary": "primary_manipulator"
    }
    integration["profile"]["action_options"].pop("simulation.move_held_object", None)
    integration["profile"]["action_options"][callid] = {"kind": "execute_demo_skill"}
    service = integration["runtime_services"]
    sources = service.pop("nominal_motion_sources", [])
    service["registered_semantic_lowerers"] = [
        {
            "kind": "demo_skill",
            "call_id": callid,
            "asset_manifest": str(manifest),
            "asset_sha256": asset.content_sha256,
        }
    ]
    if mixed:
        old = yaml.safe_load((TASK / "task_program/integration.yaml").read_text())[
            "runtime_services"
        ]["registered_semantic_lowerers"]
        service["registered_semantic_lowerers"].extend(old)
    base["task_program"].update(
        {"program": "program.yaml", "integration": "integration.yaml"}
    )
    _write(directory / "program.yaml", program)
    _write(directory / "integration.yaml", integration)
    _write(directory / "task.yaml", base)
    from embodichain.lab.gym.utils._component_composition import _resolve_gym_components

    physical = _resolve_gym_components(base, base_dir=directory)
    deployment = load_configured_task_program_deployment(
        task_program=base["task_program"],
        skill_profile=physical.embodiment_skill_profile,
        base_dir=directory,
    )
    from embodichain.lab.task_program import load_task_program

    selection = deployment.selection
    compiled = deployment.integration.registration.catalog.preflight(
        load_task_program(
            directory / "program.yaml",
            integration=selection,
            validation_context=deployment.integration.registration.catalog,
        )
    )
    fingerprint = task_program_fingerprint(compiled)
    declaration = {
        "call_index": 1 if mixed else 0,
        "program_fingerprint": fingerprint,
        "provider_id": "demo_skill",
        "revision": "1",
        "call_id": callid,
        "asset_manifest": str(manifest),
        "asset_sha256": asset.content_sha256,
    }
    if mixed:
        sources = [source for source in sources if source["call_index"] != 1]
        for source in sources:
            source["program_fingerprint"] = fingerprint
    else:
        sources = []
    service["nominal_motion_sources"] = [*sources, declaration]
    _write(directory / "integration.yaml", integration)
    finger_links = yaml.safe_load((TASK / "real2sim/collection.yaml").read_text())[
        "finger_links"
    ]
    contacts = [
        {"pair": ["body:carton", "body:worktop"], "depth_m": 0.002},
        {"pair": ["body:carton", "body:ground"], "depth_m": 0.002},
        {"pair": ["robot:arm_base_link", "body:ground"], "depth_m": 0.0035},
    ]
    contact_profile = yaml.safe_load((TASK / "real2sim/collection.yaml").read_text())
    for left, right, depth in contact_profile.get("gripper_contact", {}).get(
        "mechanism_contacts", ()
    ):
        contacts.append({"pair": ["robot:" + left, "robot:" + right], "depth_m": depth})
    contacts.extend(
        {"pair": ["body:carton", "robot:" + name], "depth_m": 0.002}
        for name in finger_links
    )
    collision = {
        "kind": "sampled_rigid_v1",
        "initial_checks": [
            {
                "kind": "pose_near",
                "entity": "carton",
                "position": [
                    0.6387912983941013,
                    -0.1782805922781439,
                    0.06189187801651421,
                ],
                "tolerance_m": 0.01,
            }
        ],
        "final_checks": [
            {
                "kind": "pose_near",
                "entity": "carton",
                "position": [
                    0.656938142745868,
                    0.01285806378570287,
                    0.06189187801651421,
                ],
                "tolerance_m": 0.025,
            },
            {
                "kind": "speed_below",
                "entity": "carton",
                "linear_m_s": 0.03,
                "angular_rad_s": 0.2,
            },
        ],
        "events": [
            {
                "event_id": "lifted",
                "conditions": [
                    {"kind": "lifted", "entity": "carton", "height_m": 0.03}
                ],
            }
        ],
        "allowed_contacts": contacts,
    }
    _write(directory / "collision.yaml", collision)
    prototype_profile = None
    if mixed:
        from embodichain.lab.task_program.integrations.simulation.real2sim_collection import (
            load_real2sim_tabletop_profile,
        )

        profile = load_real2sim_tabletop_profile(TASK / "real2sim/collection.yaml")
        profile.source_artifacts = {
            index: value
            for index, value in profile.source_artifacts.items()
            if index in {0, 2}
        }
        prototype_profile = directory / "prototype_validation.yaml"
        _write(prototype_profile, asdict(profile))
    _write(
        directory / "job.yaml",
        {
            "deployment": "task.yaml",
            "collision_profile": "collision.yaml",
            "output_dir": str(directory / "output"),
            "batch_size": 1,
            "target_episodes": 1,
            "max_attempts": 1,
            "max_steps": 2400,
            "seed": 7,
            "settling_steps": 0,
            **(
                {"prototype_validation_profile": str(prototype_profile)}
                if prototype_profile is not None
                else {}
            ),
        },
    )
    return directory / "job.yaml"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Existing verified dataset directory from the real2sim carton example.",
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    baseline = args.baseline
    if baseline is None:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "embodichain",
                "run-task",
                "--gym-config",
                str(TASK / "task.panda.yaml"),
                "--dataset-dir",
                str(root / "baseline"),
                "--max_episodes",
                "1",
                "--device",
                "cpu",
                "--headless",
                "--renderer",
                "no-render",
                "--disable-sensor",
            ],
            check=True,
            cwd=ROOT,
        )
        baseline = next((root / "baseline").glob("*"))
    for mode, selected in (("mixed", [1]), ("whole", [0, 1, 2])):
        manifest = root / (mode + "_asset") / "demo_skill.json"
        if manifest.exists():
            asset = load_demo_skill(manifest)
            if json.loads(asset.reference.preparation_json)[
                "baseline_report_sha256"
            ] != _hash(baseline / "meta/real2sim/collection-report.json"):
                raise ValueError(
                    "existing prepared asset belongs to a different baseline"
                )
        else:
            asset, manifest = _prepare_asset(
                baseline.resolve(), root / (mode + "_asset"), selected
            )
        job = _deployment(root / mode, asset, manifest, mode == "mixed")
        if not args.prepare_only:
            from embodichain.lab.expansion import ExpansionEngine, load_expansion_job

            ExpansionEngine.from_job(load_expansion_job(job)).run()


if __name__ == "__main__":
    main()
