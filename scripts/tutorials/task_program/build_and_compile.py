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

"""Build and compile a provider-independent Task Program in Python."""

from __future__ import annotations

from embodichain.lab.task_program import (
    CompiledTaskProgram,
    CyclicPoseTargetCfg,
    InvokeCfg,
    ObjectNearTargetValidatorCfg,
    PickCfg,
    PlaceCfg,
    PoseCfg,
    RepeatCfg,
    SegmentCfg,
    SequenceCfg,
    TargetRefCfg,
    TaskProgramCfg,
    TaskProgramCompiler,
    TaskProgramIntegrationCfg,
    WaitStablePostCfg,
)
from embodichain.lab.task_program.semantics import (
    SceneEntityManifest,
    SceneManifest,
    SceneObjectRef,
)

__all__ = [
    "build_program",
    "build_scene_manifest",
    "main",
    "print_compiled_program",
]


def build_scene_manifest() -> SceneManifest:
    """Declare the provider-independent scene identities used by the program."""
    return SceneManifest(
        entries=(
            SceneEntityManifest(
                ref=SceneObjectRef("cube"),
                semantic_type="cube",
            ),
        )
    )


def build_program() -> TaskProgramCfg:
    """Build a typed Pick-and-Place Task Program without YAML or JSON."""
    drop_poses = CyclicPoseTargetCfg(
        values=(
            PoseCfg(
                position=(0.40, -0.20, 0.10),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            ),
            PoseCfg(
                position=(0.40, 0.20, 0.10),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            ),
        )
    )
    move_cube = SegmentCfg(
        name="move_cube",
        steps=SequenceCfg(
            items=(
                InvokeCfg(
                    call=PickCfg(
                        object="cube",
                        resources={"primary": "manipulator"},
                    )
                ),
                InvokeCfg(
                    call=PlaceCfg(
                        object="cube",
                        at=TargetRefCfg(target="drop_pose"),
                        resources={"primary": "manipulator"},
                    )
                ),
            )
        ),
        post=(WaitStablePostCfg(entity="cube"),),
        validators=(
            ObjectNearTargetValidatorCfg(
                object="cube",
                target="drop_pose",
                position_tolerance=0.03,
            ),
        ),
    )
    return TaskProgramCfg(
        program_id="python_pick_and_place",
        integration=TaskProgramIntegrationCfg(
            robot_profile="tutorial_robot",
            scene_registry="tutorial_scene",
            runtime_preset="trajectory",
        ),
        targets={"drop_pose": drop_poses},
        program=RepeatCfg(count=3, body=move_cube),
    )


def print_compiled_program(compiled: CompiledTaskProgram) -> None:
    """Print the deterministic expansion produced by the compiler."""
    print(f"Program: {compiled.program_id}")
    print(f"Segments: {compiled.segment_count}")
    for segment in compiled:
        repeat = segment.repeat_frames[-1]
        call_names = " -> ".join(type(item.call).__name__ for item in segment.calls)
        target = segment.calls[-1].target_selections[0]
        print(
            f"[{segment.segment_index}] {segment.name} "
            f"(repeat {repeat.iteration_index + 1}/{repeat.count}): "
            f"{call_names}; {target.target_id}[{target.value_index}]"
        )


def main() -> None:
    """Compile and inspect the tutorial Task Program."""
    scene_manifest = build_scene_manifest()
    program = build_program()
    compiled = TaskProgramCompiler(scene_manifest).compile(program)
    print_compiled_program(compiled)


if __name__ == "__main__":
    main()
