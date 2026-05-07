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

from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from embodichain.utils.utility import encode_image


class CodePrompt:
    @staticmethod
    def generate_code(**kwargs) -> ChatPromptTemplate:
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=(
                        "You are a reliable robot control code generator.\n"
                        "Your task is to generate Python code that executes robot arm actions.\n\n"
                        "CRITICAL RULES:\n"
                        "- The TASK PLAN defines the available atomic actions, rules, and execution logic. You MUST strictly follow the TASK PLAN.\n"
                        "- The CONSTRAINTS section contains additional global constraints you must obey.\n"
                        "- Do NOT invent new actions, functions, parameters, or control flow.\n"
                        "- You MAY include Python comments (# ...) inside the code.\n"
                        "- Your ENTIRE response MUST be a single Python code block.\n"
                        "- The code block MUST be directly executable without modification.\n"
                        "- Do NOT include any text, explanation, or markdown outside the Python code block.\n"
                    )
                ),
                HumanMessagePromptTemplate.from_template(
                    [
                        {
                            "type": "text",
                            "text": (
                                "TASK PLAN (atomic actions, rules, and intended behavior):\n"
                                "{task_plan}\n\n"
                                "GLOBAL CONSTRAINTS (must be satisfied):\n"
                                "{code_prompt}\n\n"
                                "REFERENCE CODE (style and structure only; do NOT copy logic):\n"
                                "{code_example}\n\n"
                                "Generate the corrected Python code block."
                            ),
                        }
                    ]
                ),
            ]
        )
        return prompt.invoke(kwargs)

    @staticmethod
    def generate_recovery_code(**kwargs) -> ChatPromptTemplate:
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=(
                        "You are a reliable robot control code generator.\n"
                        "Your task is to generate Python code that executes robot arm actions, injects runtime errors when needed, attaches the provided monitoring functions to the correct execution steps, and wires recovery actions to the matching failures.\n\n"
                        "CRITICAL RULES:\n"
                        "- The TASK PLAN is the primary execution skeleton. You MUST first preserve its step order, action semantics, arm usage, and intended behavior.\n"
                        "- The ANTICIPATED FAILURES section is the augmentation source. You MUST use its per-step Errors and Policies blocks to decide which task-plan steps receive injected errors, monitor functions, and recovery actions.\n"
                        "- Do not rewrite, reorder, merge, or delete task-plan steps because of anticipated failures.\n"
                        "- The ANTICIPATED FAILURES section may contain `Step i: None`; when that happens, generate the normal `drive(...)` call for that step without adding error, monitor, or recovery logic.\n"
                        "- For non-empty steps in ANTICIPATED FAILURES, map the listed Error lines and Policy lines to the same task-plan step `i`.\n"
                        "- The Errors block specifies which runtime error functions may be injected on that step.\n"
                        "- Each `M<i>.<j>` with matching `R<i>.<j>` specifies one `monitor_sequence` and one matching `recovery_sequence` for that step.\n"
                        "- The AVAILABLE RUNTIME ERROR FUNCTIONS section lists the error sources you can use for runtime injection, including their categories and kwargs. You MUST only use these error sources.\n"
                        "- The AVAILABLE MONITOR FUNCTIONS section lists all monitor functions you can use. You MUST only use these functions when implementing the Policies block.\n"
                        "- The MONITOR CONSTRAINTS section contains additional global constraints you must obey.\n"
                        "- All monitor functions use unified semantics: `True` means trigger / failure occurred.\n"
                        "- Preserve the original task-plan step order and action semantics.\n"
                        "- Add `error_functions=[...]`, `monitor_sequences=[...]`, and `recovery_sequences=[...]` only to the steps whose anticipated failures justify them.\n"
                        "- When a step has no meaningful anticipated failure to monitor or recover from, do not add unnecessary runtime logic.\n"
                        "- Object-level error functions should be written as `partial(inject_object_error, ...)`, while action-level error functions should be written as `partial(inject_action_error, ...)`. Do not expose or set runtime error probability in generated code.\n"
                        "- Respect the runtime semantics of the error category: `object_error_types` are step-level disturbances, while `action_error_types` are applied once before the whole action trajectory.\n"
                        "- `monitor_sequences` and `recovery_sequences` align one-to-one by outer index.\n"
                        "- `error_functions` does not need to align one-to-one with `monitor_sequences`.\n"
                        "- Different errors may share the same monitor-and-recovery policy.\n"
                        "- A single step may contain multiple monitor-and-recovery policies.\n"
                        "- Recovery actions must stay inside the current atomic action system. Do not invent a new recovery API.\n"
                        "- Do NOT invent new actions, functions, parameters, or control flow.\n"
                        "- You MAY include Python comments (# ...) inside the code.\n"
                        "- Your ENTIRE response MUST be a single Python code block.\n"
                        "- The code block MUST be directly executable without modification.\n"
                        "- Do NOT include any text, explanation, or markdown outside the Python code block.\n"
                    )
                ),
                HumanMessagePromptTemplate.from_template(
                    [
                        {
                            "type": "text",
                            "text": (
                                "TASK PLAN (the executable action plan you must implement):\n"
                                "{task_plan}\n\n"
                                "ANTICIPATED FAILURES (the step-aligned error and policies specification you must translate into runtime code):\n"
                                "{anticipated_failures}\n\n"
                                "AVAILABLE RUNTIME ERROR FUNCTIONS:\n"
                                "{error_functions}\n\n"
                                "AVAILABLE MONITOR FUNCTIONS:\n"
                                "{monitor_functions}\n\n"
                                "MONITOR CONSTRAINTS (available monitor usage rules you must follow):\n"
                                "{recovery_code_prompt}\n\n"
                                "REFERENCE MONITOR CODE EXAMPLE (style and structure only; do NOT copy logic directly):\n"
                                "{recovery_code_example}\n\n"
                                "Required behavior:\n"
                                "- Convert the TASK PLAN into executable `drive(...)` calls first.\n"
                                "- Then augment each `drive(...)` call only with the error/policies information from the matching step in ANTICIPATED FAILURES.\n"
                                "- For a step whose ANTICIPATED FAILURES entry is `None`, generate only the normal task-plan `drive(...)` call.\n"
                                "- For a step with valid Errors and Policies content, keep the original task-plan action for that step and attach matching `error_functions=[...]`, `monitor_sequences=[...]`, and `recovery_sequences=[...]` to that same `drive(...)` call.\n"
                                "- Parse every error line `E<i>.<j> [error_type=...]` and use that exact `error_type` to choose the corresponding runtime error function.\n"
                                "- Put all valid error lines from the same step into the same `error_functions=[...]` list.\n"
                                "- Parse every valid `M<i>.<j>` and matching `R<i>.<j>` pair for that step.\n"
                                "- Implement each `M<i>.<j>` line exactly using the listed monitor function names and kwargs.\n"
                                "- For object failures, use `partial(inject_object_error, ...)`; for action failures, use `partial(inject_action_error, ...)`.\n"
                                "- When choosing an error function, respect its category and kwargs exactly as described in the available runtime error functions section.\n"
                                "- Do not choose new monitor functions yourself. Use the monitor call(s) already specified in `M<i>.<j>`.\n"
                                "- If `M<i>.<j>` contains multiple monitor calls joined by ` | `, convert them into a single inner list inside `monitor_sequences`.\n"
                                "- Convert each valid `M<i>.<j>` into one outer entry in `monitor_sequences` for that step.\n"
                                "- Convert each valid `R<i>.<j>` into one matching outer entry in `recovery_sequences` for that step.\n"
                                "- If a step has errors but no valid `M/R` pairs, omit recovery logic for that step.\n"
                                "- If an `M<i>.<j>` line uses an unsupported monitor function or invalid arguments, do not invent a replacement; omit that `M/R` pair.\n"
                                "- The recovery entry must faithfully implement the ordered atomic actions in `R<i>.<j>` rather than inventing a different recovery sequence.\n"
                                "- Each `recovery_sequences` entry should be a sequence of callable `partial(drive, ...)` steps.\n"
                                "- Use `partial(drive, left_arm_action=..., right_arm_action=..., ...)` so a recovery step can coordinate one or both arms together.\n"
                                '- Inside each recovery `partial(drive, ...)`, wrap every atomic action lazily with `partial(...)`, for example `right_arm_action=partial(grasp, robot_name="right_arm", obj_name="bottle")`.\n'
                                "- Do not eagerly call atomic actions inside recovery definitions. For example, do not write `right_arm_action=grasp(...)` inside `partial(drive, ...)`.\n"
                                "- The generated code must be fully determined by the TASK PLAN plus the valid per-step Errors and `M/R` pairs in ANTICIPATED FAILURES.\n"
                                "- Output only the final monitor-error-recovery Python code block.\n\n"
                                "Generate the corrected Python code block."
                            ),
                        }
                    ]
                ),
            ]
        )
        return prompt.invoke(kwargs)
