# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------
from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from embodichain.utils.utility import encode_image
from embodichain.utils.utility import encode_image, encode_image_from_path


class TaskPrompt:
    @staticmethod
    def one_stage_prompt(observations, **kwargs):
        """
        Hybrid one-pass prompt:
        Step 1: VLM analyzes the image and extracts object IDs.
        Step 2: LLM generates task instructions using only those IDs.
        """
        # Encode image
        observation = observations["rgb"]
        kwargs.update({"observation": encode_image(observation)})

        # Build hybrid prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=(
                        "You are a precise and reliable robotic manipulation planner. "
                        "Given a camera observation and a task description, you must generate "
                        "a clear, step-by-step task plan for a robotic arm. "
                        "All actions must strictly use the provided atomic API functions, "
                        "and the plan must be executable without ambiguity."
                    )
                ),
                HumanMessagePromptTemplate.from_template(
                    [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,{observation}",
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Here is the latest camera observation.\n"
                                "First, analyze the scene in the image.\n"
                                "Then, using the context below, produce an actionable task plan.\n\n"
                                "**Environment background:** \n{basic_background}\n\n"
                                '**Task goal:** \n"{task_prompt}"\n\n'
                                "**Available atomic actions:** \n{atom_actions}\n"
                            ),
                        },
                    ]
                ),
            ]
        )

        # Return the prompt template and kwargs to be executed by the caller
        return prompt.invoke(kwargs)

    @staticmethod
    def two_stage_prompt(observations, **kwargs):
        # for VLM generate image descriptions
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="You are a helpful assistant to operate a robotic arm with a camera to generate task plans according to descriptions."
                ),
                HumanMessagePromptTemplate.from_template(
                    [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpg;base64,{observation}",
                            },
                        },
                        {
                            "type": "text",
                            "text": "What is in the image? Return answer with their potential effects.",
                        },
                    ]
                ),
            ]
        )

        observation = observations["rgb"]
        kwargs.update({"observation": encode_image(observation)})
        # for LLM generate task descriptions
        prompt_query = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="You are a helpful assistant to operate a robotic arm with a camera to generate task plans according to descriptions."
                ),
                HumanMessagePromptTemplate.from_template(
                    [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpg;base64,{observation}",
                            },
                        },
                        {
                            "type": "text",
                            "text": "Here is analysis for this image: {query}.",
                        },
                        {
                            "type": "text",
                            "text": (
                                "Using the context below, produce an actionable task plan.\n\n"
                                "**Environment background:** \n{basic_background}\n\n"
                                '**Task goal:** \n"{task_prompt}"\n\n'
                                "**Available atomic actions:** \n{atom_actions}\n"
                            ),
                        },
                    ]
                ),
            ]
        )

        return [prompt.invoke(kwargs), {"prompt": prompt_query, "kwargs": kwargs}]

    @staticmethod
    def one_stage_prompt_for_correction(obs_image_path, **kwargs):
        """
        Hybrid one-pass prompt:
        Step 1: VLM analyzes the image and extracts object IDs.
        Step 2: LLM generates task instructions using only those IDs.
        """
        # Encode image
        kwargs.update({"observation": encode_image_from_path(obs_image_path)})

        # Build hybrid prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=(
                        "You are a robotic manipulation planner operating STRICTLY in the robot base coordinate frame.\n\n"
                        "COORDINATE FRAME RULE (NON-NEGOTIABLE):\n"
                        "- ALL spatial reasoning and motion descriptions (left/right/front/back/up/down, offsets, rotations)\n"
                        "  are defined ONLY in the robot base coordinate frame, oriented from the base looking outward along +x (toward the end-effector).\n"
                        "- The camera is positioned in front of the robot, facing the arm and looking toward the robot base.\n"
                        "- Due to this viewpoint, the rendered image is HORIZONTALLY MIRRORED relative to the robot base frame.\n"
                        "- LEFT–RIGHT in the image MUST be inverted when reasoning:\n"
                        "    * Image left  → Robot right\n"
                        "    * Image right → Robot left\n"
                        "- Vertical orientation is preserved:\n"
                        "    * Image up   → Robot up\n"
                        "    * Image down → Robot down\n"
                        "- Always reason as if you are physically located at the robot base, facing along +x.\n"
                        "- For your output, you must use the robot base frame and explicitly account for this horizontal mirroring when interpreting the image "
                        "(e.g., What appears as “left” in the image corresponds to “right” in the robot base frame, and vice versa. "
                        "Vertical orientation is preserved: what appears as “up” in the image is also “up” in the robot base frame.).\n\n"
                        "HARD CONSTRAINT:\n"
                        "- Any reasoning based on image left/right, visual perspective, or camera orientation is VALID.\n"
                        "- If a direction cannot be inferred from the robot base frame, you must state it explicitly."
                        "- Each arm may execute at most one atomic action per step. If multiple atomic actions are required, "
                        "they must be distributed across multiple steps.\n"
                        "- Both arms may operate in the same step, but each arm may execute at most ONE atomic action per step. "
                        "If only one arm needs to act (e.g., a single-arm step or recovery), the other arm should remain idle.\n\n"
                        "TASK:\n"
                        "- Given the observation and task, produce a step-by-step plan using ONLY the provided atomic API.\n"
                        "- The plan must be executable without ambiguity.\n\n"
                    )
                ),
                HumanMessagePromptTemplate.from_template(
                    [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,{observation}",
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Here is the latest camera observation.\n"
                                "IMPORTANT: The current image may NOT represent the initial state of the task. "
                                "It may correspond to an intermediate step where some actions have already been executed.\n\n"
                                "First, analyze the scene in the image to infer the current state.\n"
                                "Then, using the context below, produce the remaining actionable task plan from this state onward.\n\n"
                                "**Environment background:** \n"
                                "{basic_background}\n\n"
                                '**Task goal:** \n"'
                                '{task_prompt}"\n\n'
                                "**Available atomic actions:** \n"
                                "{atom_actions}\n"
                                "**Failed Task Plan (Reference)::**\n"
                                "{last_task_plan}\n\n"
                                "**Executed history (reference only):**\n"
                                "{last_executed_history}\n\n"
                                "**Most recent failure (CRITICAL):**\n"
                                "{last_executed_failure}\n\n"
                                "**REQUIRED OUTPUT**\n"
                                "[PLANS]:\n"
                                "Step 1: <intent> — <atomic_action>(...)\n"
                                "..."
                                "Step N: <intent> — <atomic_action>(...)\n\n"
                                "[VALIDATION_CONDITIONS]:\n"
                                "Step 1: <explicit, image-verifiable post-condition>\n"
                                "..."
                                "Step N: <explicit, image-verifiable post-condition>\n\n"
                                "VALIDATION_CONDITIONS MUST include the robot arm and relevant object(s), and whether the object(s) should be held or not.\n"
                                "Produce the COMPLETE remaining task plan."
                            ),
                        },
                    ]
                ),
            ]
        )

        return prompt.invoke(kwargs)
