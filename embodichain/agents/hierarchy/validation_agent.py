from langchain_core.prompts import ChatPromptTemplate
import os
from langchain_core.messages import SystemMessage, HumanMessage
from abc import ABCMeta
from embodichain.utils.utility import encode_image_from_path
import glob
from embodichain.agents.hierarchy.llm import view_selection_llm


def save_obs_image(obs_image, save_dir, step_id=None):
    """
    Save observation image using encode_image() and return its file path.
    """
    import base64
    from embodichain.utils.utility import encode_image

    if obs_image is None:
        return None

    if isinstance(save_dir, str):
        from pathlib import Path

        save_dir = Path(save_dir)

    save_dir.mkdir(parents=True, exist_ok=True)

    name = f"obs_step_{step_id}.png" if step_id is not None else "obs.png"
    img_path = save_dir / name

    # Encode to base64
    base64_image = encode_image(obs_image)

    # Decode base64 → bytes
    img_bytes = base64.b64decode(base64_image)

    # Write to file
    with open(img_path, "wb") as f:
        f.write(img_bytes)

    return img_path


def get_obj_position_info(env):
    import json

    position_info = {}
    obj_uids = env.sim.get_rigid_object_uid_list()
    for obj_name in obj_uids:
        target_obj = env.sim.get_rigid_object(obj_name)
        target_obj_pose = target_obj.get_local_pose(to_matrix=True).squeeze(0)[:3, 3]
        position_info[obj_name] = target_obj_pose.tolist()
    return json.dumps(position_info, indent=4)


class ValidationAgent(metaclass=ABCMeta):

    def __init__(self, llm, **kwargs) -> None:
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.llm = llm

    def validate(self, step_names, problematic_code, error_message, image_files):
        # Construct the prompt
        prompt = f"""
        Analyze the execution of the following robot task:

        Task name: {self.task_name}
        Task description: {self.task_description}
        Basic background knowledge: {self.basic_background}

        You will be given images showing each step of the execution. For the step sequence:
        {', '.join(step_names)}

        Provide the following analysis:
        1. Determine whether each step was executed correctly.
        2. If a step failed, identify which one and explain the cause.
        3. Decide whether the full task succeeded or failed.
        4. If the task failed, provide a precise and detailed explanation.

        Below is a potentially problematic piece of code and the corresponding execution error:

        ```python
        {problematic_code}
        # Execution error:
        {error_message}
        Explain whether (and how) this code contributed to the observed failure.
        """

        # Prepare message content for API call
        user_content = []

        # Add textual prompt
        user_content.append({"type": "text", "text": prompt})

        # Add images and step names
        for img_path in image_files:
            filename = os.path.basename(img_path)
            first_underscore_pos = filename.find("_")
            if first_underscore_pos != -1:
                step_name = filename[first_underscore_pos + 1 :].rsplit(".", 1)[0]
            else:
                step_name = filename.rsplit(".", 1)[0]

            # Add step name
            user_content.append({"type": "text", "text": f"Step: {step_name}"})

            # Add image as base64
            base64_image = encode_image_from_path(img_path)
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                }
            )

        messages = [
            SystemMessage(
                content="You are a robot task execution analysis expert. Please analyze the provided image sequence."
            ),
            HumanMessage(content=user_content),
        ]

        response = self.llm.invoke(messages)
        return response.content

    def select_best_view_dir(
        self, img_dirs: dict, action_description: str, valid_condition: str
    ):
        """
        img_dirs: {
            "cam_1": Path,
            "cam_2": Path,
            "cam_3": Path
        }
        """

        # --- collect final images ---
        last_images = {}
        for cam_id, cam_dir in img_dirs.items():
            imgs = sorted(
                glob.glob(os.path.join(cam_dir, "obs_step_*.png")),
                key=lambda p: int(os.path.basename(p).split("_")[-1].split(".")[0]),
            )
            if imgs:
                last_images[cam_id] = imgs[-1]

        if not last_images:
            raise ValueError("No images found in any camera directory.")

        # --- system prompt ---
        system_prompt = (
            "You are a robot perception assistant specialized in VIEW SELECTION.\n\n"
            "TASK:\n"
            "- You are given ONE final observation image from EACH camera view.\n"
            "- Your job is NOT to judge success or failure.\n"
            "- Your job is ONLY to select the SINGLE camera view that is MOST SUITABLE\n"
            "  for OBJECT-LEVEL validation of the action result.\n\n"
            "ACTION CONTEXT:\n"
            "- The robot has just executed ONE atomic action.\n"
            "- You are given the action intention and the expected object-level outcome\n"
            "  ONLY to help you decide which view best reveals that outcome.\n\n"
            "SELECTION CRITERIA (PRIORITY ORDER):\n"
            "- Prefer views with:\n"
            "  * the clearest visibility of the relevant object(s)\n"
            "  * minimal occlusion by the arm or environment\n"
            "  * the clearest evidence related to the expected object-level result\n"
            "    (e.g., contact, separation, support, stability)\n\n"
            "STRICT CONSTRAINTS:\n"
            "- Do NOT judge robot motion quality or execution accuracy.\n"
            "- Do NOT reason about numeric values (distance, angle, offset).\n"
            "- Do NOT decide whether the action succeeded or failed.\n"
            "- If multiple views are acceptable, choose the clearest overall view.\n\n"
            "OUTPUT FORMAT (STRICT):\n"
            "Output EXACTLY ONE of the following tokens:\n"
            "- cam_1\n"
            "- cam_2\n"
            "- cam_3\n"
        )

        # --- human content ---
        human_content = [
            {
                "type": "text",
                "text": (
                    "Select the best camera view for object-level validation.\n\n"
                    "--------------------------------------------------\n"
                    "ACTION DESCRIPTION (INTENT ONLY):\n"
                    f"{action_description}\n\n"
                    "EXPECTED OBJECT-LEVEL RESULT (REFERENCE ONLY):\n"
                    f"{valid_condition}\n"
                    "--------------------------------------------------"
                ),
            }
        ]

        for cam_id, img_path in last_images.items():
            img_b64 = encode_image_from_path(img_path)
            human_content.extend(
                [
                    {"type": "text", "text": f"View candidate: {cam_id}"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    },
                ]
            )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_content),
        ]

        response = view_selection_llm.invoke(messages).content.strip()

        if response not in img_dirs:
            raise ValueError(f"Invalid camera selection from LLM: {response}")

        return response

    def validate_single_action(
        self,
        current_action,
        action_description,
        valid_condition,
        img_dir,
        obj_position_info,
    ):
        # --- camera directories ---
        img_dirs = {
            "cam_1": img_dir / "cam_1",
            "cam_2": img_dir / "cam_2",
            "cam_3": img_dir / "cam_3",
        }

        # === Stage 1: select best view ===
        selected_cam = self.select_best_view_dir(
            img_dirs, action_description, valid_condition
        )
        selected_dir = img_dirs[selected_cam]
        print(f"\033[38;5;214mSelected camera for validation: {selected_cam}\033[0m")

        # === Stage 2: load FULL sequence from selected view ===
        image_files = glob.glob(os.path.join(selected_dir, "obs_step_*.png"))
        if len(image_files) < 2:
            raise ValueError("Need at least two observation images for validation.")

        # Sort images by step index
        image_files_sorted = sorted(
            image_files,
            key=lambda p: int(os.path.basename(p).split("_")[-1].split(".")[0]),
        )

        # Encode ALL images in sequence
        encoded_images = [encode_image_from_path(p) for p in image_files_sorted]

        system_prompt = (
            "You are a helpful robot manipulation ACTION VALIDATOR.\n\n"
            "ROLE:\n"
            "- Judge ONLY the OBJECT-LEVEL outcome of ONE atomic action.\n"
            "- Do NOT judge robot motion, planning, or execution quality.\n\n"
            "CORE ASSUMPTIONS:\n"
            "- The robot arm motion itself is correct by definition.\n"
            "- Any failure must be due to incorrect OBJECT interaction or state.\n\n"
            "COORDINATE RULE:\n"
            "- The image is horizontally mirrored: image left ↔ robot right, image right ↔ robot left.\n"
            "- Vertical direction is preserved.\n"
            "- Use robot base frame terminology in your final judgment.\n\n"
            "EVALUATION RULES:\n"
            "- Focus on the FINAL image.\n"
            "- Earlier images are context only.\n"
            "- Do NOT infer numeric precision or motion quality.\n"
            "- Ignore minor offsets or simulation noise.\n\n"
            "DECISION POLICY:\n"
            "- If visual evidence contradicts the expected object state → FAILURE.\n"
            "- If visual evidence clearly matches the expected object state → SUCCESS.\n"
        )

        prompt = f"""
            Validate the result of ONE atomic robot action.
    
            --------------------------------------------------
            ACTION
            --------------------------------------------------
            {action_description}
    
            --------------------------------------------------
            EXPECTED OBJECT-LEVEL OUTCOME
            --------------------------------------------------
            {valid_condition}
    
            --------------------------------------------------
            INPUT
            --------------------------------------------------
            You are given an ordered image sequence.
            - The FINAL image shows the state AFTER the action.
    
            --------------------------------------------------
            OUTPUT FORMAT (STRICT)
            --------------------------------------------------
            Output EXACTLY one of the following.
            
            IMPORTANT:
            - You MUST explicitly state the correctness of BOTH arms in the Evidence.
            
            [ACTION_SUCCESS]
            - Evidence: <one concise sentence describing the correctness of both arms and object state>
            
            [ACTION_FAILED]
            - Reason: <one concise sentence>
            - Evidence: <clear visual cues from the FINAL image, describing both arms>
        """

        # Build multimodal message with ALL images
        human_content = [{"type": "text", "text": prompt}]
        for img_b64 in encoded_images:
            human_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                }
            )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_content),
        ]

        llm_response = self.llm.invoke(messages)
        return llm_response.content
