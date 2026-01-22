from typing import List, Dict, Tuple
from embodichain.agents.hierarchy.agent_base import AgentBase
from langchain_core.prompts import ChatPromptTemplate
from embodichain.data import database_2d_dir
from embodichain.utils.utility import load_txt, encode_image
from embodichain.agents.mllm.prompt import TaskPrompt
from embodichain.data import database_agent_prompt_dir
from pathlib import Path
from langchain_core.messages import HumanMessage
import numpy as np

# from openai import OpenAI
import os
import time
import cv2
import glob
import json
import re

USEFUL_INFO = """The error may be caused by: 
1. You did not follow the basic background information, especially the world coordinate system with its xyz directions.
2. You did not take into account the NOTE given in the atom actions or in the example functions.
3. You did not follow the steps of the task descriptions.\n
"""


def extract_plan_and_validation(text: str) -> Tuple[str, List[str], List[str]]:
    def get_section(src: str, name: str, next_name) -> str:
        if next_name:
            pat = re.compile(
                rf"\[{name}\]\s*:\s*(.*?)\s*(?=\[{next_name}\]\s*:|\Z)",
                re.DOTALL | re.IGNORECASE,
            )
        else:
            pat = re.compile(
                rf"\[{name}\]\s*:\s*(.*?)\s*\Z",
                re.DOTALL | re.IGNORECASE,
            )
        m = pat.search(src)
        return m.group(1).strip() if m else ""

    step_re = re.compile(
        r"Step\s*\d+\s*:.*?(?=Step\s*\d+\s*:|\Z)",
        re.DOTALL | re.IGNORECASE,
    )

    # ---- plans ----
    plans_raw = get_section(text, "PLANS", "VALIDATION_CONDITIONS")
    plan_steps = [m.group(0).rstrip() for m in step_re.finditer(plans_raw)]
    plan_str = "\n".join(plan_steps)

    # normalized plan list (strip "Step k:")
    plan_list = []
    for step in plan_steps:
        content = re.sub(r"^Step\s*\d+\s*:\s*", "", step, flags=re.IGNORECASE).strip()
        if content:
            plan_list.append(content)

    # ---- validations ----
    vals_raw = get_section(text, "VALIDATION_CONDITIONS", None)
    validation_list = []
    for m in step_re.finditer(vals_raw):
        content = re.sub(
            r"^Step\s*\d+\s*:\s*", "", m.group(0), flags=re.IGNORECASE
        ).strip()
        if content:
            validation_list.append(content)

    return plan_str, plan_list, validation_list


class TaskAgent(AgentBase):
    prompt: ChatPromptTemplate
    object_list: List[str]
    target: np.ndarray
    prompt_name: str
    prompt_kwargs: Dict[str, Dict]

    def __init__(self, llm, **kwargs) -> None:
        super().__init__(**kwargs)
        if llm is None:
            raise ValueError(
                "LLM is None. Please set the following environment variables:\n"
                "  - AZURE_OPENAI_ENDPOINT\n"
                "  - AZURE_OPENAI_API_KEY\n"
                "Example:\n"
                "  export AZURE_OPENAI_ENDPOINT='https://your-endpoint.openai.azure.com/'\n"
                "  export AZURE_OPENAI_API_KEY='your-api-key'"
            )
        self.llm = llm

    def generate(self, **kwargs) -> str:
        log_dir = kwargs.get(
            "log_dir", Path(database_agent_prompt_dir) / self.task_name
        )
        file_path = log_dir / "agent_generated_plan.txt"

        # Check if the file already exists
        if not kwargs.get("regenerate", False):
            if file_path.exists():
                print(f"Plan file already exists at {file_path}, skipping writing.")
                return load_txt(file_path)

        # Generate query via LLM
        prompts_ = getattr(TaskPrompt, self.prompt_name)(**kwargs)
        if isinstance(prompts_, list):
            # TODO: support two-stage prompts with feedback
            start_time = time.time()
            response = self.llm.invoke(prompts_[0])
            query = response.content
            print(
                f"\033[92m\nSystem tasks output ({np.round(time.time()-start_time, 4)}s):\n{query}\n\033[0m"
            )
            for prompt in prompts_[1:]:
                temp = prompt["kwargs"]
                temp.update({"query": query})
                start_time = time.time()
                response = self.llm.invoke(prompt["prompt"].invoke(temp))
                query = response.content
                print(
                    f"\033[92m\nSystem tasks output({np.round(time.time()-start_time, 4)}s):\n{query}\n\033[0m"
                )
        else:
            # insert feedback if exists
            if len(kwargs.get("error_messages", [])) != 0:
                # just use the last one
                last_plan = kwargs["generated_plans"][-1]
                last_code = kwargs["generated_codes"][-1]
                last_error = kwargs["error_messages"][-1]

                # Add extra human message with feedback
                feedback_msg = self.build_feedback_message(
                    last_plan, last_code, last_error
                )
                prompts_.messages.append(feedback_msg)

            response = self.llm.invoke(prompts_)
            print(f"\033[92m\nTask agent output:\n{response.content}\n\033[0m")

        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            f.write(response.content)
        print(f"Generated task plan saved to {file_path}")

        return response.content

    def act(self, *args, **kwargs):
        return super().act(*args, **kwargs)
