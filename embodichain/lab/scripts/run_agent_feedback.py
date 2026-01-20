# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

import gymnasium
import numpy as np
import argparse
import os
import torch
import json

from threading import Thread
from tqdm import tqdm
from embodichain.utils.utility import load_json
from embodichain.lab.sim import SimulationManagerCfg
from embodichain.lab.gym.envs import EmbodiedEnvCfg
from embodichain.lab.gym.utils.gym_utils import (
    config_to_cfg,
)
from embodichain.lab.scripts.generate_video import visualize_data_dict
from embodichain.data.data_engine.online.online_generator import (
    OnlineGenerator,
)
from embodichain.utils.logger import log_warning, log_info, log_error
from embodichain.data import database_agent_prompt_dir
from pathlib import Path
import traceback

def test_code(env, code_file_path, check_num=10, kwargs=None):
    """Test the generated code multiple times and evaluate task success rate.
    
    Uses env.code_agent.act() to execute the code, which handles all the
    necessary imports and execution logic.
    """
    # ====== Read code content for display ======
    with open(code_file_path, "r", encoding="utf-8") as f:
        code_content = f.read()
    
    # ====== Initialize kwargs ======
    if kwargs is None:
        kwargs = {}
    if "env" not in kwargs:
        kwargs["env"] = env

    # ====== Initialize counters ======
    epid, suc_num, fail_num = 0, 0, 0
    run_records = []

    # Error categories (same style as previous run() function)
    error_list = [
        "Code can not run",  # 0
        "Task executed but failed",  # 1
        "No error occurred"  # 2
    ]
    error_num = [0, 0, 0]

    print("\033[93m" + "[Start Testing Task Success Rate]" + "\033[0m")

    # ====== Print generated source ======
    print("\n\033[92m=== generated source code ===\033[0m")
    print(code_content)
    print("\033[92m=== End ===\033[0m\n")

    # ====== Main loop ======
    for epid in range(check_num):
        env.reset()
        kwargs['current_check_num'] = epid
        error_id = None

        try:
            # Use code_agent.act() to execute the code
            # This method handles all imports and execution logic
            env.get_wrapper_attr("code_agent").act(code_file_path, **kwargs)

            # Check result
            if env.get_wrapper_attr("is_task_success")().item():
                print(f"simulate data episode {suc_num} success! (seed = {epid})")
                suc_num += 1
                run_records.append("Success!")
            else:
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                fail_num += 1
                error_id = 1
                run_records.append(error_list[1])

        except Exception as e:
            # Execution error
            exec_trace = traceback.format_exc()
            error_list[0] = exec_trace  # store full traceback for summary
            error_id = 0
            fail_num += 1

            run_records.append(f"Code can not run, error: {exec_trace}")

            print("-------------")
            print(f"simulate data episode {suc_num} fail! (seed = {epid})")
            print("Error:", exec_trace)
            print("-------------")

        # Count error category
        if error_id is not None:
            error_num[error_id] += 1

    # ====== Find most frequent error ======
    if sum(error_num) == 0:
        max_error_index = 2  # no errors, fallback to "NO error"
        max_error_count = 0
    else:
        max_error_index = error_num.index(max(error_num))
        max_error_count = error_num[max_error_index]

    # ====== Summary ======
    print(f'\nComplete test, success rate: {suc_num}/{check_num}')
    print(f'Error message list: {error_list}')
    print(f'Error count: {error_num}')
    print(f'Run records: {run_records}')

    return suc_num / check_num, error_list[max_error_index], max_error_count, run_records


def generate_function(
    env,
    generated_codes,
    error_messages,
    log_dir=None,
):
    # Initialize env
    env.reset()

    # First attempt case - create initial code file
    if len(error_messages) == 0:
        code_file_path, kwargs, code = env.get_wrapper_attr("generate_code_for_actions")(regenerate=True, log_dir=log_dir)
    # Generate code based on error status
    else:
        code_file_path, kwargs, code = env.get_wrapper_attr("generate_code_for_actions")(
            regenerate=True, log_dir=log_dir, generated_codes=generated_codes, error_messages=error_messages)

    try:
        # Update this section to match the new return values of the run function
        success_rate, error_message, error_count, run_records = test_code(env, code_file_path, check_num=5, kwargs=kwargs)
        generated_codes.append(code)
        error_messages.append(error_message)
        return code, success_rate, error_message, error_count, run_records
    except KeyboardInterrupt:
        print("Test interrupted by user")
        return code, 0, "Test interrupted by user", 10, None
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error occurred during testing: {e}\n{error_trace}")
        return code, 0, f"Error occurred during testing: {e}", 10, None


def main(args, env, gym_config):

    log_info("Start agent data generation with feedback.", color="green")

    # Initialize variables
    generate_num = 5
    success_threshold = 0.6
    suc_list = []

    # Store each round's code and error
    error_messages = []
    generated_codes = []

    # Store the best code and its success rate
    best_code = None
    best_success_rate = 0
    best_run_records = None

    # Create log file name with timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(database_agent_prompt_dir) / args.task_name / "feedback_logs" / timestamp
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"{log_dir}/{args.task_name}.log"

    # Store all attempt records
    all_attempts = []

    # Try multiple generations until success or limit reached
    for id in range(generate_num):
        log_info(f"Generate code for task: {args.task_name} ({id + 1}/{generate_num})", color='green')

        # Generate and test code
        code, success_rate, error_message, error_count, run_records = generate_function(
            env, generated_codes, error_messages, log_dir)

        # Track success rates
        suc_list.append(success_rate)

        # Record this attempt
        attempt_record = {
            "attempt_id": id + 1,
            "success_rate": success_rate,
            "error_message": error_message,
            "error_count": error_count,
            "code": code,
            "run_records": run_records
        }
        all_attempts.append(attempt_record)

        # Save best code
        if success_rate > best_success_rate:
            best_success_rate = success_rate
            best_code = code
            best_run_records = run_records
            print(f"New best code found, success rate: {best_success_rate}")

        # Check if generation was successful
        if success_rate >= success_threshold:
            print(f"Successfully generated code for task: {args.task_name}")
            break

        # Handle failure case
        log_warning(f"The generated code fail for task: {args.task_name} (attempt {id+1}) with succuss rate {success_rate}\nError message: \n{error_message}")

    # Ensure the final saved code is the best one
    if best_code is not None:
        file_name = log_dir / "agent_generated_code.py"
        print(f"Saving best code, success rate: {best_success_rate}")
        with open(file_name, 'w') as file:
            file.write(best_code)

    print(f"Best success rate: {best_success_rate}")
    print(f"All success rates: {suc_list}")

    # Save log data to file
    with open(log_filename, 'w') as log_file:
        log_data = {
            "task_name": args.task_name,
            "best_success_rate": best_success_rate,
            "success_rates": suc_list,
            "best_code": best_code,
            "best_run_records": best_run_records,
            "all_attempts": all_attempts
        }
        json.dump(log_data, log_file, indent=2)

    print(f"Log has been saved to: {log_filename}")

    if args.headless:
        env.reset(options={"final": True})


if __name__ == "__main__":
    np.set_printoptions(5, suppress=True)
    torch.set_printoptions(precision=5, sci_mode=False)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_envs",
        help="The number of environments to run in parallel.",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="device to run the environment on, e.g., 'cpu' or 'cuda'",
    )
    parser.add_argument(
        "--headless",
        help="Whether to perform the simulation in headless mode.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--enable_rt",
        help="Whether to use RTX rendering backend for the simulation.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--render_backend",
        help="The rendering backend to use for the simulation.",
        default="egl",
        type=str,
    )
    parser.add_argument(
        "--gpu_id",
        help="The GPU ID to use for the simulation.",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--save_video",
        help="Whether to save data as video.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--save_path", help="path", default="./outputs/thirdviewvideo", type=str
    )
    parser.add_argument(
        "--debug_mode",
        help="Enable debug mode.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--filter_visual_rand",
        help="Whether to filter out visual randomization.",
        default=False,
        action="store_true",
    )

    parser.add_argument("--online_config", type=str, help="online_config", default="")
    parser.add_argument("--gym_config", type=str, help="gym_config", default="")
    parser.add_argument("--task_name", type=str, help="Name of the task.", required=True)

    # Agent related configs
    parser.add_argument("--agent_config", type=str, help="agent_config", default=None, required=True)
    parser.add_argument("--regenerate", type=bool, help="Whether regenerate code if already existed.", default=False)

    args = parser.parse_args()

    if args.num_envs != 1:
        log_error(f"Currently only support num_envs=1, but got {args.num_envs}.")

    gym_config = load_json(args.gym_config)
    cfg: EmbodiedEnvCfg = config_to_cfg(gym_config)
    cfg.filter_visual_rand = args.filter_visual_rand

    agent_config = load_json(args.agent_config)

    cfg.num_envs = args.num_envs
    cfg.sim_cfg = SimulationManagerCfg(
        headless=args.headless,
        sim_device=args.device,
        enable_rt=args.enable_rt,
        gpu_id=args.gpu_id,
    )

    env = gymnasium.make(id=gym_config["id"], cfg=cfg, agent_config=agent_config, task_name=args.task_name)
    main(args, env, gym_config)
