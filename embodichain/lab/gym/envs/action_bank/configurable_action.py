# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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

import torch
import numpy as np
import functools
import networkx as nx
import matplotlib.pyplot as plt

from copy import deepcopy
from typing import Dict, Tuple, Union, List, Callable, Any
from tqdm import tqdm
from functools import partial

from embodichain.utils.math import pose_inv
from embodichain.utils.logger import log_info, log_warning, log_error
from embodichain.lab.sim.cfg import MarkerCfg
from embodichain.lab.gym.utils.misc import resolve_env_params, data_key_to_control_part
from embodichain.data.enum import Hints, EefExecute
from .utils import generate_affordance_from_src, get_init_affordance


# https://stackoverflow.com/questions/41834530/how-to-make-python-decorators-work-like-a-tag-to-make-function-calls-by-tag
class TagDecorator(object):
    def __init__(self, tagName):
        self.functions = {}
        self.tagName = tagName

    def __str__(self):
        return "<TagDecorator {tagName}>".format(tagName=self.tagName)

    def __call__(self, f):
        # class_key = f"{f.__module__}.{f.__qualname__.rsplit('.', 1)[0]}"
        class_name = f.__qualname__.split(".")[0]
        if class_name in self.functions.keys():
            self.functions[class_name].update({f.__name__: f})
        else:
            self.functions.update({class_name: {f.__name__: f}})
        return f


@functools.lru_cache(maxsize=None)  # memoization
def get_func_tag(tagName):
    return TagDecorator(tagName)


tag_node = get_func_tag("node")
tag_edge = get_func_tag("edge")


class ActionBank:
    _function_type: Dict[str, Callable]

    def __init__(self, conf: Dict):
        self.conf = conf

    @property
    def vis_gantt(self):
        return self.conf.get("misc", {}).get("vis_gantt", False)

    @property
    def vis_graph(self):
        return self.conf.get("misc", {}).get("vis_graph", False)

    @property
    def warpping(self):
        return self.conf.get("misc", {}).get("warpping", True)

    @staticmethod
    def get_function_name(input: Dict) -> str:
        """
        Retrieve the function name from the input dictionary.

        This method assumes that the input dictionary contains exactly one key,
        which represents the function name. If the dictionary contains more than
        one key, a ValueError is raised.

        Args:
            input (Dict): A dictionary with a single key representing the function name.

        Returns:
            str: The function name extracted from the dictionary.

        Raises:
            ValueError: If the input dictionary contains zero or more than one key.
        """
        if len(list(input.keys())) != 1:
            raise ValueError(
                "The input dict {} has invalid keys {}.".format(
                    input, list(input.keys())
                )
            )

        return list(input.keys())[0]

    def get_scope_names(
        self,
    ) -> List[str]:
        return list(self.conf["scope"].keys())

    def get_node_names(self, bool_attr_name: str = None) -> Dict[str, List[str]]:
        scopes = self.get_scope_names()
        nodes = self.conf["node"]
        node_names = {}
        for scope in scopes:
            node_names[scope] = []
            for node in nodes[scope]:
                if bool_attr_name is not None:
                    if node[self.get_function_name(node)].get(bool_attr_name, False):
                        function_name = ActionBank.get_function_name(node)
                        node_names[scope].append(function_name)
        return node_names

    def graph2id(self, type: str = "node") -> Dict[str, Dict[str, str]]:
        scopes = self.get_scope_names()
        nodes = self.conf[type]
        graph_2_id = {}
        for scope in scopes:
            graph_2_id[scope] = {}
            for i, node in enumerate(nodes[scope]):
                function_name = ActionBank.get_function_name(node)
                graph_2_id[scope].update({function_name: i})
        return graph_2_id

    def get_edge_names(self, node_name: str = None) -> Dict[str, List[Dict[str, str]]]:
        scopes = self.get_scope_names()
        edges = self.conf["edge"]
        edge_names = {}
        for i, key in enumerate(scopes):
            edge_names[key] = []
            for edge in edges[key]:
                function_name = ActionBank.get_function_name(edge)
                src = edge[function_name]["src"]
                sink = edge[function_name]["sink"]
                temp = {"name": function_name, "src": src, "sink": sink}
                edge_names[key].append(temp)
        if node_name is None:
            return edge_names
        else:
            filtered_edge_names = {}
            for scope, edge_list in edge_names.items():
                filtered_edge_names[scope] = [
                    edge
                    for edge in edge_list
                    if edge["src"] == node_name or edge["sink"] == node_name
                ]
            return filtered_edge_names

    def _infer_fill_type(self, scope: str, label: str, edge_cfg: Dict) -> str:
        # 1) explicit in config
        ft = edge_cfg.get("fill_type", None)
        # 2) built-in eef rules
        fn = edge_cfg.get("name", None)
        if fn in {EefExecute.OPEN.value, EefExecute.CLOSE.value}:
            return "still"
        # 3) explicit wins; otherwise default
        return ft if ft in ("still", "scalable") else "still"

    def _get_unit_pairs(self, legends: List[str]) -> Dict[str, str]:
        """
        Return a symmetric map executor -> partner within the same unit (arm+eef).
        Priority:
          1) explicit config: self.conf["misc"]["unit_pairs"] = [["right_arm","right_eefhand"], ["left_arm","left_eefhand"], ...]
          2) heuristic by side-prefix and name hints ('arm' vs 'eef'/'hand'/'gripper')
        """
        pairs: Dict[str, str] = {}

        # 1) explicit mapping if provided
        explicit = self.conf.get("misc", {}).get("unit_pairs", None)
        if explicit:
            for a, b in explicit:
                if a in legends and b in legends:
                    pairs[a] = b
                    pairs[b] = a

        # 2) heuristic fallback
        def side_key(name: str) -> str:
            # prefer token before '_' or '-', else prefix match
            if "_" in name:
                return name.split("_", 1)[0]
            if "-" in name:
                return name.split("-", 1)[0]
            for pref in ("left", "right", "L", "R"):
                if name.lower().startswith(pref.lower()):
                    return pref
            return ""

        eef_hints = Hints.EEF.value
        arm_hints = Hints.ARM.value

        from collections import defaultdict

        by_side = defaultdict(list)
        for n in legends:
            by_side[side_key(n)].append(n)

        for _, names in by_side.items():
            arms = [n for n in names if any(h in n.lower() for h in arm_hints)]
            eefs = [n for n in names if any(h in n.lower() for h in eef_hints)]
            # pair the first unmatched arm with the first unmatched eef
            for a in arms:
                if a in pairs:
                    continue
                partner = next((e for e in eefs if e not in pairs), None)
                if partner:
                    pairs[a] = partner
                    pairs[partner] = a

        return pairs

    def _apply_bubble_filling(self, packages, taskkey2index):
        from collections import defaultdict

        if not packages:
            return packages

        # group by executor
        per_legend = defaultdict(list)
        for p in packages:
            per_legend[p["legend"]].append(p)
        for lg in per_legend:
            per_legend[lg].sort(key=lambda x: (x["start"], x["end"]))

        legends = list(per_legend.keys())
        unit_pairs = self._get_unit_pairs(legends)

        # fill_type lookup
        fill_type = {}
        for scope, scope_edges in self.conf.get("edge", {}).items():
            for edge in scope_edges:
                lbl = list(edge.keys())[0]
                fill_type[lbl] = edge[lbl].get("fill_type", "still")

        label2pkg = {p["label"]: p for p in packages}
        global_end = max(p["end"] for p in packages)

        def first_start_at_or_after(seq, t):
            for pkg in seq:
                if pkg["start"] >= t:
                    return pkg["start"]
            return global_end

        # optional sync boundary (unchanged)
        dep_of = defaultdict(list)
        for e_label, s in self.conf.get("sync", {}).items():
            for d in s.get("depend_tasks", []):
                dep_of[d].append(e_label)

        def sync_boundary_for(lbl):
            deps = dep_of.get(lbl, [])
            if not deps:
                return None
            starts = [label2pkg[d]["start"] for d in deps if d in label2pkg]
            return max(starts) if starts else None

        # unit-aware filling
        for lg, seq in per_legend.items():
            partner = unit_pairs.get(lg, None)
            partner_seq = per_legend.get(partner, []) if partner else []

            # middle gaps
            for i in range(len(seq) - 1):
                curr, nxt = seq[i], seq[i + 1]
                if curr["end"] < nxt["start"]:
                    cap_local = nxt["start"]
                    cap_partner = (
                        first_start_at_or_after(partner_seq, curr["end"])
                        if partner_seq
                        else global_end
                    )
                    cap_sync = sync_boundary_for(curr["label"])
                    cap = (
                        min(cap_local, cap_partner, cap_sync)
                        if cap_sync is not None
                        else min(cap_local, cap_partner)
                    )
                    if cap > curr["end"]:
                        curr["end"] = cap
                        curr.setdefault(
                            "fill_type", fill_type.get(curr["label"], "still")
                        )

            # tail gap: cap by partner’s next (≥ end), not global
            if seq:
                last = seq[-1]
                cap_partner = (
                    first_start_at_or_after(partner_seq, last["end"])
                    if partner_seq
                    else last["end"]
                )
                if cap_partner > last["end"]:
                    last["end"] = cap_partner
                    last.setdefault("fill_type", fill_type.get(last["label"], "still"))

        return packages

    def parse_network(
        self,
        node_functions: Dict[str, Callable],
        edge_functions: Dict[str, Callable],
        vis_graph: bool = False,
    ) -> Tuple[nx.DiGraph, Dict[str, List], Dict[str, Tuple[int, int]]]:
        """Construct a graph with self.conf["node"]&["edge"], and node_functions, edge_functions be its node generator and edge linker.

        Return the constructed nx.DiGraph graph_compose,

        and tasks_data = {"scope name" : [(scope_id=task_id, skill_duration_{i})]},

        and taskkey2index = {"edge_name": (scope_id=task_id, edge_id=skill_id)}

        Args:
            node_functions (Dict[str, Callable]): A Dict consists of key-value pair that key be all nodes (affordance) name and value be its generating functions
            edge_functions (Dict[str, Callable]): A Dict consists of key-value pair that key be all edges (skill, a part of a trajectory) name and value be its linker functions
            vis_graph (bool, optional): Whether to show the graph or not. Defaults to True.

        Returns:
            graph_compose: A composed nx.DiGraph representing the graph defined in self.conf, while the node generators and edge linkers prepared,
            tasks_data: A Dict consists of key-value pair that key be all scopes' names, and value be List(Tuple=(scope_id=task_id, skill_duration))
            taskkey2index: A Dict consists of key-value pair that key be all edges' names, and value be Tuple=(scope_id=task_id, edge_id=skill_id)
        """
        nodes = self.conf.get("node", {})
        edges = self.conf.get("edge", {})
        graph_type = self.conf.get("scope", {})

        graphs = {key: nx.DiGraph() for key in graph_type.keys()}
        disjoint_names = {}
        tasks_data = {}
        taskkey2index = {}

        # key2index = {}
        edges_flatten = {}
        for i, key in enumerate(graphs.keys()):
            # key2index[key] = i
            for j, edge in enumerate(edges[key]):
                edge = deepcopy(edge)
                taskkey2index[ActionBank.get_function_name(edge)] = (i, j)
                edge["type"] = key
                edges_flatten.update(edge)
        for i, key in enumerate(graphs.keys()):
            tasks_data[key] = []
            for edge in edges[key]:
                label = ActionBank.get_function_name(edge)  # edge label in config
                cfg = edge[label]
                src = cfg["src"]
                sink = cfg["sink"]
                kwargs = cfg.get("kwargs", {})
                duration = cfg.get("duration", 0)
                if not isinstance(duration, int):
                    raise TypeError("Duration must be an integer.")

                # function to call
                fn_name = cfg.get("name", label)
                # normalize and persist fill_type (default + built-in rules)
                fill_type = self._infer_fill_type(key, label, cfg)

                graphs[key].add_edge(
                    src,
                    sink,
                    linker=partial(
                        edge_functions[fn_name], **kwargs, duration=duration
                    ),
                    duration=duration,
                    fill_type=fill_type,
                    edge_label=label,
                    scope=key,
                )
                tasks_data[key].append((i, duration))

            for node in nodes[key]:
                function_name = ActionBank.get_function_name(node)
                if function_name in disjoint_names.keys():
                    error_msg = f"Function {function_name} is already defined in {disjoint_names[function_name]} but re-defined in {key} again."
                    log_error(error_msg)
                disjoint_names.update({function_name: key})
                graphs[key].add_node(
                    function_name,
                    generator=partial(
                        node_functions[node[function_name]["name"]],
                        **node[function_name]["kwargs"],
                    ),
                )

        graph_compose = nx.DiGraph()
        for key, graph in graphs.items():
            if self.vis_graph or vis_graph:
                nx.draw(graph, with_labels=True)
                plt.show()
            if graph_type[key]["type"] == "tree":
                assert nx.is_tree(
                    graph.to_undirected()
                ), "{} graph is not tree.".format(key)

            graph_compose = nx.compose(graph_compose, graph)

        if self.vis_graph or vis_graph:
            nx.draw(graph_compose, with_labels=True)
            plt.show()

        return graph_compose, tasks_data, taskkey2index

    def gantt(
        self,
        tasks_data: Dict[str, List],
        taskkey2index: Dict[str, int],
        vis: bool = False,
    ) -> Dict[str, Any]:
        """Given tasks on different machines and skills within tasks that takes a specific duration, try to minimize the max length among task, while respecting:
           Constraint 1: For skills of a same task, which occupied a same machine, do not overlap with each other.
           Constraint 2: For skills of a same task, the start time of skill should not surpass the end time of the before skill
           Constraint 3: For sync edges define in self.conf["sync"], which defined a skill, its start time should not surpass the end time of the depend skill.
           with a set of start and end time of all skills. Then draw the gantt with the solution, return the solution packages with start and end time of each edge.

        Args:
            tasks_data (Dict[str, List]): A Dict consists of key-value pair that key be all scopes' names, and value be List(Tuple=(scope_id=task_id, skill_duration))
            taskkey2index (Dict[str, int]): A Dict consists of key-value pair that key be all edges' names, and value be Tuple=(scope_id=task_id, edge_id=skill_id)
            vis (bool, optional): Whether to visualize the gantt or not. Defaults to False.

        Returns:
            packages Dict[str, Any]:
                {
                    "total_num": int(solver.objective_value)=max length of a task (among tasks = scopes = machines = executors),
                    "packages": packages=List(Dict=assigned_task={
                        "labels": edge_name,
                        "start" : edge.start
                        "end": edge.start + edge.duration
                        "legend": task_name
                        "color": color representing the task_id, the former the bluer
                    }
                )
            }
        """
        import collections

        # https://developers.google.com/optimization/scheduling/task_shop?hl=zh-cn
        from ortools.sat.python import cp_model

        machines_count = len(list(tasks_data.keys()))
        all_machines = range(machines_count)
        # Computes horizon dynamically as the sum of all durations.
        id2key = {i: key for i, key in enumerate(tasks_data.keys())}
        tasks_data = list(tasks_data.values())

        horizon = sum(skill[1] for task in tasks_data for skill in task)
        model = cp_model.CpModel()
        # Named tuple to store information about created variables.
        skill_type = collections.namedtuple("skill_type", "start end interval")

        # Creates task intervals and add to the corresponding machine lists.
        all_skills = {}
        machine_to_intervals = collections.defaultdict(list)

        for task_id, task in enumerate(tasks_data):
            for skill_id, skill in enumerate(task):
                machine, duration = skill
                suffix = f"_{task_id}_{skill_id}"
                start_var = model.new_int_var(0, horizon, "start" + suffix)
                end_var = model.new_int_var(0, horizon, "end" + suffix)
                interval_var = model.new_interval_var(
                    start_var, duration, end_var, "interval" + suffix
                )
                all_skills[task_id, skill_id] = skill_type(
                    start=start_var, end=end_var, interval=interval_var
                )
                machine_to_intervals[machine].append(interval_var)

        # Create and add disjunctive constraints for each machine.
        for machine in all_machines:
            model.add_no_overlap(machine_to_intervals[machine])

        # Precedences inside a task.
        for task_id, task in enumerate(tasks_data):
            for skill_id in range(len(task) - 1):
                model.add(
                    all_skills[task_id, skill_id + 1].start
                    >= all_skills[task_id, skill_id].end
                )

        sync_edges = self.conf["sync"]
        for edge_name in sync_edges.keys():
            task_id, skill_id = taskkey2index[edge_name]
            for depend_task in sync_edges[edge_name]["depend_tasks"]:
                before_task_id, before_skill_id = taskkey2index[depend_task]
                model.add(
                    all_skills[task_id, skill_id].start
                    >= all_skills[before_task_id, before_skill_id].end
                )

        # Makespan objective.
        obj_var = model.new_int_var(0, horizon, "makespan")

        max_equality = []
        for task_id, task in enumerate(tasks_data):
            if len(task) != 0:
                max_equality.append(all_skills[task_id, len(task) - 1].end)

        model.add_max_equality(obj_var, max_equality)
        model.minimize(obj_var)
        solver = cp_model.CpSolver()
        status = solver.solve(model)

        packages = []
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            # Create one list of assigned skills per machine.
            n = len(list(id2key.keys()))
            color = [(1 - i / n, 0, i / n) for i in range(n)]
            keys_ = list(taskkey2index.keys())
            values_ = list(taskkey2index.values())
            for task_id, task in enumerate(tasks_data):
                for skill_id, skill in enumerate(task):
                    machine = skill[0]
                    duration = skill[1]
                    start = solver.value(all_skills[task_id, skill_id].start)

                    assigned_task = {}
                    assigned_task["label"] = keys_[values_.index((task_id, skill_id))]
                    assigned_task["start"] = start
                    assigned_task["end"] = start + duration
                    assigned_task["legend"] = id2key[task_id]
                    assigned_task["color"] = color[task_id]
                    packages.append(assigned_task)

            # Finally print the solution found.
            log_warning(f"Optimal Schedule Length: {solver.objective_value}")
        else:
            log_error("No solution found.")

        packages = self._apply_bubble_filling(packages, taskkey2index)

        new_total = max((p["end"] for p in packages), default=0)

        if self.vis_gantt or vis:
            from embodichain.utils.visualizer import Gantt

            draw_gantt_data = {
                "title": " Sample GANTT",
                "xlabel": "Trajectory (steps)",
                "packages": packages,
            }
            g = Gantt(draw_gantt_data)
            g.render()
            g.show()

        return {"total_num": int(new_total), "packages": packages}

    def initialize_action_list(
        self, env, action_list, executor: str, executor_init_info: Dict
    ) -> np.ndarray:
        """
        Initialize the action list for a specific executor.

        This method initializes the action trajectory for the given executor based on the provided initialization information.
        The initialization can be done using predefined qpos values or the current qpos of the executor.

        Args:
            self (ActionBank): The ActionBank instance.
            env (object): The environment instance containing executor and affordance data.
            action_list (np.ndarray): A numpy array of shape (T, qpos_dim), representing the uninitialized action trajectory for the executor.
            executor (str): The name of the executor (e.g., "left_arm", "right_arm").
            executor_init_info (Dict): A dictionary containing initialization information for the executor, such as method and parameters.

        Returns:
            np.ndarray: The initialized action list for the executor.
        """

        def initialize_with_given_qpos(action_list, executor, executor_init_info, env):
            given_qpos = executor_init_info.get("kwargs", {}).get("given_qpos", None)
            if given_qpos is None:
                log_warning(
                    "No given_qpos is provided for initialize_with_given_qpos. Using {}.".format(
                        get_init_affordance(executor)
                    )
                )
                given_qpos = env.affordance_datas[get_init_affordance(executor)]

            executor_qpos_dim = action_list[executor].shape[0]
            given_qpos = np.asarray(given_qpos)
            if len(given_qpos.shape) != 1:
                log_warning(
                    f"Shape of given init qpos should be (1,), but got {given_qpos.shape} with length {len(given_qpos.shape)}. Using 0-th element with {given_qpos.shape[-1]}."
                )
                last_ids = (0,) * (given_qpos.ndim - 1) + (Ellipsis,)
                given_qpos = given_qpos[last_ids]

            if given_qpos.shape[0] != executor_qpos_dim:
                log_error(
                    f"Shape of given init qpos should be {(executor_qpos_dim,)}, but got {given_qpos.shape[0]}."
                )

            init_node_name = executor_init_info.get(
                "init_node_name", f"{executor}_init_qpos"
            )
            if (
                len(init_node_name) > 0
            ):  # so if you don't need to inject it, just assign the "init_node_name" to be "" in action_config
                env.affordance_datas[init_node_name] = given_qpos
            action_list[executor][:, 0] = given_qpos

            return action_list

        def initialize_with_current_qpos(
            action_list, executor, executor_init_info, env
        ):
            # TODO: Hard to get current qpos for multi-agent env
            current_qpos = env.robot.get_qpos()
            joint_ids = env.robot.get_joint_ids(name=get_control_part(env, executor))

            # Handle multi-environment case
            if current_qpos.ndim == 2:
                # current_qpos shape: [num_envs, num_joints]
                # Take first environment and then select joints
                current_qpos = current_qpos[0, joint_ids].cpu()
            else:
                # Single environment case
                # current_qpos shape: [num_joints]
                current_qpos = current_qpos[joint_ids].cpu()

            executor_qpos_dim = action_list[executor].shape[0]

            # NOTE: hard code!
            current_qpos = current_qpos[:executor_qpos_dim]

            if current_qpos.shape[0] != executor_qpos_dim:
                log_error(
                    f"Shape of given init qpos should be {(executor_qpos_dim,)}, but got {current_qpos.shape[0]}."
                )

            init_node_name = executor_init_info.get(
                "init_node_name", f"{executor}_init_qpos"
            )
            if (
                len(init_node_name) > 0
            ):  # so if you don't need to inject it, just assign the "init_node_name" to be "" in action_config
                env.affordance_datas[init_node_name] = current_qpos
            action_list[executor][:, 0] = current_qpos

            return action_list

        INIT_METHOD_MAPPING = {
            "given_qpos": initialize_with_given_qpos,
            "current_qpos": initialize_with_current_qpos,
        }

        init_method = executor_init_info.get("method", "current_qpos")

        if init_method is None:
            log_warning(
                f"No init method provided in action config for executor {executor}, please check. Skipping.."
            )
            return action_list
        if init_method not in INIT_METHOD_MAPPING:
            log_warning(
                f"Provided init method action config for executor {executor}: {init_method} is not accomplished yet, please check. Skipping.."
            )
            return action_list
        else:
            init_func = INIT_METHOD_MAPPING[init_method]
            action_list = init_func(action_list, executor, executor_init_info, env)

        return action_list

    @staticmethod
    @tag_node
    @resolve_env_params
    def generate_affordances_from_src(env, affordance_infos: List[Dict]) -> bool:
        for affordance_info in affordance_infos:
            src_key = affordance_info["src_key"]
            dst_key = affordance_info["dst_key"]
            valid_funcs_name_kwargs_proc = affordance_info[
                "valid_funcs_name_kwargs_proc"
            ]
            to_array = env.action_bank.warpping

            ret = generate_affordance_from_src(
                env, src_key, dst_key, valid_funcs_name_kwargs_proc, to_array
            )
            if not ret:
                return False
        return True

    def _prepare_warpping(self, env):
        if hasattr(env, "affordance_datas"):
            for affordance_name, affordance_value in env.affordance_datas.items():
                # NOTE: take only first arena's affordance data
                if affordance_value.ndim == 3:
                    affordance_value = affordance_value[0]
                if isinstance(affordance_value, torch.Tensor):
                    affordance_value = np.asarray(affordance_value.cpu())
                env.affordance_datas[affordance_name] = affordance_value
        else:
            log_warning("No env.affordance_datas, skip _prepare_warpping..")

    def create_action_list(
        self, env, graph_compose: nx.DiGraph, packages: List[Dict], **kwargs
    ) -> Dict:
        """Create an action list based on the given environment, graph, and packages.

        Args:
            env (embodichain.lab.gym.envs.BaseEnv): The environment instance.
            graph_compose (nx.DiGraph): The composed graph containing nodes and edges.
            packages (List[Dict]): The task packages with scheduling information.

        Returns:
            Dict: The generated action list for all executors.
        """

        def initialize_action_list(
            scope: Dict, total_num: int
        ) -> Tuple[Dict, Dict, Dict]:
            """Initialize action list and related variables."""
            action_list = {}
            end_time = {}
            in_working = {}

            for executor in scope.keys():
                end_time[executor] = 0
                in_working[executor] = False

                action_list[executor] = np.zeros(
                    tuple(scope[executor]["dim"]) + (total_num,),
                    dtype=getattr(np, scope[executor]["dtype"]),
                )

                init_info = scope[executor].get("init", {})
                action_list = self.initialize_action_list(
                    env, action_list, executor, init_info
                )

            return action_list, end_time, in_working

        def generate_nodes(graph_compose: nx.DiGraph, nodes: Dict) -> bool:
            """Generate nodes using the graph's node generators."""
            node_generators = nx.get_node_attributes(graph_compose, "generator")

            failed_nodes = []
            log_info("Action bank start node generation for action graph...")
            for node_dict_list in nodes.values():
                for node in node_dict_list:
                    node_name = list(node.keys())[0]
                    try:
                        log_info(f"\tGenerating node '{node_name}' .")
                        ret = node_generators[node_name](env, **kwargs)
                        if not ret:
                            log_warning(f"Node '{node_name}' generation fails.")
                            failed_nodes.append(node_name)
                    except KeyError as e:
                        log_warning(
                            f"[KeyError] '{node_name}': {e}. Node generator might be missing or invalid."
                        )
                        failed_nodes.append(node_name)
                    except AttributeError as e:
                        log_warning(
                            f"[AttributeError] '{node_name}': {e}. Missing required attributes in environment."
                        )
                        failed_nodes.append(node_name)
                    except TypeError as e:
                        log_warning(
                            f"[TypeError] '{node_name}': {e}. Check input data types."
                        )
                        failed_nodes.append(node_name)
                    except ValueError as e:
                        log_warning(
                            f"[ValueError] '{node_name}': {e}. Check input values."
                        )
                        failed_nodes.append(node_name)
                    except Exception as e:
                        log_warning(
                            f"[UnexpectedError] '{node_name}': {e}. Debug dependencies or implementation."
                        )
                        failed_nodes.append(node_name)
            if failed_nodes:
                log_warning(f"Failed to generate the following nodes: {failed_nodes}")
                return False

            log_info(
                f"Node generation is finished. Total nodes generated: {sum(len(v) for v in nodes.values())}."
            )
            return True

        def generate_edges(
            total_num: int,
            all_executors: List[str],
            edges_flatten: Dict,
            node_linkers: Dict,
        ) -> None:
            """
            Generate edges and populate the action list for all executors.

            Args:
                total_num (int): The total number of time steps for the action list.
                all_executors (List[str]): A list of executor names (e.g., "left_arm", "right_arm").
                edges_flatten (Dict[str, Dict]): A flattened dictionary of edges, where keys are edge labels
                    and values are dictionaries containing edge details (e.g., "src", "sink").
                node_linkers (Dict[Tuple[str, str], Callable]): A dictionary mapping edge (source, sink) pairs
                    to their corresponding linker functions.

            Returns:
                None: This function modifies the `action_list` in place.
            """

            def get_task_in_time(tasks, time):
                """Get the task that is active at the given time."""
                return next(
                    (task for task in tasks if task["start"] <= time < task["end"]),
                    None,
                )

            for i in tqdm(range(total_num), desc="Generating edges"):
                for executor in all_executors:
                    if end_time[executor] == i:
                        in_working[executor] = False

                    if not in_working[executor]:
                        pkg = get_task_in_time(
                            [
                                pkg
                                for pkg in packages["packages"]
                                if pkg["legend"] == executor
                            ],
                            i,
                        )
                        if pkg is None:
                            if i >= 1:
                                action_list[executor][..., i] = action_list[executor][
                                    ..., i - 1
                                ]
                        else:
                            end_time[executor] = pkg["end"]
                            skill_idx = (
                                edges_flatten[pkg["label"]]["src"],
                                edges_flatten[pkg["label"]]["sink"],
                            )
                            ret = node_linkers[skill_idx](env)
                            if not isinstance(ret, np.ndarray):

                                if isinstance(ret, torch.Tensor):
                                    ret = ret.cpu().numpy()
                                else:
                                    raise TypeError(
                                        "The return value of the linker {} must be a numpy array, but a {}.".format(
                                            skill_idx, type(ret)
                                        )
                                    )

                            start_idx = pkg["start"]
                            end_idx = pkg["end"]

                            T_need = end_idx - start_idx
                            T_orig = ret.shape[1]

                            # fill_type of this edge
                            ft = edges_flatten[pkg["label"]].get(
                                "fill_type", pkg.get("fill_type", "still")
                            )

                            def _resample_time(x, new_T):
                                if new_T == x.shape[1]:
                                    return x
                                if x.shape[1] <= 1:
                                    return np.repeat(x, new_T, axis=1)[:, :new_T]
                                t_old = np.linspace(0.0, 1.0, x.shape[1])
                                t_new = np.linspace(0.0, 1.0, new_T)
                                out = np.empty((x.shape[0], new_T), dtype=x.dtype)
                                for d in range(x.shape[0]):
                                    out[d] = np.interp(t_new, t_old, x[d])
                                return out

                            def _pad_or_trim_last(x, new_T):
                                if new_T <= x.shape[1]:
                                    return x[:, :new_T]
                                pad = np.repeat(x[:, -1:], new_T - x.shape[1], axis=1)
                                return np.concatenate([x, pad], axis=1)

                            if T_need != T_orig:
                                if ft == "scalable":
                                    ret = _resample_time(ret, T_need)
                                else:  # "still"
                                    ret = _pad_or_trim_last(ret, T_need)

                            action_list[executor][..., start_idx:end_idx] = ret
                            in_working[executor] = True

        # Main logic
        scope = self.conf["scope"]
        total_num = packages["total_num"]
        all_executors = list(scope.keys())
        edges_flatten = {
            k: v
            for edges in self.conf["edge"].values()
            for edge in edges
            for k, v in edge.items()
        }
        node_linkers = nx.get_edge_attributes(graph_compose, "linker")

        action_list, end_time, in_working = initialize_action_list(scope, total_num)

        if self.warpping:
            self._prepare_warpping(env)

        if not generate_nodes(graph_compose, self.conf["node"]):
            return None

        # After node initialization, check if env.affordance_datas contains updated initial value for each executor.
        for executor in scope.keys():
            init_node_name = get_init_affordance(executor)
            if (
                not hasattr(env, "affordance_datas")
                or init_node_name not in env.affordance_datas
            ):
                log_warning(
                    f"Executor '{executor}': init_node_name '{init_node_name}' not found in env.affordance_datas. Skipping initial value update."
                )
                continue
            affordance_init = env.affordance_datas[init_node_name]
            affordance_init = np.asarray(affordance_init)
            action_init_slice = action_list[executor][:, 0]
            if affordance_init.shape != action_init_slice.shape:
                log_warning(
                    f"Executor '{executor}': affordance_init shape {affordance_init.shape} does not match action_list[executor][:, 0] shape {action_init_slice.shape}. Skipping initial value update."
                )
                continue
            if not np.allclose(action_init_slice, affordance_init):
                log_info(
                    f"Updated initial value for executor '{executor}' in action_list from affordance_datas['{init_node_name}']."
                )
                action_list[executor][:, 0] = affordance_init

        generate_edges(total_num, all_executors, edges_flatten, node_linkers)

        return action_list


def attach_node_and_edge(
    cls: ActionBank, functions_dict: Dict[str, Dict[str, Callable]]
) -> ActionBank:
    for tag, funcs in functions_dict.items():
        tag_function = get_func_tag(tag)
        for func_name, func in funcs.items():
            setattr(cls, func_name, staticmethod(func))

            class_name = cls.__name__
            if class_name in tag_function.functions.keys():
                tag_function.functions[class_name].update({func_name: func})
            else:
                tag_function.functions.update({class_name: {func_name: func}})
    return cls


def attach_action_bank(cls, action_bank: ActionBank, **kwargs):
    def set_attr_for_cls(cls, attr_name: str, attr_value: Any):
        if hasattr(cls, attr_name):
            getattr(cls, attr_name).append(attr_value)
        else:
            setattr(cls, attr_name, [attr_value])

    action_config = kwargs.get("action_config", None)
    if action_config is None:
        log_error(
            f"The action config is None, but it's needed for Env: {type(cls).__name__}, Task Type: {cls.metadata['task_type']}."
        )
    set_attr_for_cls(cls, "action_banks", action_bank(action_config))

    vis_graph = kwargs.get("vis", False)
    graph_compose, jobs_data, jobkey2index = cls.action_banks[-1].parse_network(
        get_func_tag("node").functions[cls.action_banks[-1].__class__.__name__],
        get_func_tag("edge").functions[cls.action_banks[-1].__class__.__name__],
        vis_graph=vis_graph,
    )

    vis_gantt = kwargs.get("vis", False)
    package = cls.action_banks[-1].gantt(jobs_data, jobkey2index, vis=vis_gantt)

    set_attr_for_cls(cls, "packages", package)
    set_attr_for_cls(cls, "graph_composes", graph_compose)

    return cls


def get_xpos_name(affordance_name: str) -> str:
    if affordance_name.find("qpos") == -1:
        affordance_xpos_name = affordance_name + "_xpos"
    else:
        affordance_xpos_name = affordance_name.replace("qpos", "xpos")
    return affordance_xpos_name


def get_control_part(env, agent_uid):
    control_parts = env.metadata["dataset"]["robot_meta"].get("control_parts", [])

    if agent_uid in control_parts:
        return agent_uid
    else:
        return data_key_to_control_part(
            robot=env.robot,
            control_parts=control_parts,
            data_key=agent_uid,
        )


def generate_trajectory_qpos(
    env,
    agent_uid: str,
    trajectory: Dict[str, np.ndarray],
    trajectory_id: str,
    gather_index: List[int],
    trajectory_index: int,
    affordance_name: str,
    slaver: str = "",
    canonical_trajectory: List[float] = None,
    canonical_trajectory_index: int = None,
    canonical_pose: List[float] = [],
    vis: bool = False,
) -> bool:
    affordance_xpos_name = get_xpos_name(affordance_name)

    current_qpos = torch.as_tensor(trajectory[trajectory_id])[trajectory_index][
        None, gather_index
    ]  # TODO: only for 1 env
    control_part = get_control_part(env, agent_uid)
    try:
        affordance_xpos = env.robot.compute_fk(
            torch.as_tensor(current_qpos),
            control_part,
            to_matrix=True,
        )
    except RuntimeError as e:
        log_warning(f"control part {control_part} has no solver.")
        affordance_xpos = torch.zeros((1, 4, 4), device=current_qpos.device)
    if slaver != "":
        assert canonical_trajectory is not None
        assert canonical_trajectory_index is not None
        assert (
            len(canonical_pose) == 4
        ), f"canonical_pose should be a 4x4 matrix, but got {len(canonical_pose)} elements."
        canonical_pose = torch.as_tensor(
            canonical_pose,
            device=affordance_xpos.device,
            dtype=affordance_xpos.dtype,
        ).reshape(1, 4, 4)
        can_affordance_xpos = env.robot.compute_fk(
            torch.as_tensor(canonical_trajectory)[canonical_trajectory_index][
                gather_index
            ],
            get_control_part(env, agent_uid),
            to_matrix=True,
        )
        can_obj_xpos = canonical_pose
        obj_xpos = env.sim.get_asset(slaver).get_local_pose(to_matrix=True)
        affordance_xpos = torch.bmm(
            obj_xpos, torch.bmm(pose_inv(can_obj_xpos), can_affordance_xpos)
        )
        control_part = get_control_part(env, agent_uid)
        qpos_seed = env.robot.get_qpos()[:, env.robot.get_joint_ids(name=control_part)]
        ret, current_qpos = env.robot.compute_ik(
            affordance_xpos, qpos_seed, control_part
        )
        ret = ret.all().item()
        if not ret:
            log_warning(
                f"IK failed for slaver {slaver} with xpos {affordance_xpos}. Using the previous qpos instead."
            )
            return False

    if vis:
        env.sim.draw_marker(
            cfg=MarkerCfg(
                marker_type="axis",
                axis_xpos=affordance_xpos,
                axis_size=0.002,
                axis_len=0.005,
            )
        )
    # TODO: only support 1 env numpy now
    current_qpos = current_qpos.squeeze(0).cpu().numpy()
    affordance_xpos = affordance_xpos.squeeze(0).cpu().numpy()

    env.affordance_datas[affordance_name] = current_qpos
    env.affordance_datas[affordance_xpos_name] = affordance_xpos
    return True


def modify_action_config_edges(
    action_config: Dict,
    duration_updates: Dict[str, int] = None,
    trajectory_updates: Dict[str, List] = None,
    analytic_planner: bool = False,
) -> Dict:
    """
    Modify the action configuration by updating the duration and trajectory of edges.

    This function iterates through all edges in the action configuration and applies updates to their
    duration and trajectory based on the provided mappings. If `analytic_planner` is enabled, the edge
    name is set to "plan_trajectory".

    Args:
        action_config (Dict): The original action configuration.
        duration_updates (Dict[str, int], optional): A mapping of edge names to their new durations.
        trajectory_updates (Dict[str, List], optional): A mapping of edge names to their new trajectories.
        analytic_planner (bool, optional): If True, sets the edge name to "plan_trajectory". Defaults to False.

    Returns:
        Dict: The modified action configuration.
    """
    modified_config = deepcopy(action_config)

    # Iterate through all scopes in the action configuration
    for scope_name, scope_edges in modified_config["edge"].items():
        for edge_config in scope_edges:
            edge_name = list(edge_config.keys())[0]
            edge_data = edge_config[edge_name]
            # If analytic_planner is enabled, set the edge name to "plan_trajectory"
            if analytic_planner:
                edge_data["name"] = "plan_trajectory"

            # Update the duration if a mapping is provided
            if duration_updates and edge_name in duration_updates:
                edge_data["duration"] = duration_updates[edge_name]

            # Update the trajectory if a mapping is provided
            if trajectory_updates and edge_name in trajectory_updates:
                edge_data.setdefault("kwargs", {})  # Ensure "kwargs" exists
                edge_data["kwargs"]["trajectory"] = trajectory_updates[edge_name]

    return modified_config


def to_affordance_name(name: str) -> str:
    return name.replace("generate_", "")


def to_affordance_node_func(name: str) -> str:
    return "generate_" + name


class GeneralActionBank(ActionBank):
    @staticmethod
    @tag_edge
    def load_trajectory(
        env,
        trajectory_id: str,
        gather_index: List[int],
        keypose_timesteps: Tuple[int, int],
        raw_duration: int,
        duration: int,
        **kwargs,
    ):
        from scipy import interpolate

        f = {}
        start_t, end_t = keypose_timesteps[0], keypose_timesteps[1]
        trajectory = np.asarray(env.trajectory[trajectory_id])[:, gather_index]
        sub_trajectory = trajectory[start_t:end_t, :]
        ds_sub_trajectory = np.zeros((duration, sub_trajectory.shape[1]))
        for i in range(sub_trajectory.shape[1]):
            x = np.arange(sub_trajectory.shape[0])
            f[i] = interpolate.interp1d(x, sub_trajectory[:, i], axis=-1)
            ds_sub_trajectory[:, i] = f[i](np.linspace(0, raw_duration - 1, duration))

        return ds_sub_trajectory.T  # (D, T)

    @staticmethod
    @tag_edge
    def mimic_trajectory(
        env,
        agent_uid: str,
        raw_edge: Dict,
        raw_affordance: Dict,
        target_edge: Dict,
        vis: bool = False,
        **kwargs,
    ):

        GeneralActionBank.generate_trajectory_qpos = generate_trajectory_qpos
        if isinstance(raw_affordance, dict):
            aff_kwargs = deepcopy(raw_affordance.get("kwargs", {}))
            aff_kwargs.pop("trajectory", {})
            aff_kwargs.pop("canonical_trajectory", {})
            getattr(GeneralActionBank, raw_affordance["name"])(
                env,
                **aff_kwargs,
                trajectory=env.trajectory,
                canonical_trajectory=env.canonical_trajectory,
            )
            xpos = env.affordance_datas[
                get_xpos_name(to_affordance_name(raw_affordance["name"]))
            ]
        else:
            log_warning(
                f"raw_affordance is not a dict, but {type(raw_affordance)} and {raw_affordance}. Using it as a string name directly."
            )
            xpos = env.affordance_datas[get_xpos_name(raw_affordance)]

        # raw_trajectory = getattr(GeneralActionBank, raw_edge["name"])(
        #     env, **raw_edge.get("kwargs", {})
        # )
        # base_pose = env.agent.get_base_xpos(agent_uid)

        # import time
        # for t, temp in enumerate([env.agent.get_fk(raw_trajectory[:, i], uid=agent_uid)
        #     for i in range(raw_trajectory.shape[1])
        # ]):
        #     if t % 10 ==0:

        #         print(temp)
        # env.scene.draw_marker(cfg=MarkerCfg(
        #     marker_type="axis",
        #     axis_xpos=env.agent.get_base_xpos(agent_uid) @ temp,
        #     axis_size=0.002,
        #     axis_len=0.005
        # ))
        #         time.sleep(0.01)

        # trans = np.linalg.inv(base_pose) @ new_xpos @ np.linalg.inv(xpos) @ base_pose
        # ref_poses = [
        #     trans @ env.agent.get_fk(raw_trajectory[:, i], uid=agent_uid)
        #     for i in range(raw_trajectory.shape[1])
        # ]
        # if vis:
        # env.scene.draw_marker(cfg=MarkerCfg(
        #     marker_type="axis",
        #     axis_xpos=xpos,
        #     axis_size=0.002,
        #     axis_len=0.005
        # ))

        # for t, temp in enumerate(ref_poses):
        #     print(temp)
        #     if t % 10 ==0:
        # env.scene.draw_marker(cfg=MarkerCfg(
        #     marker_type="axis",
        #     axis_xpos=env.agent.get_base_xpos(agent_uid) @ temp,
        #     axis_size=0.002,
        #     axis_len=0.005
        # ))
        #         time.sleep(0.01)

        ref_poses = []
        target_edge["name"] = "plan_trajectory"
        return getattr(GeneralActionBank, target_edge["name"])(
            env,
            ref_poses=ref_poses,
            duration=target_edge["duration"],
            vis=vis,
            **target_edge.get("kwargs", {}),
        )

    @staticmethod
    @tag_edge
    def plan_trajectory(
        env,
        agent_uid: str,
        keypose_names: List[str],
        duration: int,
        ref_poses: List[np.ndarray] = [],
        vis: bool = False,
        **kwargs,
    ) -> np.ndarray:
        from embodichain.lab.sim.planners.motion_generator import MotionGenerator

        # Retrieve the start and end positions
        start_qpos = env.affordance_datas[keypose_names[0]]

        control_part = get_control_part(env, agent_uid)
        start_qpos = torch.as_tensor(env.affordance_datas[keypose_names[0]])[None]
        start_xpos = torch.bmm(
            env.robot.get_control_part_base_pose(control_part, to_matrix=True),
            env.robot.compute_fk(start_qpos, control_part, to_matrix=True),
        )

        end_qpos = torch.as_tensor(env.affordance_datas[keypose_names[-1]])
        end_xpos = torch.bmm(
            env.robot.get_control_part_base_pose(control_part, to_matrix=True),
            env.robot.compute_fk(end_qpos, control_part, to_matrix=True),
        )

        # TODO: only 1 env
        start_qpos = start_qpos.squeeze(0).cpu().numpy()
        start_xpos = start_xpos.squeeze(0).cpu().numpy()
        end_qpos = end_qpos.squeeze(0).cpu().numpy()
        end_xpos = end_xpos.squeeze(0).cpu().numpy()

        if vis:
            env.sim.draw_marker(
                cfg=MarkerCfg(
                    marker_type="axis",
                    axis_xpos=start_xpos,
                    axis_size=0.002,
                    axis_len=0.005,
                )
            )

            env.sim.draw_marker(
                cfg=MarkerCfg(
                    marker_type="axis",
                    axis_xpos=end_xpos,
                    axis_size=0.002,
                    axis_len=0.005,
                )
            )

        filtered_keyposes = [start_qpos, end_qpos]
        if "eef" in agent_uid:
            filtered_keyposes = [start_qpos]

        if len(filtered_keyposes) == 1 and len(ref_poses) == 0:

            ret = np.array([filtered_keyposes[0]] * duration)
        else:
            mo_gen = MotionGenerator(robot=env.robot, uid=agent_uid)

            if len(ref_poses) == 0:
                ret, _ = mo_gen.create_discrete_trajectory(
                    qpos_list=filtered_keyposes,
                    sample_num=duration,
                    qpos_seed=filtered_keyposes[0],
                    is_use_current_qpos=False,
                    **getattr(env, "planning_config", {}),
                )
            else:
                ret, _ = mo_gen.create_discrete_trajectory(
                    xpos_list=[start_xpos] + ref_poses + [end_xpos],
                    sample_num=duration,
                    is_use_current_qpos=False,
                    **getattr(env, "planning_config", {}),
                )
        if isinstance(ret, list):
            print(ret)
        return ret.T

    @staticmethod
    @tag_edge
    def execute_open(env, **kwargs):
        from embodichain.lab.gym.utils.misc import (
            mul_linear_expand,
        )

        duration = kwargs.get("duration", 1)
        expand = kwargs.get("expand", True)
        if expand:
            action = mul_linear_expand(np.array([[1.0], [0.0]]), [duration - 1])
            action = np.concatenate([action, np.array([[0.0]])]).transpose()
        else:
            action = np.zeros((1, duration))
        return action

    @staticmethod
    @tag_edge
    def execute_close(env, **kwargs):
        from embodichain.lab.gym.utils.misc import (
            mul_linear_expand,
        )

        duration = kwargs.get("duration", 1)
        expand = kwargs.get("expand", True)
        if expand:
            action = mul_linear_expand(np.array([[0.0], [1.0]]), [duration - 1])
            action = np.concatenate([action, np.array([[1.0]])]).transpose()
        else:
            action = np.ones((1, duration))
        return action


class ActionBankMimic:
    def __init__(self, action_banks: List[ActionBank], prob: float = 0.5) -> None:
        self.action_banks = action_banks
        self.keyword = "mimicable"
        self.prob = prob

    def mimic(self, id=None) -> ActionBank:
        if len(self.action_banks) == 1:
            return self.action_banks[0]

        if id is None:
            id = np.random.randint(len(self.action_banks))
        assert id < len(
            self.action_banks
        ), f"Invalid id {id}, should be less than {len(self.action_banks)}"

        acb = self.action_banks[id]
        ret_acb = deepcopy(acb)
        node_names = acb.get_node_names(bool_attr_name=self.keyword)

        ret_node_grap2id = ret_acb.graph2id()
        edge_need_modify = {}
        for scope in node_names.keys():
            edge_need_modify[scope] = []
            # if np.random.random() < self.prob:
            #     continue
            for node in node_names[scope]:
                mimic_id = np.random.randint(len(self.action_banks))
                action_bank_mimic_for_this_node = self.action_banks[mimic_id]
                mimic_node_names = action_bank_mimic_for_this_node.get_node_names(
                    bool_attr_name=self.keyword
                )
                temp_graph2id = action_bank_mimic_for_this_node.graph2id()

                if node in mimic_node_names[scope]:
                    ret_acb.conf["node"][scope][ret_node_grap2id[scope][node]][node] = (
                        deepcopy(
                            action_bank_mimic_for_this_node.conf["node"][scope][
                                temp_graph2id[scope][node]
                            ][node]
                        )
                    )

                    edges = ret_acb.get_edge_names(node_name=node)
                    for edge in edges[scope]:
                        edge.update({"mimic_id": mimic_id})
                    edge_need_modify[scope].extend(edges[scope])
                else:
                    log_warning(
                        f"Node {node} in scope {scope} not found in action bank {mimic_id} [{mimic_node_names[scope]}] to mimic."
                    )
            edge_need_modify[scope] = {
                v["name"]: v for v in edge_need_modify[scope]
            }.values()

        raw_edge_grap2id = acb.graph2id("edge")
        raw_node_grap2id = acb.graph2id("node")
        ret_edge_grap2id = ret_acb.graph2id("edge")
        for scope in node_names.keys():
            for edge in edge_need_modify[scope]:
                edge_name = edge["name"]
                mimic_id = edge.pop("mimic_id")
                mimic_grap2id = self.action_banks[mimic_id].graph2id("edge")
                ret_acb.conf["edge"][scope][ret_edge_grap2id[scope][edge_name]][
                    edge_name
                ]["name"] = "mimic_trajectory"
                ret_acb.conf["edge"][scope][ret_edge_grap2id[scope][edge_name]][
                    edge_name
                ]["duration"] = deepcopy(
                    self.action_banks[mimic_id].conf["edge"][scope][
                        mimic_grap2id[scope][edge_name]
                    ][edge_name]["duration"]
                )

                ret_acb.conf["edge"][scope][ret_edge_grap2id[scope][edge_name]][
                    edge_name
                ]["kwargs"] = {}
                ret_acb.conf["edge"][scope][ret_edge_grap2id[scope][edge_name]][
                    edge_name
                ]["kwargs"]["agent_uid"] = scope
                ret_acb.conf["edge"][scope][ret_edge_grap2id[scope][edge_name]][
                    edge_name
                ]["kwargs"]["raw_edge"] = deepcopy(
                    acb.conf["edge"][scope][raw_edge_grap2id[scope][edge_name]][
                        edge_name
                    ]
                )
                ret_acb.conf["edge"][scope][ret_edge_grap2id[scope][edge_name]][
                    edge_name
                ]["kwargs"]["raw_affordance"] = (
                    deepcopy(
                        acb.conf["node"][scope][raw_node_grap2id[scope][node]][node]
                    )
                    if node in raw_node_grap2id[scope]
                    else node
                )
                ret_acb.conf["edge"][scope][ret_edge_grap2id[scope][edge_name]][
                    edge_name
                ]["kwargs"]["target_edge"] = deepcopy(
                    self.action_banks[mimic_id].conf["edge"][scope][
                        mimic_grap2id[scope][edge_name]
                    ][edge_name]
                )

        return ret_acb
