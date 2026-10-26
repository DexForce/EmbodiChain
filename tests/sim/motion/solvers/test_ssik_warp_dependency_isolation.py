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

"""Prove the four public Warp routes do not depend on installed official SSIK."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "robot,variant,part,mode,search",
    [
        ("AlohaMiniCfg", {}, "left_arm", "strict", "uniform"),
        ("TianjiMarvinCfg", {"with_gripper": True}, "left_arm", "strict", "seed_local"),
        ("ARX5Cfg", {}, "arm", "near_parallel", "uniform"),
        ("FlexivRizon4SCfg", {}, "arm", "rr", "seed_local"),
    ],
)
def test_four_warp_routes_are_independent_of_official_ssik_in_fresh_process(
    robot: str,
    variant: dict[str, bool],
    part: str,
    mode: str,
    search: str,
) -> None:
    """Cold import, actual-model IK and failures must not touch optional SSIK."""
    case = {
        "robot": robot,
        "variant": variant,
        "part": part,
        "mode": mode,
        "search": search,
    }
    program = r"""
import importlib.abc
import json
import sys

class NoOfficialSSIK(importlib.abc.MetaPathFinder):
    simulate_missing = False
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "ssik" or fullname.startswith("ssik."):
            if self.simulate_missing:
                raise ModuleNotFoundError("Optional official SSIK is hidden", name=fullname)
            raise AssertionError("Core Warp tried to import optional official SSIK: " + fullname)

blocker = NoOfficialSSIK()
sys.meta_path.insert(0, blocker)
case = json.loads(sys.argv[1])
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from embodichain.lab.sim.motion.solvers import SSIKSolverCfg
import embodichain_robots
preset = getattr(embodichain_robots, case["robot"]).from_dict(case["variant"])
source = preset.solver_cfg[case["part"]]
chain = preset.build_pk_serial_chain(device=torch.device("cpu"))[case["part"]].to(dtype=torch.float64)
assert isinstance(source, SSIKSolverCfg), "Official arm defaults must use Warp SSIK"
configuration = source.copy()
assert configuration.backend == "warp" and configuration.geometry_mode == case["mode"]
assert getattr(configuration, "rr_search" if case["mode"] == "rr" else "srs_search") == case["search"]
configuration.urdf_path = preset.fpath
configuration.joint_names = list(chain.get_joint_parameter_names())
restored = SSIKSolverCfg.from_dict(configuration.to_dict())
assert restored.backend == "warp" and restored.geometry_mode == case["mode"]
solver = restored.init_solver(device="cpu", pk_serial_chain=chain)
limits = torch.tensor(chain.get_joint_limits(), dtype=torch.float64)
phase = torch.arange(len(limits[0]), dtype=torch.float64) + .37
joints = (limits.mean(0) + .08 * (limits[1] - limits[0]) * phase.sin())[None].float()
tcp = torch.as_tensor(source.tcp, dtype=torch.float64)
goal = (chain.forward_kinematics(joints.double()).get_matrix() @ tcp).float()
valid, result = solver.get_ik(goal, qpos_seed=joints)
assert valid.shape == (1,) and result.shape == joints.shape and bool(valid.all())
assert result.device.type == "cpu" and torch.isfinite(result).all()
assert ((result.double() >= limits[0]) & (result.double() <= limits[1])).all()
actual = chain.forward_kinematics(result.double()).get_matrix() @ tcp
assert torch.linalg.vector_norm(actual[:, :3, 3] - goal.double()[:, :3, 3], dim=-1).max() <= 1e-5
relative = actual[:, :3, :3] @ goal.double()[:, :3, :3].transpose(-1, -2)
skew = torch.stack((relative[:, 2, 1] - relative[:, 1, 2], relative[:, 0, 2] - relative[:, 2, 0], relative[:, 1, 0] - relative[:, 0, 1]), dim=-1)
angle = torch.atan2(torch.linalg.vector_norm(skew, dim=-1), relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1)
assert angle.abs().max() <= 1e-5
unreachable = goal.clone()
unreachable[:, 0, 3] = 100
failed, hold = solver.get_ik(unreachable, qpos_seed=result)
assert not bool(failed.any())
torch.testing.assert_close(hold, result, rtol=0, atol=0)
assert "embodichain.lab.sim.motion.solvers._ssik_reference" not in sys.modules
assert not any(name == "ssik" or name.startswith("ssik.") for name in sys.modules)
if case["robot"] == "AlohaMiniCfg":
    # Only explicit reference selection may touch the optional dependency,
    # and its absence must produce an actionable error rather than fallback.
    blocker.simulate_missing = True
    reference = configuration.copy()
    reference.backend = "reference"
    try:
        reference.init_solver(device="cpu", pk_serial_chain=chain)
    except ImportError as error:
        assert "requirements/ssik-reference.txt" in str(error)
    else:
        raise AssertionError("Explicit reference unexpectedly worked without official SSIK")
print(json.dumps({"passed": True, "mode": case["mode"], "device": result.device.type}), flush=True)
"""
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = "1"
    completed = subprocess.run(
        [sys.executable, "-u", "-c", program, json.dumps(case)],
        capture_output=True,
        text=True,
        check=False,
        timeout=240,
        env=environment,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert '"passed": true' in completed.stdout


def test_explicit_warp_rr_rejects_a_different_joint_prefix() -> None:
    import pytorch_kinematics as pk
    from embodichain.lab.sim.motion.solvers import SSIKSolverCfg

    links = "".join(f'<link name="link{i}"/>' for i in range(8))
    joints = "".join(
        f'<joint name="j{i}" type="revolute"><parent link="link{i}"/>'
        f'<child link="link{i+1}"/><origin xyz=".12 .03 .05"/>'
        '<axis xyz="0 0 1"/><limit lower="-2" upper="2" effort="1" velocity="1"/></joint>'
        for i in range(7)
    )
    chain = pk.build_serial_chain_from_urdf(
        f'<robot name="different">{links}{joints}</robot>', "link7"
    )
    with pytest.raises(ValueError, match="joint prefix|match.*model|model.*match"):
        SSIKSolverCfg(
            backend="warp", geometry_mode="rr", rr_search="seed_local"
        ).init_solver(device="cpu", pk_serial_chain=chain)
