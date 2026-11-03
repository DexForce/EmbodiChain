# Dexterous manipulation with USD assets

The five tasks construct their physical scenes from USD. The source MJCF in each
archive supplies task metadata and the independent physical-model audit. Initial
states, meshes, textures, provenance and licenses are included in the same ZIP.

| Task | Archive | USD | MJCF |
|---|---|---|---|
| Fixed and continuous pen spinning | ShadowHandPen | `usd/shadow_pen.usdc` | `shadow_pen.xml` |
| Cube face turning | LeapHandCubeFaceTurn | `usd/leap_cube_top_layer.usdc` | `leap_cube_top_layer.xml` |
| Cube reorientation | LeapHandCube | `usd/M4.usdc` | `M4.xml` |
| Cube Z-axis rotation | LeapHandCube | `usd/M5.usdc` | `M5.xml` |

The archives are published under
[robot_assets](https://huggingface.co/datasets/DexForceAI/embodichain_data/tree/e1ebe53462d6f48b5961018e625b7c1304e9e83a/robot_assets)
at revision `e1ebe53462d6f48b5961018e625b7c1304e9e83a`.
The asset descriptors verify the archive content and refresh older cached ZIPs.

## Download and exercise the task interfaces

Use Python 3.11 with the project dependencies installed, a CUDA device, and a
compatible development engine build with USD scene metadata, mocap frames and
CUDA graph state preservation. Run the following commands from the repository
root. The `ppo.dexuni.yaml` configurations assign the hand and free rigid bodies
to MuJoCo. Each task runs in a separate process with four environments and eight
control steps, including a selective reset of world 0.

The expected output contains five `TASK_RESULT` records with `status: passed`.
Checks cover finite observations/rewards, output shapes and dtypes, selective
reset isolation, the solver route, CUDA graph capture and the MJCF model audit.
The script uses `EMBODICHAIN_DATA_ROOT` when set; otherwise it creates a temporary
empty asset cache. It can be stopped with Ctrl-C.

```bash
#!/usr/bin/env bash
set -euo pipefail
export EMBODICHAIN_DOWNLOAD_PREFIX=https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/e1ebe53462d6f48b5961018e625b7c1304e9e83a/
export EMBODICHAIN_DATA_ROOT="${EMBODICHAIN_DATA_ROOT:-$(mktemp -d -t manipulation-assets.XXXXXX)}"
export EMBODICHAIN_SIM_EXIT_PROCESS=0
for asset in ShadowHandPen LeapHandCubeFaceTurn LeapHandCube; do
    python -m embodichain data download --name "$asset"
done
for task in shadow_pen_fixed shadow_pen_continuous leap_cube_face_turn leap_cube_reorient leap_cube_rotate_z; do
    python - "$task" <<'PY'
import importlib
import json
import sys
from pathlib import Path
import torch
import yaml
from embodichain.learning.rl.env import build_learning_env

name = sys.argv[1]
importlib.import_module(f"embodichain_tasks.manipulation.dexterous.{name}")
config = Path("extensions/embodichain_tasks/configs/tasks/manipulation/dexterous") / name / "agents/ppo.dexuni.yaml"
trainer = yaml.safe_load(config.read_text())["trainer"]
spec = trainer["learning_env"]
torch.set_num_threads(4)
env = build_learning_env(spec["name"], num_envs=4, device=trainer["device"], seed=trainer["seed"], **spec["cfg"])
try:
    obs, _ = env.reset(seed=trainer["seed"])
    for step in range(8):
        if step == 4:
            previous = {key: value[1:].clone() for key, value in obs.items()}
            obs, _ = env.reset(options={"reset_ids": torch.tensor([0], device=env.device)})
            assert all(torch.equal(obs[key][1:], value) for key, value in previous.items())
        action = .15 * torch.sin(torch.arange(env.action_dim, device=env.device)[None, :] + torch.arange(4, device=env.device)[:, None] * .13 + step * .3)
        obs, reward, terminated, truncated, _ = env.step(action)
        assert set(obs) == set(env.single_observation_space.spaces)
        for key, value in obs.items():
            assert value.shape == (4, *env.single_observation_space.spaces[key].shape)
            assert torch.isfinite(value).all()
        assert reward.shape == (4,) and torch.isfinite(reward).all()
        assert terminated.shape == truncated.shape == (4,)
        assert terminated.dtype == truncated.dtype == torch.bool
    assert env.backend.solver.features.backend == "pure_mujoco"
    assert env.backend._graph is not None
    assert env.model_audit["physical_model"]["status"] == "passed"
    print("TASK_RESULT " + json.dumps({"task": name, "status": "passed", "num_envs": 4, "control_steps": 8, "usd": str(env.usd), "mjcf": str(env.xml), "solver": "pure_mujoco", "cuda_graph": True, "selective_reset": "passed", "model_audit": "passed"}), flush=True)
finally:
    env.close()
PY
done
```

## Training configurations

Each task's `agents/ppo.dexuni.yaml` selects all-MuJoCo DexUni;
`agents/ppo.mujoco_warp.yaml` selects Newton/MJWarp. For example:

```bash
python -m embodichain train-rl --config extensions/embodichain_tasks/configs/tasks/manipulation/dexterous/shadow_pen_fixed/agents/ppo.dexuni.yaml
```

The short interface check above exercises loading, reset and stepping. Sustained
training, checkpoint resume and policy evaluation have separate validation results.
