# Environment Task Replay - Async + Dedicated Buffer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evolve PR #425's replay feature to support async parallel envs via a dedicated per-env `_traj_buffer` (states + pre-process action), per-env step counters, default auto-save to `~/.cache/embodichain_data/trajectories/{run_id}/`, and multi-env replay with per-env lengths.

**Architecture:** A dedicated `self._traj_buffer` (TensorDict: `states` + `actions`, `[num_envs, max_steps]`) replaces `rollout_buffer["states"]`. A per-env `self._traj_steps` tensor replaces the global `current_rollout_step` for trajectory recording (only reset envs' counters clear). The pre-process action is stashed in `_preprocess_action` and recorded per-env, so dynamic replay feeds it back through `env.step` and the `ActionManager` re-applies the transform. Auto-save fires at episode end + `close()` to the data cache dir. `ReplayWrapper` replays per-env lengths. `current_rollout_step` / `rollout_buffer` / LeRobot / RL are untouched.

**Tech Stack:** Python, PyTorch, TensorDict, Gymnasium, DexSim, `@configclass`, pytest.

## Global Constraints

- Package import root is `embodichain` (lowercase, one word); repo folder is `EmbodiChain`.
- New source files start with the Apache 2.0 header; existing files keep theirs. `from __future__ import annotations` at top.
- Run `black .` (black==26.3.1) before every commit; `/pre-commit-check` for CI.
- Full type hints; prefer `A | B`; `__all__` in public modules.
- Real-sim tests (import `SimulationManager`) are auto-marked `requires_sim` by `tests/conftest.py`. Standalone scripts that build multiple envs must set `os.environ["EMBODICHAIN_SIM_EXIT_PROCESS"] = "0"`. Teardown: `env.close()` + `SimulationManager.flush_cleanup_queue()` + `gc.collect()`.
- `EMBODICHAIN_DEFAULT_DATA_ROOT` (`~/.cache/embodichain_data`, env-configurable) from `embodichain/data/constants.py` is the default save root.
- Scope: RL `"rl"`-mode buffer, `current_rollout_step`, `rollout_buffer` (obs/actions/rewards), LeRobot recorder, RL reward functors - all untouched. `qvel` deferred.

## File Structure

- **Modify** `embodichain/lab/gym/utils/gym_utils.py` - rename `build_trajectory_states_buffer` -> `build_trajectory_buffer` (add `actions` field from `action_space`); `load_trajectory` unchanged (already permissive).
- **Modify** `embodichain/lab/gym/envs/embodied_env.py` - cfg fields; `__init__` creates `_traj_buffer`/`_traj_steps`; `_preprocess_action` stashes raw action; `_write_trajectory_step` per-env; `_initialize_episode` per-env reset + auto-save; `save_trajectory(path, env_ids)`; `_save_trajectory_for_env`; `close` finalize.
- **Modify** `embodichain/lab/gym/envs/wrapper/replay.py` - per-env `_replay_steps`/`_lengths`; pre-process action for dynamic.
- **Modify** `tests/gym/envs/test_replay.py` - update existing tests + add async/auto-save/ActionManager/multi-env-length tests.
- **Modify** `tests/gym/utils/test_gym_utils.py` - update builder tests for new name + `actions`.
- `embodichain/lab/gym/envs/base_env.py` - UNCHANGED (auto-reset guard stays).

---

### Task 1: ReplayWrapper per-env + pre-process action (format-compatible)

**Files:**
- Modify: `embodichain/lab/gym/envs/wrapper/replay.py`
- Test: `tests/gym/envs/test_replay.py`

**Interfaces:**
- Consumes: trajectory `.pt` with `{"states", "actions", "meta"}`; `meta["lengths"]` optional (defaults to `num_steps` per env for backward compat with PR #425 files).
- Produces: `ReplayWrapper` with per-env `_replay_steps` (`torch.Tensor[num_envs]`), `_lengths` (`torch.Tensor[num_envs]`); `step` returns per-env `truncated` tensor.

**Context:** This task refactors `ReplayWrapper` to per-env replay and pre-process action, but is **bi-directionally format-compatible**: it reads both PR #425 files (no `lengths` -> uniform `num_steps`) and the new format (with `lengths`). Recording stays PR #425 (unchanged) in this task, so existing record/save tests still produce files the new replay can read. The recording+save switch to the new format happens in Task 2.

- [ ] **Step 1: Write the failing tests**

Append to `tests/gym/envs/test_replay.py`:

```python
def test_replay_reads_legacy_file_without_lengths(tmp_path):
    """PR #425 files (no meta["lengths"]) replay as uniform-length (backward compat)."""
    env = ReplayTestEnv(record_trajectory=True, num_envs=2, device="cpu")
    try:
        env.reset()
        _drive(env, num_steps=4)
        path = tmp_path / "legacy.pt"
        env.save_trajectory(str(path))
    finally:
        env.close()
        SimulationManager.flush_cleanup_queue()
        gc.collect()

    # Strip "lengths" to simulate a PR #425 file.
    data = torch.load(path, weights_only=False)
    data["meta"].pop("lengths", None)
    torch.save(data, path)

    env2 = ReplayTestEnv(record_trajectory=False, num_envs=2, device="cpu")
    env2 = ReplayWrapper(env2, str(path), mode="kinematic")
    try:
        env2.reset()
        trunc_all = torch.zeros(2, dtype=torch.bool)
        for _ in range(4):
            _, _, _, trunc, _ = env2.step(None)
            trunc_all = trunc_all | trunc
        assert bool(trunc_all.all())  # all envs done after num_steps
    finally:
        env2.close()
        SimulationManager.flush_cleanup_queue()
        gc.collect()


def test_replay_respects_per_env_lengths(tmp_path):
    """Replay truncates each env at its own recorded length."""
    env = ReplayTestEnv(record_trajectory=True, num_envs=2, device="cpu")
    try:
        env.reset()
        _drive(env, num_steps=5)
        path = tmp_path / "traj.pt"
        env.save_trajectory(str(path))
    finally:
        env.close()
        SimulationManager.flush_cleanup_queue()
        gc.collect()

    # Override lengths: env0 -> 3 steps, env1 -> 5 steps.
    data = torch.load(path, weights_only=False)
    data["meta"]["lengths"] = [3, 5]
    torch.save(data, path)

    env2 = ReplayTestEnv(record_trajectory=False, num_envs=2, device="cpu")
    env2 = ReplayWrapper(env2, str(path), mode="kinematic")
    try:
        env2.reset()
        trunc = torch.zeros(2, dtype=torch.bool)
        for _ in range(5):
            _, _, _, t, _ = env2.step(None)
            trunc = trunc | t
        assert bool(trunc[0]) and bool(trunc[1])  # both eventually done
        # env0 finishes at step 3, env1 at step 5
    finally:
        env2.close()
        SimulationManager.flush_cleanup_queue()
        gc.collect()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/gym/envs/test_replay.py::test_replay_reads_legacy_file_without_lengths tests/gym/envs/test_replay.py::test_replay_respects_per_env_lengths -v`
Expected: FAIL (`ReplayWrapper` still uses scalar `_idx`/`_num_steps`; per-env lengths ignored).

- [ ] **Step 3: Refactor ReplayWrapper to per-env**

Replace the body of `ReplayWrapper.__init__` (after `self._trajectory = load_trajectory(trajectory)`) and the `_expand_to_env_count`, `reset`, `step` methods in `embodichain/lab/gym/envs/wrapper/replay.py`. Keep `_set_all_states`, `close` unchanged. The new `__init__` (replacing the block from `self._num_steps = ...` through `self._expand_to_env_count()`):

```python
        self._mode = mode
        self._trajectory = load_trajectory(trajectory)
        meta = self._trajectory["meta"]

        # Sanity-check that the trajectory matches the replay env's robot.
        traj_robot_dof = int(meta.get("robot_dof", self.env.robot.dof))
        traj_active_joint_ids = list(meta.get("active_joint_ids", []))
        env_robot_dof = int(self.env.robot.dof)
        env_active_joint_ids = list(self.env.active_joint_ids)
        if (
            traj_robot_dof != env_robot_dof
            or traj_active_joint_ids != env_active_joint_ids
        ):
            raise ValueError(
                f"Trajectory was recorded with robot_dof={traj_robot_dof} / "
                f"active_joint_ids={traj_active_joint_ids} but replay env has "
                f"robot_dof={env_robot_dof} / active_joint_ids={env_active_joint_ids}."
            )

        self._expand_to_env_count()

        # Per-env lengths: fall back to uniform num_steps for legacy files.
        num_envs = int(self._trajectory["meta"]["num_envs"])
        lengths = meta.get("lengths", [int(meta["num_steps"])] * num_envs)
        self._lengths = torch.tensor(lengths, dtype=torch.long, device=self.env.device)

        # Clamp replay length to the wrapped env's horizon.
        max_steps = int(self.env.max_episode_steps)
        if bool((self._lengths > max_steps).any()):
            logger.log_warning(
                f"Trajectory lengths exceed env max_episode_steps={max_steps}; clamping."
            )
            self._lengths = self._lengths.clamp(max=max_steps)
        self._replay_steps = torch.zeros(
            self.env.num_envs, dtype=torch.long, device=self.env.device
        )
```

New `_expand_to_env_count` (adds lengths broadcast):

```python
    def _expand_to_env_count(self) -> None:
        """Broadcast a single-env trajectory to the wrapped env's env count."""
        meta = self._trajectory["meta"]
        traj_envs = int(meta["num_envs"])
        env_envs = int(self.env.num_envs)
        if traj_envs == env_envs:
            return
        if traj_envs != 1:
            raise ValueError(
                f"Trajectory has {traj_envs} envs but wrapped env has {env_envs}; "
                "only single-env trajectories can be broadcast."
            )
        for key in ("states", "actions"):
            t = self._trajectory[key]
            self._trajectory[key] = t.expand(env_envs, *t.shape[1:]).clone()
        meta["num_envs"] = env_envs
        if "lengths" in meta:
            meta["lengths"] = meta["lengths"] * env_envs
```

New `reset` (per-env `_replay_steps`):

```python
    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[EnvObs, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        self.env.sim.enable_physics(False)
        self._set_all_states(self._trajectory["states"][:, 0])
        if self._mode == "dynamic":
            self.env.sim.enable_physics(True)
            self.env._replay_no_auto_reset = True
        self._replay_steps = torch.zeros(
            self.env.num_envs, dtype=torch.long, device=self.env.device
        )
        return self.env.get_obs(), info
```

New `step` (per-env clamp + trunc; dynamic feeds `actions` through `env.step` so ActionManager re-preprocesses):

```python
    def step(
        self, action: Any
    ) -> tuple[EnvObs, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        env = self.env
        n = env.num_envs
        idx = torch.arange(n, device=env.device)
        st = self._replay_steps.clamp(max=self._lengths - 1)  # finished envs hold last

        if self._mode == "kinematic":
            self._set_all_states(self._trajectory["states"][idx, st])
            env.sim.update(env.sim_cfg.physics_dt, env.cfg.sim_steps_per_control)
            obs = env.get_obs()
            self._replay_steps = (self._replay_steps + 1).clamp(max=self._lengths)
            trunc = self._replay_steps >= self._lengths
            return (
                obs,
                torch.zeros(n, device=env.device),
                torch.zeros(n, dtype=torch.bool, device=env.device),
                trunc,
                {},
            )

        # dynamic: feed the recorded (pre-process) action; env.step re-preprocesses.
        action_t = self._trajectory["actions"][idx, st]
        obs, reward, term, trunc, info = env.step(action_t)
        self._replay_steps = (self._replay_steps + 1).clamp(max=self._lengths)
        trunc = trunc | (self._replay_steps >= self._lengths)
        return obs, reward, term, trunc, info
```

- [ ] **Step 4: Run the full replay suite**

Run: `pytest tests/gym/envs/test_replay.py -v`
Expected: PASS (all existing tests + 2 new). Existing tests pass because per-env replay of a uniform-length trajectory is behavior-identical to the old scalar `_idx`.

- [ ] **Step 5: Commit**

```bash
black .
git add embodichain/lab/gym/envs/wrapper/replay.py tests/gym/envs/test_replay.py
git commit -m "refactor(replay): per-env replay steps + pre-process action (format-compatible)"
```

---

### Task 2: Dedicated `_traj_buffer` + per-env recording + save (new format)

**Files:**
- Modify: `embodichain/lab/gym/utils/gym_utils.py` - rename builder, add `actions` field.
- Modify: `embodichain/lab/gym/envs/embodied_env.py` - cfg fields; `__init__` creates `_traj_buffer`/`_traj_steps`; `_preprocess_action` stash; `_write_trajectory_step` per-env; `save_trajectory(path, env_ids)`; `_initialize_episode` per-env reset; remove `rollout_buffer["states"]` + `_write_trajectory_states`.
- Modify: `tests/gym/utils/test_gym_utils.py` - update builder tests.
- Modify: `tests/gym/envs/test_replay.py` - update record/save tests for `_traj_buffer` + `lengths`.

**Interfaces:**
- Consumes: `build_trajectory_buffer(env, max_steps, num_envs, device, uids, action_space)` (this task defines it); `ReplayWrapper` (Task 1, reads new format).
- Produces: `EmbodiedEnv._traj_buffer` (TensorDict `{states, actions}`), `EmbodiedEnv._traj_steps` (Tensor `[num_envs]`), `EmbodiedEnv.save_trajectory(path, env_ids=None) -> str`, `EmbodiedEnvCfg.trajectory_save_dir`, `EmbodiedEnvCfg.trajectory_auto_save`.

- [ ] **Step 1: Rename builder + add `actions` field**

In `embodichain/lab/gym/utils/gym_utils.py`, rename `build_trajectory_states_buffer` to `build_trajectory_buffer`, add an `action_space` parameter, and add an `actions` buffer (Box action space; `action_space` is the **batched** space with `shape[0] == num_envs`). Replace the function signature and the `return` line:

```python
def build_trajectory_buffer(
    env,
    max_steps: int,
    num_envs: int,
    device: str | torch.device,
    uids: list[str] | None = None,
    action_space=None,
) -> TensorDict:
```

... (docstring updated to mention `action_space` and the returned `actions` field) ...

At the end, replace `return TensorDict(states, ...)` with:

```python
    td: dict = {"states": TensorDict(states, batch_size=[num_envs, max_steps], device=device)}
    if action_space is not None and hasattr(action_space, "shape"):
        # action_space is the batched space (shape[0] == num_envs).
        action_shape = tuple(action_space.shape[1:])
        td["actions"] = torch.zeros(
            (num_envs, max_steps, *action_shape), dtype=torch.float32, device=device
        )
    return TensorDict(td, batch_size=[num_envs, max_steps], device=device)
```

Update `tests/gym/utils/test_gym_utils.py`: rename imports and test names from `build_trajectory_states_buffer` to `build_trajectory_buffer`; add an `action_space` arg (use `gymnasium.spaces.Box(low=-1, high=1, shape=(num_envs, 6), dtype=np.float32)`), and assert `buf["actions"].shape == (num_envs, max_steps, 6)` in the shape test.

- [ ] **Step 2: Run builder tests to verify they fail/pass**

Run: `pytest tests/gym/utils/test_gym_utils.py -v -k "build_trajectory_buffer or load_trajectory"`
Expected: PASS after the rename + action assertion.

- [ ] **Step 3: Add cfg fields**

In `embodichain/lab/gym/envs/embodied_env.py`, after the `trajectory_uids` field (line ~214), add:

```python
    trajectory_save_dir: str | None = None
    """Directory for auto-saved trajectories. Defaults to
    ``<EMBODICHAIN_DEFAULT_DATA_ROOT>/trajectories/{run_id}/``."""

    trajectory_auto_save: bool = True
    """If True (and record_trajectory is True), auto-save each env's trajectory to
    ``trajectory_save_dir`` at episode end and on close()."""
```

- [ ] **Step 4: Update import + `__init__` to build `_traj_buffer`**

In `embodied_env.py`, update the import (lines 53-56):

```python
from embodichain.lab.gym.utils.gym_utils import (
    build_trajectory_buffer,
    init_rollout_buffer_from_gym_space,
)
```

Replace the `__init__` trajectory block (the `if self.cfg.record_trajectory: self.cfg.init_rollout_buffer = True` at line ~287 AND the `if self.cfg.record_trajectory: self.rollout_buffer["states"] = build_trajectory_states_buffer(...)` block at lines ~313-320) with:

```python
        # Dedicated per-env trajectory buffer (states + actions). Decoupled from
        # rollout_buffer so async parallel envs and ActionManager are supported.
        self._traj_buffer: TensorDict | None = None
        self._traj_steps: torch.Tensor | None = None
        self._traj_raw_action = None
        self._traj_save_count = 0
        from datetime import datetime

        self._traj_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.cfg.record_trajectory:
            self._traj_buffer = build_trajectory_buffer(
                env=self,
                max_steps=self.max_episode_steps,
                num_envs=self.num_envs,
                device=self.device,
                uids=self.cfg.trajectory_uids,
                action_space=self.action_space,
            )
            self._traj_steps = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
```

(Remove the old `if self.cfg.record_trajectory: self.cfg.init_rollout_buffer = True` line - the trajectory no longer needs `rollout_buffer`.)

- [ ] **Step 5: Stash raw action in `_preprocess_action`**

Replace `_preprocess_action` (lines ~908-912):

```python
    def _preprocess_action(self, action: EnvAction) -> EnvAction:
        """Delegate to ActionManager when configured; stash raw action for trajectory."""
        if self._traj_buffer is not None:
            self._traj_raw_action = action
        if self.action_manager is not None:
            return self.action_manager.process_action(action, mode="pre")
        return super()._preprocess_action(action)
```

- [ ] **Step 6: Replace `_write_trajectory_states` with per-env `_write_trajectory_step`**

In `_write_episode_rollout_step`, **remove** the final `self._write_trajectory_states()` call (line ~654). Then replace the `_write_trajectory_states` method (lines ~656-684) with:

```python
    def _write_trajectory_step(self) -> None:
        """Write one step of per-env ``states`` + pre-process ``action`` into ``_traj_buffer``."""
        if self._traj_buffer is None:
            return
        max_steps = self._traj_buffer.shape[1]
        env_idx = torch.arange(self.num_envs, device=self.device)
        step = self._traj_steps
        mask = step < max_steps
        if not bool(mask.any()):
            self._traj_steps = (self._traj_steps + 1).clamp(max=max_steps)
            return
        idx = env_idx[mask]
        st = step[mask]
        slot = self._traj_buffer["states"][idx, st]
        slot["robot"]["root_pose"].copy_(self.robot.get_local_pose()[idx])
        slot["robot"]["qpos"].copy_(self.robot.get_qpos()[idx])
        if "articulations" in slot.keys():
            for uid, art in self.sim._articulations.items():
                if uid in slot["articulations"].keys():
                    slot["articulations"][uid]["root_pose"].copy_(art.get_local_pose()[idx])
                    slot["articulations"][uid]["qpos"].copy_(art.get_qpos()[idx])
        if "rigid_objects" in slot.keys():
            for uid, obj in self.sim._rigid_objects.items():
                if uid in slot["rigid_objects"].keys():
                    slot["rigid_objects"][uid]["pose"].copy_(obj.get_local_pose()[idx])
        if self._traj_raw_action is not None:
            self._traj_buffer["actions"][idx, st].copy_(self._traj_raw_action[idx])
        self._traj_steps = (self._traj_steps + 1).clamp(max=max_steps)
```

- [ ] **Step 7: Call `_write_trajectory_step` from `_hook_after_sim_step`**

In `_hook_after_sim_step` (lines ~470-498), after the `if self.rollout_buffer is not None:` block (which increments `current_rollout_step`), add a call outside that block (so it runs even when `rollout_buffer` is None):

```python
        self._write_trajectory_step()
```

(Place it just before the `# Update success status ...` comment at the end of `_hook_after_sim_step`.)

- [ ] **Step 8: Per-env reset in `_initialize_episode`**

In `_initialize_episode`, right after the `if self.rollout_buffer is not None and self._rollout_buffer_mode != "rl": self.current_rollout_step = 0` block (line ~583-584), add:

```python
        if self._traj_steps is not None:
            self._traj_steps[env_ids_to_process] = 0
```

- [ ] **Step 9: Rewrite `save_trajectory` to read `_traj_buffer` + per-env `lengths`**

Replace `save_trajectory` (lines ~1120-1148) with:

```python
    def save_trajectory(
        self, path: str, env_ids: Sequence[int] | None = None
    ) -> str:
        """Save recorded trajectory (states + actions) to a ``.pt`` file.

        Args:
            path: Destination ``.pt`` file path.
            env_ids: Env indices to save (default: all). Each saved env's actual
                recorded length is stored in ``meta["lengths"]``.

        Raises:
            RuntimeError: If trajectory recording was never enabled.
        """
        if self._traj_buffer is None:
            raise RuntimeError(
                "Trajectory recording is not enabled (set cfg.record_trajectory=True)."
            )
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        env_ids_t = torch.as_tensor(list(env_ids), dtype=torch.long, device=self.device)
        lengths = self._traj_steps[env_ids_t]
        max_len = int(lengths.max().item()) if len(env_ids) > 0 else 0
        sub = self._traj_buffer[env_ids_t]
        states = sub["states"][:, :max_len].clone()
        actions = sub["actions"][:, :max_len].clone()
        meta = {
            "lengths": lengths.tolist(),
            "num_steps": max_len,
            "num_envs": int(len(env_ids)),
            "dt": float(self.sim_cfg.physics_dt),
            "active_joint_ids": list(self.active_joint_ids),
            "robot_uid": self.robot.uid,
            "robot_dof": int(self.robot.dof),
            "articulation_uids": list(self.sim._articulations.keys()),
            "articulation_dofs": {
                uid: int(art.dof) for uid, art in self.sim._articulations.items()
            },
            "rigid_object_uids": list(self.sim._rigid_objects.keys()),
            "env_ids": [int(e) for e in env_ids],
        }
        torch.save({"states": states, "actions": actions, "meta": meta}, path)
        return path
```

- [ ] **Step 10: Update record/save tests for `_traj_buffer` + `lengths`**

In `tests/gym/envs/test_replay.py`:
- `test_record_trajectory_populates_states`: replace `env.rollout_buffer["states"]` with `env._traj_buffer["states"]`; replace `env.current_rollout_step == 5` with `env._traj_steps.tolist() == [5, 5]`; keep the shape + readback assertions (now indexing `env._traj_buffer["states"]["robot"]["qpos"][:, env._traj_steps[0].item() - 1]`).
- `test_save_trajectory_round_trip`: assert `data["meta"]["lengths"] == [4, 4]`; assert `data["actions"].shape == (2, 4, 6)`; assert `data["meta"]["num_envs"] == 2`.

- [ ] **Step 11: Run the full replay + gym_utils suites**

Run: `pytest tests/gym/envs/test_replay.py tests/gym/utils/test_gym_utils.py -v`
Expected: PASS (recording now uses `_traj_buffer`; ReplayWrapper from Task 1 reads the new `lengths` format).

- [ ] **Step 12: Commit**

```bash
black .
git add embodichain/lab/gym/utils/gym_utils.py embodichain/lab/gym/envs/embodied_env.py tests/gym/utils/test_gym_utils.py tests/gym/envs/test_replay.py
git commit -m "refactor(replay): dedicated _traj_buffer with per-env states + pre-process action"
```

---

### Task 3: Default auto-save to `~/.cache/embodichain_data/trajectories/{run_id}/`

**Files:**
- Modify: `embodichain/lab/gym/envs/embodied_env.py` - `_save_trajectory_for_env`, auto-save in `_initialize_episode` + `close`.
- Test: `tests/gym/envs/test_replay.py`

**Interfaces:**
- Consumes: `save_trajectory(path, env_ids)` (Task 2), `EMBODICHAIN_DEFAULT_DATA_ROOT`, `cfg.trajectory_save_dir`, `cfg.trajectory_auto_save`.
- Produces: `EmbodiedEnv._save_trajectory_for_env(env_id) -> str | None`; auto-save at episode end + close.

- [ ] **Step 1: Write the failing tests**

Append to `tests/gym/envs/test_replay.py`:

```python
def test_auto_save_at_episode_end(tmp_path):
    save_dir = tmp_path / "trajs"
    env = ReplayTestEnv(record_trajectory=True, num_envs=2, device="cpu")
    env.cfg.trajectory_save_dir = str(save_dir)
    env.cfg.trajectory_auto_save = True
    try:
        env.reset()
        _drive(env, num_steps=4)
        # Trigger an episode-end reset for env 0 only.
        env._initialize_episode(torch.tensor([0]))
    finally:
        env.close()
        SimulationManager.flush_cleanup_queue()
        gc.collect()

    files = list(save_dir.glob("*.pt"))
    assert len(files) == 1, f"expected 1 auto-saved file for env 0, got {files}"
    data = torch.load(files[0], weights_only=False)
    assert data["meta"]["env_ids"] == [0]
    assert data["meta"]["lengths"] == [4]


def test_auto_save_on_close(tmp_path):
    save_dir = tmp_path / "trajs"
    env = ReplayTestEnv(record_trajectory=True, num_envs=2, device="cpu")
    env.cfg.trajectory_save_dir = str(save_dir)
    try:
        env.reset()
        _drive(env, num_steps=3)
        env.close()
    finally:
        SimulationManager.flush_cleanup_queue()
        gc.collect()

    files = list(save_dir.glob("*.pt"))
    assert len(files) == 2  # one per in-flight env
```

(Add `import torch` already present; ensure `torch` is imported - it is.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/gym/envs/test_replay.py::test_auto_save_at_episode_end tests/gym/envs/test_replay.py::test_auto_save_on_close -v`
Expected: FAIL (no `_save_trajectory_for_env`, no auto-save).

- [ ] **Step 3: Implement `_save_trajectory_for_env` + default dir**

In `embodied_env.py`, near the top imports add:

```python
import os

from embodichain.data.constants import EMBODICHAIN_DEFAULT_DATA_ROOT
```

Add a method after `save_trajectory`:

```python
    def _save_trajectory_for_env(self, env_id: int) -> str | None:
        """Auto-save one env's trajectory to ``cfg.trajectory_save_dir`` (or default)."""
        if self._traj_buffer is None or not self.cfg.trajectory_auto_save:
            return None
        if int(self._traj_steps[env_id].item()) == 0:
            return None
        base = self.cfg.trajectory_save_dir
        if base is None:
            base = os.path.join(
                EMBODICHAIN_DEFAULT_DATA_ROOT, "trajectories", self._traj_run_id
            )
        os.makedirs(base, exist_ok=True)
        path = os.path.join(base, f"traj_env{env_id}_{self._traj_save_count:06d}.pt")
        self._traj_save_count += 1
        return self.save_trajectory(path, env_ids=[env_id])
```

- [ ] **Step 4: Auto-save at episode end in `_initialize_episode`**

In `_initialize_episode`, in the block where `current_rollout_step`/`_traj_steps` are reset (Step 8 of Task 2), auto-save the resetting envs **before** clearing their `_traj_steps`. Insert before `if self._traj_steps is not None: self._traj_steps[env_ids_to_process] = 0`:

```python
        if self._traj_buffer is not None and self.cfg.trajectory_auto_save:
            for env_id in env_ids_to_process.tolist():
                self._save_trajectory_for_env(env_id)
```

- [ ] **Step 5: Auto-save (finalize) in `close`**

Replace `close` (lines ~1150+):

```python
    def close(self) -> None:
        """Close the environment and release resources."""
        if self._traj_buffer is not None and self.cfg.trajectory_auto_save:
            for env_id in range(self.num_envs):
                self._save_trajectory_for_env(env_id)
        if self.dataset_manager:
            self.dataset_manager.finalize()
        self.sim.destroy()
```

- [ ] **Step 6: Run the auto-save + full replay suites**

Run: `pytest tests/gym/envs/test_replay.py -v`
Expected: PASS (all tests including auto-save).

- [ ] **Step 7: Commit**

```bash
black .
git add embodichain/lab/gym/envs/embodied_env.py tests/gym/envs/test_replay.py
git commit -m "feat(replay): default auto-save of trajectories to cache dir at episode end + close"
```

---

### Task 4: Async recording + ActionManager dynamic-replay tests

**Files:**
- Modify: `tests/gym/envs/test_replay.py` (add tests; may extend `ReplayTestEnv` for an ActionManager variant).

**Interfaces:**
- Consumes: the full feature (Tasks 1-3).
- Produces: behavioral tests proving async parallel envs don't corrupt recording, and dynamic replay works through an `ActionManager`.

- [ ] **Step 1: Write the async recording test**

Append to `tests/gym/envs/test_replay.py`:

```python
def test_async_envs_do_not_corrupt_recording(tmp_path):
    """env0 terminates early; env1 keeps recording without being overwritten."""
    env = ReplayTestEnv(record_trajectory=True, num_envs=2, device="cpu")
    try:
        env.reset()
        _drive(env, num_steps=3)
        # env0 "finishes" its episode at step 3 -> its counter resets to 0.
        env._initialize_episode(torch.tensor([0]))
        # env1 continues for 2 more steps; env0 records a new episode from 0.
        _drive(env, num_steps=2)
        # env1 should have 5 recorded steps; env0 should have 2 (not overwrite env1).
        assert env._traj_steps.tolist() == [2, 5]
        # env1 step 4 (the 5th) was recorded and is intact.
        env1_qpos_step4 = env._traj_buffer["states"]["robot"]["qpos"][1, 4]
        assert not torch.all(env1_qpos_step4 == 0)
    finally:
        env.close()
        SimulationManager.flush_cleanup_queue()
        gc.collect()
```

- [ ] **Step 2: Write the ActionManager dynamic-replay test**

Append a delta-ActionManager env variant + test:

```python
from embodichain.lab.gym.envs.managers import ActionManager
from embodichain.lab.gym.envs.managers.cfg import ActionTermCfg


@register_env("ReplayDeltaTask-v1", max_episode_steps=100, override=True)
class ReplayDeltaEnv(EmbodiedEnv):
    """Same as ReplayTestEnv but with a delta-qpos ActionManager."""

    def __init__(self, record_trajectory: bool = True, num_envs: int = 1, device: str = "cpu", **kwargs):
        cfg = EmbodiedEnvCfg()
        cfg.num_envs = num_envs
        cfg.max_episode_steps = 100
        cfg.sim_cfg = SimulationManagerCfg(headless=True, sim_device=device)
        cfg.robot = RobotCfg(
            uid="UR10",
            fpath=get_data_path("UniversalRobots/UR10/UR10.urdf"),
            init_pos=(0.0, 0.0, 1.0),
            drive_pros=JointDrivePropertiesCfg(drive_type="force"),
        )
        cfg.rigid_object = [
            RigidObjectCfg(uid="cube", shape=CubeCfg(size=[0.03, 0.03, 0.03]),
                           init_pos=(0.0, 0.0, 0.5), body_type="dynamic")
        ]
        cfg.actions = ActionTermCfg(func="DeltaQposTerm", mode="pre", params={"scale": 1.0})
        cfg.record_trajectory = record_trajectory
        super().__init__(cfg, **kwargs)


def test_dynamic_replay_with_action_manager(tmp_path):
    """Dynamic replay feeds pre-process (delta) action; ActionManager re-applies it."""
    env = ReplayDeltaEnv(record_trajectory=True, num_envs=1, device="cpu")
    try:
        env.reset()
        init_qpos = env.robot.get_qpos()
        deltas = []
        for i in range(4):
            d = torch.zeros_like(init_qpos)
            d[:, 0] = 0.05 * (i + 1)
            deltas.append(d)
            env.step(d)
        path = tmp_path / "delta.pt"
        env.save_trajectory(str(path))
        # Recorded action must be the raw delta (pre-process), not the resolved qpos.
        rec = torch.load(path, weights_only=False)
        assert torch.allclose(rec["actions"][0, 0], deltas[0][0], atol=1e-6)
    finally:
        env.close()
        SimulationManager.flush_cleanup_queue()
        gc.collect()

    env2 = ReplayDeltaEnv(record_trajectory=False, num_envs=1, device="cpu")
    env2 = ReplayWrapper(env2, str(path), mode="dynamic")
    try:
        env2.reset()
        for i in range(4):
            obs, reward, term, trunc, info = env2.step(None)
        # Robot ended near init + sum(deltas) on joint 0.
        assert abs(float(env2.env.robot.get_qpos()[0, 0]) - (0.05 * (1 + 2 + 3 + 4))) < 0.1
    finally:
        env2.close()
        SimulationManager.flush_cleanup_queue()
        gc.collect()
```

> **Note:** Verify `ActionTermCfg` is the correct config class and `"DeltaQposTerm"` the correct `func` string by reading `embodichain/lab/gym/envs/managers/cfg.py` and `actions.py` before writing the test. If the delta term's signature differs (e.g., requires a different `params` key), adjust accordingly and report it. If a delta ActionManager is too fiddly to construct from config, fall back to constructing the `ActionManager` cfg minimally and assert only that the recorded `actions` equal the raw delta (drop the replay portion), reporting the simplification.

- [ ] **Step 3: Run the new tests**

Run: `pytest tests/gym/envs/test_replay.py::test_async_envs_do_not_corrupt_recording tests/gym/envs/test_replay.py::test_dynamic_replay_with_action_manager -v`
Expected: PASS.

- [ ] **Step 4: Run the full replay suite**

Run: `pytest tests/gym/envs/test_replay.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
black .
git add tests/gym/envs/test_replay.py
git commit -m "test(replay): async parallel envs + ActionManager dynamic replay"
```

---

## Self-Review Notes

- **Spec coverage:** dedicated buffer (Task 2), per-env `_traj_steps` (Task 2), pre-process action (Task 2), auto-save default path (Task 3), multi-env replay per-env lengths (Task 1), async test (Task 4), ActionManager dynamic (Task 4), backward compat (Task 1). All spec sections 5-11 covered.
- **Type consistency:** `build_trajectory_buffer` signature matches across Task 2's gym_utils + embodied_env. `save_trajectory(path, env_ids)` matches Task 2 + Task 3. `_traj_steps`/`_traj_buffer`/`_traj_raw_action` consistent. ReplayWrapper `_replay_steps`/`_lengths` consistent across Task 1.
- **Green at each task:** Task 1 replay is bi-directionally format-compatible (reads old + new), so Task 2's format switch doesn't break replay. Task 2's recording reads `_traj_buffer`; Task 1 replay reads the saved file. Task 3 adds auto-save. Task 4 adds behavioral tests.
- **Deferred (not in plan):** Dict (non-Box) action spaces in `build_trajectory_buffer` (Box-only; current action spaces are Box); `qvel` recording; async multi-env replay "freeze finished envs" mode (finished envs currently hold last state/action).
