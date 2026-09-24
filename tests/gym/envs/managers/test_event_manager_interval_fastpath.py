"""interval_step==1 快路径：functor 收到显式全量 ID 张量（非 None）。"""
import torch

from embodichain.lab.gym.envs.managers.event_manager import EventManager


class _RecordingFunctor:
    """记录 env_ids 的哑 functor。"""

    def __init__(self):
        self.calls = []

    def __call__(self, env, env_ids, **kwargs):
        self.calls.append(
            None if env_ids is None else env_ids.clone()
        )


def test_interval_one_fastpath_passes_explicit_all_row_ids(monkeypatch):
    """interval_step==1 且非 global：functor 收到全量 ID 张量（非 None）。"""
    manager = EventManager.__new__(EventManager)
    manager._env = object()
    manager._seed = None

    num_envs = 4096
    counter = torch.zeros(num_envs, dtype=torch.long, device="cpu")
    manager._interval_functor_step_count = [counter]

    class _Cfg:
        is_global = False
        interval_step = 1

    manager._mode_functor_names = {"interval": ["push_robot"]}
    manager._mode_functor_cfgs = {"interval": [_Cfg()]}

    # 拦截 _call_event_functor，记录传入的 env_ids
    received = []
    monkeypatch.setattr(
        manager, "_call_event_functor",
        lambda mode, name, cfg, env, ids: received.append(ids),
    )

    manager.apply(mode="interval")

    assert len(received) == 1
    ids = received[0]
    assert ids is not None, "快路径传了 None——functor 的 len(env_ids) 会崩"
    assert ids.shape[0] == num_envs
    assert ids.dtype == torch.long
    assert torch.equal(ids, torch.arange(num_envs, dtype=torch.long))
