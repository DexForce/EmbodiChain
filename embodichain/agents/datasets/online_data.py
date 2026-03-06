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

from typing import Callable, Optional

from tensordict import TensorDict
from torch.utils.data import Dataset

from embodichain.lab.engine.data import OnlineDataEngine


class OnlineDataset(Dataset):
    """PyTorch Dataset backed by a live :class:`OnlineDataEngine` shared buffer.

    Wraps an :class:`OnlineDataEngine` to expose a standard
    :class:`torch.utils.data.Dataset` interface that draws trajectory chunks
    on-the-fly from the shared rollout buffer populated by the simulation
    subprocess.

    Because the underlying data is generated continuously, the dataset has a
    *virtual* length (``dataset_size``) that controls how many samples are
    considered per epoch by a :class:`~torch.utils.data.DataLoader`.  Each
    call to :meth:`__getitem__` independently samples a fresh chunk from the
    engine regardless of the ``idx`` argument, so every iteration sees
    freshly sampled (potentially new) data.

    Layout of a single sample returned by :meth:`__getitem__`:
        TensorDict with batch size ``[chunk_size]`` containing all keys
        present in the shared buffer (``obs``, ``actions``, ``rewards``, …)
        after the optional ``transform`` has been applied.

    Args:
        engine: An :class:`OnlineDataEngine` instance whose shared buffer is
            used for data sampling.  The engine must have been set up with a
            ``shared_buffer`` of shape ``(buffer_size, max_episode_steps, …)``.
        chunk_size: Number of consecutive timesteps in each sample.  Must not
            exceed ``max_episode_steps`` configured in the engine's environment.
        dataset_size: Virtual dataset length — the value returned by
            :meth:`__len__` and used by DataLoader to determine epoch size.
            Defaults to ``10_000``.
        transform: Optional callable ``(TensorDict) -> TensorDict`` applied to
            each sampled chunk before it is returned.  Use this for per-sample
            post-processing such as type casting, normalisation, or key
            selection.  The callable receives a TensorDict with batch size
            ``[chunk_size]`` and must return one of the same batch size.

    Example::

        engine = OnlineDataEngine(shared_buffer, index_list, env_config)

        def normalize(sample: TensorDict) -> TensorDict:
            sample["actions"] = sample["actions"].float() / action_scale
            return sample

        dataset = OnlineDataset(engine, chunk_size=64, transform=normalize)
        loader  = DataLoader(dataset, batch_size=32, num_workers=4)

        for batch in loader:
            # batch["obs"], batch["actions"], batch["rewards"]
            # each has shape (32, 64, ...)
            train_step(batch)
    """

    def __init__(
        self,
        engine: OnlineDataEngine,
        chunk_size: int,
        transform: Optional[Callable[[TensorDict], TensorDict]] = None,
    ) -> None:
        self._engine = engine
        self._chunk_size = chunk_size
        self._transform = transform

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the buffer size as the virtual length of the dataset."""
        return self._engine.buffer_size

    def __getitem__(self, idx: int) -> TensorDict:
        """Sample a single trajectory chunk from the shared buffer.

        The ``idx`` argument is intentionally ignored — each call draws an
        independent random chunk from the engine so that the DataLoader
        receives diverse, freshly sampled data on every access.

        Args:
            idx: Ignored sample index (required by the Dataset protocol).

        Returns:
            TensorDict with batch size ``[chunk_size]`` containing the sampled
            trajectory data, post-processed by ``transform`` if provided.
        """
        # Draw one chunk (batch_size=1) and remove the outer batch dimension
        # so the returned TensorDict has shape [chunk_size, ...].
        batch = self._engine.sample_batch(batch_size=1, chunk_size=self._chunk_size)
        sample: TensorDict = batch[0]

        if self._transform is not None:
            sample = self._transform(sample)

        return sample
