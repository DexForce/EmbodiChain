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

"""Cached scene-layout event adapter; EventManager owns scheduling and seeds."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from embodichain.lab.sim.scene.expansion import (
    apply_scene_variants,
    generate_scene_variants,
    load_scene_variants,
    prepare_scene_randomization,
    scene_layout_input_from_sim,
)
from ..cfg import FunctorCfg, SceneEntityCfg
from ..manager_base import Functor

if TYPE_CHECKING:
    from embodichain.lab.gym.envs import EmbodiedEnv

__all__ = ["RandomizeSceneLayout"]


class RandomizeSceneLayout(Functor):
    """Apply bounded online or saved layouts through an ordinary reset event.

    Args:
        cfg: EventCfg with support ``entity_cfg``, complete ``entity_cfgs``,
            ``parent_by_uid``, ``source`` and ``policy`` in params. Source kind
            is ``online`` or ``saved``; saved sources require ``layout_dir``.
            Optional candidate_count/max_attempts control each bounded refill.
        env: Caller-owned manager environment; no Task Program is required.

    Geometry is prepared once. Ordinary reset retains prepared geometry and
    unused candidates. EventManager explicit reseeding rewinds candidate and
    replay cursors. This functor does not step physics or certify task success.
    """

    def __init__(self, cfg: FunctorCfg, env: EmbodiedEnv) -> None:
        super().__init__(cfg, env)
        if getattr(cfg, "mode", None) != "reset":
            raise ValueError("RandomizeSceneLayout currently requires mode='reset'.")
        params = cfg.params
        allowed = {
            "entity_cfg",
            "entity_cfgs",
            "parent_by_uid",
            "source",
            "policy",
            "candidate_count",
            "max_attempts",
            "cache_dir",
        }
        if set(params) - allowed:
            raise ValueError(
                f"Unknown scene layout event params: {sorted(set(params) - allowed)}"
            )
        self._support = params["entity_cfg"]
        self._entities = tuple(params["entity_cfgs"])
        if not isinstance(self._support, SceneEntityCfg) or any(
            not isinstance(x, SceneEntityCfg) for x in self._entities
        ):
            raise TypeError("Scene layout selections require SceneEntityCfg values.")
        self._parents = deepcopy(params["parent_by_uid"])
        self._source = deepcopy(params.get("source", {"kind": "online"}))
        if not isinstance(self._source, dict) or self._source.get("kind") not in (
            "online",
            "saved",
        ):
            raise ValueError("Scene layout source.kind must be online or saved.")
        if set(self._source) - {"kind", "layout_dir"}:
            raise ValueError("Unsupported scene layout source fields.")
        if self._source["kind"] == "saved" and not self._source.get("layout_dir"):
            raise ValueError("Saved scene layout sources require layout_dir.")
        self._policy = deepcopy(params.get("policy", {}))
        self._candidate_count = params.get("candidate_count", 64)
        self._max_attempts = params.get("max_attempts", 10000)
        if (
            type(self._candidate_count) is not int
            or self._candidate_count < 1
            or type(self._max_attempts) is not int
            or self._max_attempts < self._candidate_count
        ):
            raise ValueError(
                "Scene layout candidate count and attempt budget are invalid."
            )
        self._cache_dir = params.get("cache_dir")
        self._prepared = self._loaded = None
        self._pools = deque()
        self._available = self._next_ordinal = self._replay_cursor = 0
        self._seed_revision = None
        self._conflicts_checked = False
        if not hasattr(env, "_scene_randomization_metadata"):
            env._scene_randomization_metadata = [None for _ in range(env.num_envs)]
            env._scene_randomization_invalid_rows = torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            )

    def reset(self, env_ids: list[int] | torch.Tensor | None = None) -> None:
        """Clear selected transient results while preserving geometry and streams."""
        ids = range(self.num_envs) if env_ids is None else env_ids
        for row in ids:
            self._env._scene_randomization_metadata[int(row)] = None

    def _check_conflicts(self, env: EmbodiedEnv) -> None:
        manager = env.event_manager
        if manager is None:
            return
        managed = set(self._parents)
        for name in manager.active_functors.get("reset", ()):
            cfg = manager.get_functor_cfg(name)
            func = cfg.func
            if func is self:
                continue
            label = getattr(func, "__name__", type(func).__name__)
            if not (
                label == "RandomizeSceneLayout"
                or ("randomize" in label and ("pose" in label or "scale" in label))
                or label == "apply_expansion_profile_reset"
            ):
                continue
            selected = set()

            def collect(value: object) -> None:
                if isinstance(value, SceneEntityCfg):
                    selected.add(value.uid)
                elif isinstance(value, dict):
                    for item in value.values():
                        collect(item)
                elif isinstance(value, (tuple, list)):
                    for item in value:
                        collect(item)

            collect(cfg.params)
            if isinstance(func, RandomizeSceneLayout):
                selected.update(func._parents)
            if label == "apply_expansion_profile_reset":
                selected.add("cube")
            if managed & selected:
                raise ValueError(
                    f"Scene layout event conflicts with pose/scale event {name!r}: {sorted(managed & selected)}"
                )
        self._conflicts_checked = True

    def _prepare(self, env: EmbodiedEnv, prototype: int) -> None:
        live = scene_layout_input_from_sim(
            env.sim,
            support_uid=self._support.uid,
            parent_by_uid=self._parents,
            scene_id="authored-scene",
            env_id=prototype,
        )
        expected = {self._support.uid, *(entity.uid for entity in self._entities)}
        if expected != set(live.poses):
            raise ValueError(
                "Scene layout entity selections must cover the explicit support tree exactly."
            )
        if self._source["kind"] == "online":
            self._prepared = prepare_scene_randomization(
                live, self._policy, cache_dir=self._cache_dir
            )
        else:
            self._loaded = load_scene_variants(Path(self._source["layout_dir"]))
            saved = self._loaded.prepared.source
            if (
                set(saved.poses) != set(live.poses)
                or dict(saved.parent_by_uid) != dict(live.parent_by_uid)
                or saved.support_uid != live.support_uid
            ):
                raise ValueError(
                    "Saved layouts do not match the live entity/support topology."
                )
            for uid in saved.poses:
                if (
                    saved.vertices[uid].shape != live.vertices[uid].shape
                    or not np.allclose(
                        saved.vertices[uid], live.vertices[uid], atol=1e-6, rtol=0
                    )
                    or not np.array_equal(saved.faces[uid], live.faces[uid])
                ):
                    raise ValueError(
                        f"Saved layout geometry differs from live entity {uid!r}."
                    )
            if not len(self._loaded.variants):
                raise ValueError(
                    "Saved scene layout package contains no valid candidates."
                )
            self._prepared = self._loaded.prepared

    def __call__(
        self,
        env: EmbodiedEnv,
        env_ids: list[int] | torch.Tensor | None,
        **kwargs: object,
    ) -> None:
        """Apply one layout per selected row and publish owned provenance.

        Args:
            env: Existing manager environment.
            env_ids: Selected reset rows; None selects all rows.
            **kwargs: Construction parameters already validated from EventCfg.
        """
        del kwargs
        if isinstance(env_ids, slice):
            rows = list(range(self.num_envs))[env_ids]
        elif env_ids is None:
            rows = list(range(self.num_envs))
        else:
            rows = (
                env_ids.detach().cpu().tolist()
                if isinstance(env_ids, torch.Tensor)
                else list(env_ids)
            )
        if any(
            type(row) is not int or not 0 <= row < self.num_envs for row in rows
        ) or len(rows) != len(set(rows)):
            raise ValueError("Scene layout reset rows must be unique valid IDs.")
        rows.sort()
        if not rows:
            return
        env._scene_randomization_invalid_rows[rows] = True
        if not self._conflicts_checked:
            self._check_conflicts(env)
        if self._prepared is None:
            self._prepare(env, rows[0])
        revision = env.event_manager.seed_revision
        if revision != self._seed_revision:
            self._pools.clear()
            self._available = self._next_ordinal = self._replay_cursor = 0
            self._seed_revision = revision
        if self._loaded is not None:
            variants = tuple(
                self._loaded.variants[
                    (self._replay_cursor + i) % len(self._loaded.variants)
                ]
                for i in range(len(rows))
            )
        else:
            if self._available < len(rows):
                seed = int(np.random.randint(0, 2**32))
                count = max(self._candidate_count, len(rows) - self._available)
                result = generate_scene_variants(
                    self._prepared,
                    seed=seed,
                    candidate_count=count,
                    max_attempts=self._max_attempts,
                    start_ordinal=self._next_ordinal,
                )
                self._next_ordinal = result.next_ordinal
                if len(result.variants):
                    self._pools.append([result.variants, 0])
                    self._available += len(result.variants)
                if self._available < len(rows):
                    raise RuntimeError(
                        "Scene randomization budget_exhausted before all reset rows could be assigned."
                    )
            variants = []
            for _ in rows:
                pool, cursor = self._pools[0]
                variants.append(pool[cursor])
                self._pools[0][1] += 1
                self._available -= 1
                if self._pools[0][1] == len(pool):
                    self._pools.popleft()
            variants = tuple(variants)
        source = self._prepared.source
        for uid in source.poses:
            if uid in self._prepared.movable_uids:
                continue
            actual = env.sim.get_rigid_object(uid).get_local_pose(to_matrix=True)[rows]
            expected = torch.as_tensor(
                np.asarray(source.poses[uid]), dtype=actual.dtype, device=actual.device
            )
            if not torch.allclose(
                actual, expected.expand_as(actual), atol=1e-6, rtol=0
            ):
                raise ValueError(
                    f"Fixed scene entity {uid!r} moved relative to the prepared geometry."
                )
        result = apply_scene_variants(
            env.sim,
            variants,
            env_ids=rows,
            movable_entity_ids=self._prepared.movable_uids,
        )
        if self._loaded is not None:
            self._replay_cursor += len(rows)
        for row, variant in zip(rows, variants):
            metadata = {
                "schema_version": 1,
                "source_sha256": source.fingerprint,
                "geometry_sha256": self._prepared.geometry_sha256,
                "variant_id": variant.variant_id,
                "seed": variant.seed,
                "proposal_ordinal": dict(variant.metadata).get(
                    "proposal_ordinal", str(variant.ordinal)
                ),
                "geometry_status": "passed",
                "refresh_required": list(result.refresh_required),
            }
            env._scene_randomization_metadata[row] = deepcopy(metadata)
            episodes = getattr(env, "_demo_episode_metadata", None)
            if episodes is not None:
                episodes[row]["scene_randomization"] = deepcopy(metadata)
        callback = getattr(env, "on_scene_variants_applied", None)
        if callable(callback):
            callback(result)
        env._scene_randomization_invalid_rows[rows] = False
