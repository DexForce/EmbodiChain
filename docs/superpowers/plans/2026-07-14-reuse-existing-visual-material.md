# Reuse Existing Visual Material Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `randomize_visual_material` reuse each object's existing dexsim-parsed material (instance-swap + pre-created textures) instead of creating a new material each time, with a three-tier texture choice (original/library/solid) and a `fallback_to_new` flag preserving the old behavior.

**Architecture:** Add a `ReuseSegmentState` (original immutable `MaterialInst` + working `VisualMaterialInst`) per render-body segment, built by new `get_existing_visual_material` / `apply_render_material_inst` methods on `RigidObject`/`Articulation`. The functor's new path swaps the original instance back for the "original" tier and mutates the working instance for "library"/"solid" tiers, side-stepping dexsim's inability to read back textures. Old init/call code is preserved verbatim behind `fallback_to_new`.

**Tech Stack:** Python 3.10, PyTorch, dexsim (compiled C++ pybind), `@configclass`, pytest with custom mocks.

## Global Constraints

- Format with `black==26.3.1` before every commit; run `/pre-commit-check`.
- Every source file starts with the Apache 2.0 copyright header (see `AGENTS.md`).
- `from __future__ import annotations` at the top of every file; prefer `A | B` over `Union[A, B]`.
- Configs use `@configclass`; public modules define `__all__`.
- Logger: `logger.log_warning(msg)`, `logger.log_info(msg)`, `logger.log_error(msg, error_type=RuntimeError)` (log_error **raises**, does not return).
- Package import path is `embodichain` (lowercase).
- Google-style docstrings with Sphinx directives.

---

## File Structure

- **Modify** `embodichain/lab/sim/material.py` — add `texture_obj` param to `VisualMaterialInst.set_base_color_texture`; add `ReuseSegmentState` dataclass.
- **Modify** `embodichain/lab/sim/objects/rigid_object.py` — add `get_existing_visual_material`, `apply_render_material_inst`.
- **Modify** `embodichain/lab/sim/objects/articulation.py` — add `get_existing_visual_material`, `apply_render_material_inst` (link-aware).
- **Modify** `embodichain/lab/sim/__init__.py` — export `ReuseSegmentState`.
- **Modify** `embodichain/lab/gym/envs/managers/randomization/visual.py` — refactor legacy init/call into methods; add reuse init/call paths, three-tier logic, pre-created textures, fallback gating.
- **Create** `tests/gym/envs/managers/test_randomize_visual_material.py` — unit tests with mocks.

---

## Task 1: `VisualMaterialInst.set_base_color_texture` accepts a pre-created `Texture`

**Files:**
- Modify: `embodichain/lab/sim/material.py:243-271` (`set_base_color_texture`)
- Test: `tests/sim/test_material_texture_obj.py`

**Interfaces:**
- Produces: `VisualMaterialInst.set_base_color_texture(texture_path=None, texture_data=None, texture_obj=None)` — when `texture_obj` (a dexsim `Texture`) is given, calls `MaterialInst.set_base_color_map(texture_obj)` directly with no `create_color_texture` upload.

- [ ] **Step 1: Write the failing test**

Create `tests/sim/test_material_texture_obj.py`:

```python
# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
# ... (Apache 2.0 header - copy from an existing test file)
# ----------------------------------------------------------------------------
from __future__ import annotations

from unittest.mock import MagicMock

from embodichain.lab.sim.material import VisualMaterialInst


def _make_inst():
    mat = MagicMock(name="Material")
    inst = VisualMaterialInst("uid_test", mat)
    return inst, mat


def test_texture_obj_sets_map_without_upload():
    inst, mat = MagicMock(), MagicMock()
    obj = VisualMaterialInst.__new__(VisualMaterialInst)
    obj.uid = "u"
    obj._mat = mat
    obj.base_color_texture = None

    texture = MagicMock(name="Texture")
    obj.set_base_color_texture(texture_obj=texture)

    dexsim_inst = mat.get_inst.return_value
    dexsim_inst.set_base_color_map.assert_called_once_with(texture)
    assert obj.base_color_texture is texture


def test_texture_obj_takes_priority_over_data():
    obj = VisualMaterialInst.__new__(VisualMaterialInst)
    obj.uid = "u"
    mat = MagicMock()
    obj._mat = mat
    obj.base_color_texture = None

    obj.set_base_color_texture(texture_data=MagicMock(), texture_obj=MagicMock())

    # texture_obj branch used, create_color_texture never called
    mat.get_inst.assert_called_with("u")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/sim/test_material_texture_obj.py -v`
Expected: FAIL with `TypeError: set_base_color_texture() got an unexpected keyword argument 'texture_obj'`

- [ ] **Step 3: Write minimal implementation**

Replace the body of `set_base_color_texture` in `embodichain/lab/sim/material.py` (currently lines 243-271) with:

```python
    def set_base_color_texture(
        self,
        texture_path: str = None,
        texture_data: torch.Tensor | None = None,
        texture_obj=None,
    ) -> None:
        """Set base color texture from file path, tensor data, or a pre-created Texture.

        Args:
            texture_path: Path to texture file.
            texture_data: Texture data as a torch.Tensor (uploaded each call).
            texture_obj: A pre-created dexsim ``Texture`` object. When provided, it is
                bound directly without re-uploading (priority over ``texture_data``).
        """
        if texture_path is not None and (texture_data is not None or texture_obj is not None):
            logger.log_warning(
                "Both texture_path and another texture source are provided. Using texture_path."
            )

        if texture_path is not None:
            self.base_color_texture = texture_path
            inst = self._mat.get_inst(self.uid)
            inst.set_base_color_map(texture_path)
        elif texture_obj is not None:
            self.base_color_texture = texture_obj
            inst = self._mat.get_inst(self.uid)
            inst.set_base_color_map(texture_obj)
        elif texture_data is not None:
            self.base_color_texture = texture_data
            inst = self._mat.get_inst(self.uid)

            # TODO: Optimize texture creation method.
            world = dexsim.default_world()
            env = world.get_env()
            color_texture = env.create_color_texture(
                texture_data.cpu().numpy(), has_alpha=True
            )
            inst.set_base_color_map(color_texture)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/sim/test_material_texture_obj.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
black embodichain/lab/sim/material.py tests/sim/test_material_texture_obj.py
git add embodichain/lab/sim/material.py tests/sim/test_material_texture_obj.py
git commit -m "feat(material): accept pre-created Texture in set_base_color_texture"
```

---

## Task 2: `ReuseSegmentState` + `RigidObject` reuse-material methods

**Files:**
- Modify: `embodichain/lab/sim/material.py` (add `ReuseSegmentState`, import `dataclass`)
- Modify: `embodichain/lab/sim/__init__.py` (export `ReuseSegmentState`)
- Modify: `embodichain/lab/sim/objects/rigid_object.py` (add two methods)
- Test: `tests/sim/objects/test_rigid_object_reuse_material.py`

**Interfaces:**
- Produces:
  - `ReuseSegmentState` dataclass (`material.py`) with fields `mesh_id: int`, `original_inst: MaterialInst`, `working_inst: VisualMaterialInst`.
  - `RigidObject.get_existing_visual_material(env_ids=None, shared=False) -> List[List[ReuseSegmentState]]` — per-env (length 1 if `shared`) list of per-segment states. Raises `ValueError` if a segment has no material or no template.
  - `RigidObject.apply_render_material_inst(env_idx, mat_inst, mesh_id=0) -> None` — swaps a dexsim `MaterialInst` onto env's render-body segment via `RenderBody.set_material(mesh_id, mat_inst)`.

- [ ] **Step 1: Write the failing test**

Create `tests/sim/objects/test_rigid_object_reuse_material.py` (with Apache header). Use the "subclass and skip `__init__`" mock trick:

```python
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from embodichain.lab.sim.material import ReuseSegmentState, VisualMaterialInst
from embodichain.lab.sim.objects.rigid_object import RigidObject


class _MockRigidObject(RigidObject):
    def __init__(self, entities, uid):
        # Skip the heavy RigidObject.__init__; set only what the new methods use.
        self._entities = entities
        self._all_indices = list(range(len(entities)))
        self.uid = uid
        self.num_instances = len(entities)


def _make_entity(num_segments=1):
    entity = MagicMock(name="MeshObject")
    render_body = MagicMock(name="RenderBody")
    render_body.get_mesh_count.return_value = num_segments
    seg_mats = []
    for i in range(num_segments):
        orig = MagicMock(name=f"orig_inst_{i}")
        tmpl = MagicMock(name=f"template_{i}")
        orig.get_template.return_value = tmpl
        render_body.get_material.return_value = orig
        seg_mats.append((orig, tmpl))
    entity.get_render_body.return_value = render_body
    return entity, render_body, seg_mats


def test_get_existing_visual_material_builds_state_per_segment():
    entity, render_body, seg_mats = _make_entity(num_segments=2)
    obj = _MockRigidObject([entity], "obj")

    states = obj.get_existing_visual_material()

    assert len(states) == 1  # one env
    assert len(states[0]) == 2  # two segments
    for seg, (orig, tmpl) in zip(states[0], seg_mats):
        assert isinstance(seg, ReuseSegmentState)
        assert seg.original_inst is orig
        assert isinstance(seg.working_inst, VisualMaterialInst)
        tmpl.create_inst.assert_called_once()  # working instance created from template


def test_get_existing_visual_material_shared_returns_single_env():
    entity, render_body, seg_mats = _make_entity(num_segments=1)
    obj = _MockRigidObject([entity, entity, entity], "obj")

    states = obj.get_existing_visual_material(shared=True)
    assert len(states) == 1  # shared -> first env only


def test_get_existing_visual_material_raises_when_no_material():
    entity, render_body, _ = _make_entity(num_segments=1)
    render_body.get_material.return_value = None
    obj = _MockRigidObject([entity], "obj")

    with pytest.raises(ValueError, match="no material"):
        obj.get_existing_visual_material()


def test_apply_render_material_inst_swaps_on_render_body():
    entity, render_body, _ = _make_entity(num_segments=1)
    obj = _MockRigidObject([entity], "obj")
    inst = MagicMock(name="MaterialInst")

    obj.apply_render_material_inst(0, inst, mesh_id=3)

    render_body.set_material.assert_called_once_with(3, inst)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/sim/objects/test_rigid_object_reuse_material.py -v`
Expected: FAIL with `ImportError: cannot import name 'ReuseSegmentState'`

- [ ] **Step 3: Add `ReuseSegmentState` to `material.py`**

In `embodichain/lab/sim/material.py`, add `dataclass` to the imports (top of file, after `import copy`):

```python
import copy
import torch
import dexsim
import numpy as np

from dataclasses import dataclass
from typing import Dict, Union
from functools import cached_property

from dexsim.engine import MaterialInst, Material
from embodichain.utils import configclass, logger
```

Append at the end of `material.py` (after the `VisualMaterialInst` class):

```python
@dataclass
class ReuseSegmentState:
    """Reuse state for one render-body segment of a parsed object.

    Used by ``randomize_visual_material`` to randomize on top of the material dexsim
    parsed from the asset, instead of creating a new material.

    Attributes:
        mesh_id: The render-body segment index.
        original_inst: The dexsim ``MaterialInst`` parsed from the asset. Kept immutable
            and swapped back onto the render body for the "original" tier.
        working_inst: A ``VisualMaterialInst`` created from ``original_inst``'s template;
            mutated in place for the "library"/"solid" tiers.
    """

    mesh_id: int
    original_inst: MaterialInst
    working_inst: VisualMaterialInst
```

Update `__all__` in `material.py` (add `"ReuseSegmentState"`).

- [ ] **Step 4: Export `ReuseSegmentState` from `embodichain/lab/sim/__init__.py`**

Find the line importing `VisualMaterial, VisualMaterialInst, VisualMaterialCfg` from `.material` and add `ReuseSegmentState`; add it to the package `__all__` as well. (Read the file first to match the exact existing import style.)

- [ ] **Step 5: Add the two methods to `RigidObject`**

In `embodichain/lab/sim/objects/rigid_object.py`, ensure `ReuseSegmentState` is imported. Update the existing `from embodichain.lab.sim import VisualMaterial, VisualMaterialInst, BatchEntity` line to also import `ReuseSegmentState`. Then add these methods to the `RigidObject` class (place them right after `get_visual_material_inst`, ~line 891):

```python
    def get_existing_visual_material(
        self,
        env_ids: Sequence[int] | None = None,
        shared: bool = False,
    ) -> List[List[ReuseSegmentState]]:
        """Build reuse state from the material dexsim parsed onto each env's render body.

        For each env (first only if ``shared``) and each render-body segment, the existing
        ``MaterialInst`` is captured as an immutable original and a working instance is
        created from its template and wrapped as ``VisualMaterialInst``.

        Args:
            env_ids: Environment indices. If None, all instances are used.
            shared: If True, build state for the first env only (caller applies it to all).

        Returns:
            Per-env list of per-segment :obj:`ReuseSegmentState` (length 1 if ``shared``).

        Raises:
            ValueError: If a segment has no material or no retrievable template.
        """
        if shared:
            local_env_ids = [self._all_indices[0]]
        else:
            local_env_ids = self._all_indices if env_ids is None else list(env_ids)

        per_env: List[List[ReuseSegmentState]] = []
        for env_idx in local_env_ids:
            render_body = self._entities[env_idx].get_render_body()
            mesh_count = render_body.get_mesh_count()
            segments: List[ReuseSegmentState] = []
            for mesh_id in range(mesh_count):
                original_inst = render_body.get_material(mesh_id)
                if original_inst is None:
                    raise ValueError(
                        f"RigidObject '{self.uid}' env {env_idx} segment {mesh_id} has no material."
                    )
                template = original_inst.get_template()
                if template is None:
                    raise ValueError(
                        f"RigidObject '{self.uid}' segment {mesh_id} material has no template."
                    )
                working_name = f"{self.uid}_reuse_{env_idx}_{mesh_id}"
                template.create_inst(working_name)
                working_inst = VisualMaterialInst(working_name, template)
                segments.append(
                    ReuseSegmentState(
                        mesh_id=mesh_id,
                        original_inst=original_inst,
                        working_inst=working_inst,
                    )
                )
            per_env.append(segments)
        return per_env

    def apply_render_material_inst(
        self,
        env_idx: int,
        mat_inst,
        mesh_id: int = 0,
    ) -> None:
        """Swap a dexsim MaterialInst onto a render-body segment for the given env.

        Args:
            env_idx: Environment index.
            mat_inst: dexsim ``MaterialInst`` to attach.
            mesh_id: Render-body segment index.
        """
        self._entities[env_idx].get_render_body().set_material(mesh_id, mat_inst)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/sim/objects/test_rigid_object_reuse_material.py -v`
Expected: PASS (4 tests)

- [ ] **Step 7: Commit**

```bash
black embodichain/lab/sim/material.py embodichain/lab/sim/__init__.py embodichain/lab/sim/objects/rigid_object.py tests/sim/objects/test_rigid_object_reuse_material.py
git add embodichain/lab/sim/material.py embodichain/lab/sim/__init__.py embodichain/lab/sim/objects/rigid_object.py tests/sim/objects/test_rigid_object_reuse_material.py
git commit -m "feat(rigid_object): add get_existing_visual_material and apply_render_material_inst"
```

---

## Task 3: `Articulation` reuse-material methods (link-aware)

**Files:**
- Modify: `embodichain/lab/sim/objects/articulation.py` (add two methods)
- Test: `tests/sim/objects/test_articulation_reuse_material.py`

**Interfaces:**
- Produces:
  - `Articulation.get_existing_visual_material(env_ids=None, link_names=None, shared=False) -> List[Dict[str, List[ReuseSegmentState]]]` — per-env (length 1 if `shared`) dict mapping link_name -> per-segment states. Raises `ValueError` if a link/segment has no material or no template.
  - `Articulation.apply_render_material_inst(env_idx, mat_inst, link_name, mesh_id=0) -> None`.

- [ ] **Step 1: Write the failing test**

Create `tests/sim/objects/test_articulation_reuse_material.py` (Apache header):

```python
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from embodichain.lab.sim.material import ReuseSegmentState, VisualMaterialInst
from embodichain.lab.sim.objects.articulation import Articulation


class _MockArticulation(Articulation):
    def __init__(self, entities, uid, link_names):
        self._entities = entities
        self._all_indices = list(range(len(entities)))
        self.uid = uid
        self.num_instances = len(entities)
        self.link_names = link_names


def _make_entity(links, segs_per_link=1):
    entity = MagicMock(name="ArtEntity")
    rbs = {}
    for link in links:
        rb = MagicMock(name=f"rb_{link}")
        rb.get_mesh_count.return_value = segs_per_link
        orig = MagicMock(name=f"orig_{link}")
        tmpl = MagicMock(name=f"tmpl_{link}")
        orig.get_template.return_value = tmpl
        rb.get_material.return_value = orig
        rbs[link] = rb
    entity.get_render_body.side_effect = lambda name: rbs.get(name)
    return entity, rbs


def test_get_existing_visual_material_per_link():
    links = ["base", "gripper"]
    entity, rbs = _make_entity(links, segs_per_link=1)
    obj = _MockArticulation([entity], "art", links)

    states = obj.get_existing_visual_material(link_names=links)

    assert len(states) == 1
    assert set(states[0].keys()) == set(links)
    for link in links:
        assert len(states[0][link]) == 1
        seg = states[0][link][0]
        assert isinstance(seg, ReuseSegmentState)
        assert isinstance(seg.working_inst, VisualMaterialInst)


def test_get_existing_visual_material_raises_when_no_material():
    links = ["base"]
    entity, rbs = _make_entity(links)
    rbs["base"].get_material.return_value = None
    obj = _MockArticulation([entity], "art", links)

    with pytest.raises(ValueError, match="no material"):
        obj.get_existing_visual_material(link_names=links)


def test_apply_render_material_inst_swaps_on_link_render_body():
    links = ["base"]
    entity, rbs = _make_entity(links)
    obj = _MockArticulation([entity], "art", links)
    inst = MagicMock(name="MaterialInst")

    obj.apply_render_material_inst(0, inst, link_name="base", mesh_id=2)

    rbs["base"].set_material.assert_called_once_with(2, inst)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/sim/objects/test_articulation_reuse_material.py -v`
Expected: FAIL with `AttributeError: ... has no attribute 'get_existing_visual_material'`

- [ ] **Step 3: Add the two methods to `Articulation`**

In `embodichain/lab/sim/objects/articulation.py`, update the import that brings in `VisualMaterial`/`VisualMaterialInst` to also import `ReuseSegmentState` (match the existing import line; read the file head first). Then add these methods right after `get_visual_material_inst` (~line 2229):

```python
    def get_existing_visual_material(
        self,
        env_ids: Sequence[int] | None = None,
        link_names: List[str] | None = None,
        shared: bool = False,
    ) -> List[Dict[str, List[ReuseSegmentState]]]:
        """Build reuse state from materials dexsim parsed onto each link's render body.

        Args:
            env_ids: Environment indices. If None, all instances are used.
            link_names: Links to include. If None, all links are used.
            shared: If True, build state for the first env only.

        Returns:
            Per-env dict mapping link name to per-segment :obj:`ReuseSegmentState`.

        Raises:
            ValueError: If a link/segment has no material or no retrievable template.
        """
        if shared:
            local_env_ids = [self._all_indices[0]]
        else:
            local_env_ids = self._all_indices if env_ids is None else list(env_ids)
        link_names = self.link_names if link_names is None else list(link_names)

        per_env: List[Dict[str, List[ReuseSegmentState]]] = []
        for env_idx in local_env_ids:
            link_map: Dict[str, List[ReuseSegmentState]] = {}
            for link_name in link_names:
                render_body = self._entities[env_idx].get_render_body(link_name)
                if render_body is None:
                    raise ValueError(
                        f"Articulation '{self.uid}' link '{link_name}' has no render body."
                    )
                mesh_count = render_body.get_mesh_count()
                segments: List[ReuseSegmentState] = []
                for mesh_id in range(mesh_count):
                    original_inst = render_body.get_material(mesh_id)
                    if original_inst is None:
                        raise ValueError(
                            f"Articulation '{self.uid}' link '{link_name}' segment {mesh_id} has no material."
                        )
                    template = original_inst.get_template()
                    if template is None:
                        raise ValueError(
                            f"Articulation '{self.uid}' link '{link_name}' material has no template."
                        )
                    working_name = f"{self.uid}_reuse_{env_idx}_{link_name}_{mesh_id}"
                    template.create_inst(working_name)
                    working_inst = VisualMaterialInst(working_name, template)
                    segments.append(
                        ReuseSegmentState(
                            mesh_id=mesh_id,
                            original_inst=original_inst,
                            working_inst=working_inst,
                        )
                    )
                link_map[link_name] = segments
            per_env.append(link_map)
        return per_env

    def apply_render_material_inst(
        self,
        env_idx: int,
        mat_inst,
        link_name: str,
        mesh_id: int = 0,
    ) -> None:
        """Swap a dexsim MaterialInst onto a link's render-body segment for the given env.

        Args:
            env_idx: Environment index.
            mat_inst: dexsim ``MaterialInst`` to attach.
            link_name: Link whose render body receives the material.
            mesh_id: Render-body segment index.
        """
        self._entities[env_idx].get_render_body(link_name).set_material(mesh_id, mat_inst)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/sim/objects/test_articulation_reuse_material.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
black embodichain/lab/sim/objects/articulation.py tests/sim/objects/test_articulation_reuse_material.py
git add embodichain/lab/sim/objects/articulation.py tests/sim/objects/test_articulation_reuse_material.py
git commit -m "feat(articulation): add get_existing_visual_material and apply_render_material_inst"
```

---

## Task 4: Refactor `randomize_visual_material` legacy init/call into methods

This is a pure refactor: extract the current `__init__` material-setup and `__call__` body into `_init_legacy` / `_call_legacy` so the new path can be added alongside without touching behavior. No new feature yet.

**Files:**
- Modify: `embodichain/lab/gym/envs/managers/randomization/visual.py`
- Test: `tests/gym/envs/managers/test_randomize_visual_material.py` (new, characterization tests)

**Interfaces:**
- Produces (internal): `_init_legacy()` and `_call_legacy(env, env_ids, random_texture_prob, base_color_range, metallic_range, roughness_range, ior_range, clean: bool)` on `randomize_visual_material`.

- [ ] **Step 1: Write characterization tests for current behavior**

Create `tests/gym/envs/managers/test_randomize_visual_material.py` (Apache header). These mocks mirror the patterns in `test_event_functors.py` (`MockEnv`/`MockSim`):

```python
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from embodichain.lab.gym.envs.managers import FunctorCfg
from embodichain.lab.gym.envs.managers.randomization.visual import (
    randomize_visual_material,
)
from embodichain.lab.gym.envs.managers.cfg import SceneEntityCfg
from embodichain.lab.sim.objects.rigid_object import RigidObject


class _MockRigidObject(RigidObject):
    """RigidObject that skips the heavy __init__; methods mocked for tests."""

    def __init__(self, uid="obj", num_envs=2):
        self.uid = uid
        self.num_instances = num_envs
        self._all_indices = list(range(num_envs))
        self._entities = [MagicMock(name=f"mesh{i}") for i in range(num_envs)]
        self._visual_material = [None] * num_envs
        self.is_shared_visual_material = False
        self.set_visual_material = MagicMock()
        self.get_visual_material_inst = MagicMock(
            return_value=[MagicMock(name=f"inst{i}") for i in range(num_envs)]
        )
        self.get_existing_visual_material = MagicMock(return_value=[])
        self.apply_render_material_inst = MagicMock()


class _MockSim:
    def __init__(self, num_envs=2):
        self.textures = {}
        self._visual_materials = {}
        self.created_visual_materials = []
        self.env = MagicMock(name="dexsim_env")
        self.env.create_color_texture = MagicMock(return_value=MagicMock(name="Texture"))
        self.env.clean_materials = MagicMock()
        self.asset_uids = ["obj"]
        self._asset = _MockRigidObject(num_envs=num_envs)

    def get_texture_cache(self, key=None):
        if key is None:
            return self.textures
        return self.textures.get(key)

    def set_texture_cache(self, key, value):
        self.textures[key] = value

    def create_visual_material(self, cfg):
        self.created_visual_materials.append(cfg.uid)
        mat = MagicMock(name="VisualMaterial")
        inst = MagicMock(name="VisualMaterialInst")
        mat.create_instance.return_value = inst
        inst.mat = MagicMock(name="MaterialInst")
        self._visual_materials[cfg.uid] = mat
        return mat

    def get_visual_material(self, uid):
        m = MagicMock(name="plane_VisualMaterial")
        m.get_default_instance.return_value = MagicMock(name="plane_inst")
        return m

    def get_asset(self, uid):
        return self._asset

    def get_env(self):
        return self.env


class _MockEnv:
    def __init__(self, num_envs=2):
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.sim = _MockSim(num_envs=num_envs)


def _make_cfg(params):
    cfg = FunctorCfg(func=randomize_visual_material)
    cfg.params = params
    return cfg


def test_legacy_init_creates_visual_material():
    env = _MockEnv()
    cfg = _make_cfg({"entity_cfg": SceneEntityCfg(uid="obj"), "fallback_to_new": True})
    functor = randomize_visual_material(cfg, env)
    assert env.sim.created_visual_materials  # legacy path creates a material


def test_legacy_call_runs_clean_materials():
    env = _MockEnv()
    cfg = _make_cfg({"entity_cfg": SceneEntityCfg(uid="obj"), "fallback_to_new": True})
    functor = randomize_visual_material(cfg, env)
    env.sim.env.clean_materials = MagicMock()
    functor(env, torch.arange(env.num_envs))
    env.sim.env.clean_materials.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/gym/envs/managers/test_randomize_visual_material.py -v`
Expected: FAIL (the `fallback_to_new` branch and `_call_legacy`/`_init_legacy` do not exist yet; legacy init may not record `created_visual_materials` because the gating isn't there).

- [ ] **Step 3: Refactor `__init__` and `__call__` into legacy methods**

In `embodichain/lab/gym/envs/managers/randomization/visual.py`, modify `randomize_visual_material`:

1. At the end of the existing `__init__` common section (after texture preload), replace the material-creation block (the `if self.entity_cfg.uid == "default_plane": ... else: mat = env.sim.create_visual_material(...)` block and the `_mat_insts` assignment, lines ~618-654) with a call:

```python
        self._fallback_to_new = bool(cfg.params.get("fallback_to_new", False))
        self._new_mode = False  # set True in Task 5 for the reuse path
        self._init_legacy(env)
```

2. Add the `_init_legacy` method holding the exact current material-creation + `_mat_insts` logic (move the existing code verbatim into it). It should reference `env` via `self._env`:

```python
    def _init_legacy(self, env: EmbodiedEnv) -> None:
        """Legacy init: create a new material and replace the object's material."""
        if self.entity_cfg.uid == "default_plane":
            pass
        else:
            mat: VisualMaterial = env.sim.create_visual_material(
                cfg=VisualMaterialCfg(
                    base_color=[1.0, 1.0, 1.0, 1.0],
                    uid=f"{self.entity_cfg.uid}_random_mat",
                )
            )
            if isinstance(self.entity, RigidObject):
                self.entity.set_visual_material(mat)
            elif isinstance(self.entity, Articulation):
                _, link_names = resolve_matching_names(
                    self.entity_cfg.link_names, self.entity.link_names
                )
                self.entity_cfg.link_names = link_names
                self.entity.set_visual_material(mat, link_names=link_names)

        self._mat_insts = None
        if self.entity_cfg.uid == "default_plane":
            self._mat_insts = env.sim.get_visual_material("plane_mat").get_default_instance()
            return
        elif isinstance(self.entity, RigidObject):
            self._mat_insts = self.entity.get_visual_material_inst()
            if self.entity.is_shared_visual_material:
                self._mat_insts = self._mat_insts[:1]
        elif isinstance(self.entity, Articulation):
            self._mat_insts = self.entity.get_visual_material_inst(
                link_names=self.entity_cfg.link_names,
            )
            if self.entity.is_shared_visual_material:
                self._mat_insts = self._mat_insts[:1]
```

3. Replace the body of `__call__` with a dispatch to `_call_legacy` (Task 5 will add the reuse branch):

```python
    def __call__(
        self,
        env: EmbodiedEnv,
        env_ids: Union[torch.Tensor, None],
        entity_cfg: SceneEntityCfg,
        random_texture_prob: float = 0.5,
        texture_path: str | None = None,
        base_color_range: tuple[list[float], list[float]] | None = None,
        metallic_range: tuple[float, float] | None = None,
        roughness_range: tuple[float, float] | None = None,
        ior_range: tuple[float, float] | None = None,
        fallback_to_new: bool = False,
        p_original: float | None = None,
        p_library: float | None = None,
        p_solid: float | None = None,
        shared: bool | None = None,
    ):
        return self._call_legacy(
            env,
            env_ids,
            random_texture_prob,
            base_color_range,
            metallic_range,
            roughness_range,
            ior_range,
            clean=True,
        )
```

4. Add `_call_legacy` holding the exact current `__call__` body (plan sampling + `_randomize_mat_inst` loop + plane handling), with the final `clean_materials()` gated by `clean`:

```python
    def _call_legacy(
        self,
        env: EmbodiedEnv,
        env_ids: Union[torch.Tensor, None],
        random_texture_prob: float,
        base_color_range,
        metallic_range,
        roughness_range,
        ior_range,
        clean: bool,
    ) -> None:
        if self.entity_cfg.uid != "default_plane" and self.entity is None:
            return

        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device="cpu")
        else:
            env_ids = env_ids.cpu()

        if self.entity_cfg.uid == "default_plane":
            env_ids = [0]

        randomize_plan = {}
        if base_color_range:
            base_color = sample_uniform(
                lower=torch.tensor(base_color_range[0], dtype=torch.float32),
                upper=torch.tensor(base_color_range[1], dtype=torch.float32),
                size=(len(env_ids), 3),
            )
            alpha_channel = torch.ones((len(env_ids), 1), dtype=torch.float32)
            base_color = torch.cat((base_color, alpha_channel), dim=1)
            randomize_plan["base_color"] = base_color
        if metallic_range:
            metallic = sample_uniform(
                lower=torch.tensor(metallic_range[0], dtype=torch.float32),
                upper=torch.tensor(metallic_range[1], dtype=torch.float32),
                size=(len(env_ids), 1),
            )
            randomize_plan["metallic"] = metallic
        if roughness_range:
            roughness = sample_uniform(
                lower=torch.tensor(roughness_range[0], dtype=torch.float32),
                upper=torch.tensor(roughness_range[1], dtype=torch.float32),
                size=(len(env_ids), 1),
            )
            randomize_plan["roughness"] = roughness
        if ior_range:
            ior = sample_uniform(
                lower=torch.tensor(ior_range[0], dtype=torch.float32),
                upper=torch.tensor(ior_range[1], dtype=torch.float32),
                size=(len(env_ids), 1),
            )
            randomize_plan["ior"] = ior

        if self.entity_cfg.uid == "default_plane":
            mat_inst = env.sim.get_visual_material("plane_mat").get_default_instance()
            self._randomize_mat_inst(
                mat_inst=mat_inst, plan=randomize_plan,
                random_texture_prob=random_texture_prob, idx=0,
            )
            if clean:
                env.sim.get_env().clean_materials()
            return

        for i, data in enumerate(self._mat_insts):
            if isinstance(self.entity, RigidObject):
                mat: VisualMaterialInst = data
            elif isinstance(self.entity, Articulation):
                mat: Dict[str, VisualMaterialInst] = data

            if isinstance(self.entity, RigidObject):
                self._randomize_mat_inst(
                    mat_inst=mat, plan=randomize_plan,
                    random_texture_prob=random_texture_prob, idx=i,
                )
            else:
                for name, mat_inst in mat.items():
                    self._randomize_mat_inst(
                        mat_inst=mat_inst, plan=randomize_plan,
                        random_texture_prob=random_texture_prob, idx=i,
                    )

        if clean:
            env.sim.get_env().clean_materials()
```

Leave the existing `_randomize_mat_inst`, `_randomize_texture`, and `gen_random_base_color_texture` unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/gym/envs/managers/test_randomize_visual_material.py -v`
Expected: PASS (2 tests). The legacy path still creates a material and calls `clean_materials`.

- [ ] **Step 5: Commit**

```bash
black embodichain/lab/gym/envs/managers/randomization/visual.py tests/gym/envs/managers/test_randomize_visual_material.py
git add embodichain/lab/gym/envs/managers/randomization/visual.py tests/gym/envs/managers/test_randomize_visual_material.py
git commit -m "refactor(visual): extract randomize_visual_material legacy init/call into methods"
```

---

## Task 5: Reuse path for `RigidObject` (new init + three-tier swap call)

**Files:**
- Modify: `embodichain/lab/gym/envs/managers/randomization/visual.py`
- Test: `tests/gym/envs/managers/test_randomize_visual_material.py` (extend)

**Interfaces:**
- Produces (internal): `_init_reuse()`, `_call_reuse(env, env_ids, base_color_range, metallic_range, roughness_range, ior_range)`, `_build_library_textures()`, `_resolve_tier_probs()`, `_sample_tiers(num_reuse)`, `_apply_library_tier(...)`, `_apply_solid_tier(...)`, `_apply_plan_props(...)`.
- Behavior: when `fallback_to_new` is False and the entity is a `RigidObject` (not plane), `__init__` tries the reuse path; on success `self._new_mode = True`, on failure it logs a warning and falls back to `_init_legacy`. `__call__` routes to `_call_reuse` when `_new_mode`, else `_call_legacy(clean=...)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/gym/envs/managers/test_randomize_visual_material.py`:

```python
def _seg(mesh_id, orig, tmpl):
    """Build a ReuseSegmentState whose working_inst is a fully-mocked VisualMaterialInst."""
    from embodichain.lab.sim.material import ReuseSegmentState, VisualMaterialInst

    inst = VisualMaterialInst.__new__(VisualMaterialInst)
    inst.uid = f"w_{mesh_id}"
    inst._mat = tmpl
    inst.base_color_texture = None
    inst.base_color = [1, 1, 1, 1]
    inst.metallic = 0.0
    inst.roughness = 0.7
    inst.ior = 1.5
    inst.emissive = [0, 0, 0]
    inst.set_base_color = MagicMock()
    inst.set_metallic = MagicMock()
    inst.set_roughness = MagicMock()
    inst.set_ior = MagicMock()
    inst.set_base_color_texture = MagicMock()
    inst.mat = MagicMock(name="working_mat_inst")
    return ReuseSegmentState(mesh_id=mesh_id, original_inst=orig, working_inst=inst)


def test_new_init_does_not_create_visual_material():
    env = _MockEnv()
    obj = env.sim.get_asset("obj")  # cached _MockRigidObject
    seg = _seg(0, MagicMock(name="orig"), MagicMock(name="tmpl"))
    obj.get_existing_visual_material = MagicMock(return_value=[[seg]])
    cfg = _make_cfg({"entity_cfg": SceneEntityCfg(uid="obj")})  # fallback_to_new defaults False

    functor = randomize_visual_material(cfg, env)

    assert env.sim.created_visual_materials == []  # no new material
    assert functor._new_mode is True


def test_new_init_degrades_to_legacy_on_failure():
    env = _MockEnv()
    obj = env.sim.get_asset("obj")
    obj.get_existing_visual_material = MagicMock(side_effect=ValueError("no material"))
    cfg = _make_cfg({"entity_cfg": SceneEntityCfg(uid="obj")})

    functor = randomize_visual_material(cfg, env)

    assert functor._new_mode is False
    assert env.sim.created_visual_materials  # degraded to legacy


def test_new_call_no_clean_and_swaps():
    env = _MockEnv()
    obj = env.sim.get_asset("obj")
    seg = _seg(0, MagicMock(name="orig"), MagicMock(name="tmpl"))
    obj.get_existing_visual_material = MagicMock(return_value=[[seg]])
    cfg = _make_cfg({"entity_cfg": SceneEntityCfg(uid="obj")})
    functor = randomize_visual_material(cfg, env)

    env.sim.env.clean_materials.reset_mock()
    # force original tier (p_original=1)
    functor._p_original, functor._p_library, functor._p_solid = 1.0, 0.0, 0.0

    functor(env, torch.arange(env.num_envs))

    env.sim.env.clean_materials.assert_not_called()
    obj.apply_render_material_inst.assert_called()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/gym/envs/managers/test_randomize_visual_material.py -v`
Expected: FAIL (reuse path does not exist; `functor._new_mode` absent).

- [ ] **Step 3: Implement the reuse init**

In `randomize_visual_material.__init__` (in `visual.py`), replace the `self._init_legacy(env)` line added in Task 4 with the branching:

```python
        self._fallback_to_new = bool(cfg.params.get("fallback_to_new", False))
        self._shared = bool(cfg.params.get("shared", False))
        self._new_mode = False
        self._reuse_state = None
        self._library_textures: list = []
        self._texture_key = os.path.basename(get_data_path(cfg.params.get("texture_path", None))) if cfg.params.get("texture_path", None) else ""

        can_reuse = (
            not self._fallback_to_new
            and self.entity_cfg.uid != "default_plane"
            and isinstance(self.entity, (RigidObject, Articulation))
        )
        if can_reuse:
            try:
                self._init_reuse(env)
                self._new_mode = True
            except Exception as e:  # noqa: BLE001 - degrade gracefully
                logger.log_warning(
                    f"randomize_visual_material: reuse-existing-material unavailable for "
                    f"'{self.entity_cfg.uid}' ({e}); falling back to new-material path."
                )
                self._new_mode = False
        if not self._new_mode:
            self._init_legacy(env)
```

Add the `_init_reuse` method (and helpers) to the class:

```python
    def _init_reuse(self, env: EmbodiedEnv) -> None:
        """Init the reuse path: capture existing materials, pre-create textures, resolve tiers."""
        if isinstance(self.entity, RigidObject):
            self._reuse_state = self.entity.get_existing_visual_material(
                shared=self._shared
            )
        elif isinstance(self.entity, Articulation):
            _, link_names = resolve_matching_names(
                self.entity_cfg.link_names, self.entity.link_names
            )
            self.entity_cfg.link_names = link_names
            self._reuse_state = self.entity.get_existing_visual_material(
                link_names=link_names, shared=self._shared
            )
        self._build_library_textures(env)
        self._resolve_tier_probs()

    def _build_library_textures(self, env: EmbodiedEnv) -> None:
        """Pre-create dexsim Texture objects once, cached at sim level across functors."""
        self._library_textures = []
        if not self.textures:
            return
        sim = env.sim
        cache = sim.get_texture_cache()  # whole dict when key is None
        tex_key = f"{self._texture_key}__tex_objs"
        if tex_key in cache:
            self._library_textures = cache[tex_key]
            return
        dexsim_env = sim.get_env()
        self._library_textures = [
            dexsim_env.create_color_texture(t.cpu().numpy(), has_alpha=True)
            for t in self.textures
        ]
        sim.set_texture_cache(tex_key, self._library_textures)

    def _resolve_tier_probs(self) -> None:
        """Resolve p_original/p_library/p_solid with backward-compat derivation."""
        cfg = self.cfg
        p_original = cfg.params.get("p_original", None)
        p_library = cfg.params.get("p_library", None)
        p_solid = cfg.params.get("p_solid", None)
        random_texture_prob = float(cfg.params.get("random_texture_prob", 0.5))

        if p_original is None and p_library is None and p_solid is None:
            p_original, p_library, p_solid = 0.0, random_texture_prob, 1.0 - random_texture_prob
        else:
            p_original = 0.0 if p_original is None else float(p_original)
            p_library = 0.0 if p_library is None else float(p_library)
            p_solid = 0.0 if p_solid is None else float(p_solid)

        if not self._library_textures:
            p_solid += p_library
            p_library = 0.0

        total = p_original + p_library + p_solid
        if total <= 0:
            p_solid = 1.0
            total = 1.0
        if abs(total - 1.0) > 1e-6:
            logger.log_warning(
                f"randomize_visual_material: tier probabilities sum to {total}; normalizing."
            )
        self._p_original = p_original / total
        self._p_library = p_library / total
        self._p_solid = p_solid / total
```

- [ ] **Step 4: Implement the reuse call + dispatch**

Update `__call__` to dispatch to `_call_reuse` when `_new_mode`:

```python
    def __call__(
        self,
        env: EmbodiedEnv,
        env_ids: Union[torch.Tensor, None],
        entity_cfg: SceneEntityCfg,
        random_texture_prob: float = 0.5,
        texture_path: str | None = None,
        base_color_range: tuple[list[float], list[float]] | None = None,
        metallic_range: tuple[float, float] | None = None,
        roughness_range: tuple[float, float] | None = None,
        ior_range: tuple[float, float] | None = None,
        fallback_to_new: bool = False,
        p_original: float | None = None,
        p_library: float | None = None,
        p_solid: float | None = None,
        shared: bool | None = None,
    ):
        if self._new_mode:
            return self._call_reuse(
                env, env_ids, base_color_range, metallic_range, roughness_range, ior_range
            )
        clean = bool(self._fallback_to_new)
        return self._call_legacy(
            env, env_ids, random_texture_prob, base_color_range,
            metallic_range, roughness_range, ior_range, clean=clean,
        )
```

Add `_call_reuse` and tier helpers (RigidObject branch; Articulation added in Task 6):

```python
    def _call_reuse(self, env, env_ids, base_color_range, metallic_range,
                    roughness_range, ior_range) -> None:
        if self.entity is None:
            return
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device="cpu")
        else:
            env_ids = env_ids.cpu()

        num_reuse = len(self._reuse_state)  # 1 if shared, else num_envs
        plan = self._sample_plan(num_reuse, base_color_range, metallic_range, roughness_range, ior_range)
        tiers = self._sample_tiers(num_reuse)

        is_articulation = isinstance(self.entity, Articulation)

        def _apply(reuse_i: int, env_idx: int) -> None:
            tier = int(tiers[reuse_i].item())
            if is_articulation:
                self._apply_tier_articulation(reuse_i, env_idx, tier, plan)
            else:
                for seg in self._reuse_state[reuse_i]:
                    self._apply_tier_rigid(seg, env_idx, tier, plan, reuse_i)

        if self._shared:
            # single reuse state applied to every env
            for env_idx in env_ids.tolist():
                _apply(0, int(env_idx))
        else:
            for reuse_i in range(num_reuse):
                _apply(reuse_i, int(env_ids[reuse_i]))

    def _sample_plan(self, num, base_color_range, metallic_range, roughness_range, ior_range):
        plan = {}
        if base_color_range:
            base_color = sample_uniform(
                lower=torch.tensor(base_color_range[0], dtype=torch.float32),
                upper=torch.tensor(base_color_range[1], dtype=torch.float32),
                size=(num, 3),
            )
            alpha = torch.ones((num, 1), dtype=torch.float32)
            plan["base_color"] = torch.cat((base_color, alpha), dim=1)
        if metallic_range:
            plan["metallic"] = sample_uniform(
                lower=torch.tensor(metallic_range[0], dtype=torch.float32),
                upper=torch.tensor(metallic_range[1], dtype=torch.float32),
                size=(num, 1),
            )
        if roughness_range:
            plan["roughness"] = sample_uniform(
                lower=torch.tensor(roughness_range[0], dtype=torch.float32),
                upper=torch.tensor(roughness_range[1], dtype=torch.float32),
                size=(num, 1),
            )
        if ior_range:
            plan["ior"] = sample_uniform(
                lower=torch.tensor(ior_range[0], dtype=torch.float32),
                upper=torch.tensor(ior_range[1], dtype=torch.float32),
                size=(num, 1),
            )
        return plan

    def _sample_tiers(self, num_reuse: int) -> torch.Tensor:
        probs = torch.tensor(
            [self._p_original, self._p_library, self._p_solid], dtype=torch.float32
        )
        return torch.multinomial(probs, num_samples=num_reuse, replacement=True)

    def _apply_inst(self, env_idx, mat_inst, mesh_id, link_name=None) -> None:
        """Swap a MaterialInst onto the render body (link-aware for Articulation)."""
        if link_name is None:
            self.entity.apply_render_material_inst(env_idx, mat_inst, mesh_id)
        else:
            self.entity.apply_render_material_inst(env_idx, mat_inst, link_name, mesh_id)

    def _apply_tier_rigid(self, seg, env_idx, tier, plan, idx) -> None:
        if tier == 0:  # original
            self._apply_inst(env_idx, seg.original_inst, seg.mesh_id)
            return
        if tier == 1 and self._library_textures:  # library
            self._apply_library_tier(seg, env_idx, plan, idx)
        else:  # solid (or library with empty library)
            self._apply_solid_tier(seg, env_idx)

    def _apply_library_tier(self, seg, env_idx, plan, idx, link_name=None) -> None:
        tex_idx = torch.randint(0, len(self._library_textures), (1,)).item()
        seg.working_inst.set_base_color_texture(texture_obj=self._library_textures[tex_idx])
        self._apply_plan_props(seg.working_inst, plan, idx)
        self._apply_inst(env_idx, seg.working_inst.mat, seg.mesh_id, link_name)

    def _apply_solid_tier(self, seg, env_idx, link_name=None) -> None:
        seg.working_inst.set_base_color([1.0, 1.0, 1.0, 1.0])
        seg.working_inst.set_metallic(0.0)
        seg.working_inst.set_roughness(0.7)
        solid = randomize_visual_material.gen_random_base_color_texture(2, 2)
        seg.working_inst.set_base_color_texture(texture_data=solid)
        self._apply_inst(env_idx, seg.working_inst.mat, seg.mesh_id, link_name)

    def _apply_plan_props(self, working_inst, plan, idx) -> None:
        if "base_color" in plan:
            working_inst.set_base_color(plan["base_color"][idx].tolist())
        if "metallic" in plan:
            working_inst.set_metallic(plan["metallic"][idx].item())
        if "roughness" in plan:
            working_inst.set_roughness(plan["roughness"][idx].item())
        if "ior" in plan:
            working_inst.set_ior(plan["ior"][idx].item())

    def _apply_tier_articulation(self, reuse_i, env_idx, tier, plan) -> None:
        # Implemented in Task 6.
        raise NotImplementedError
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/gym/envs/managers/test_randomize_visual_material.py -v`
Expected: PASS (all 5 tests). New init does not create a material; degrades on failure; new call does not clean and swaps.

- [ ] **Step 6: Commit**

```bash
black embodichain/lab/gym/envs/managers/randomization/visual.py tests/gym/envs/managers/test_randomize_visual_material.py
git add embodichain/lab/gym/envs/managers/randomization/visual.py tests/gym/envs/managers/test_randomize_visual_material.py
git commit -m "feat(visual): reuse existing material for RigidObject with three-tier swap"
```

---

## Task 6: Articulation + default_plane reuse handling

**Files:**
- Modify: `embodichain/lab/gym/envs/managers/randomization/visual.py` (implement `_apply_tier_articulation`; plane stays on legacy in-place path without clean)
- Test: `tests/gym/envs/managers/test_randomize_visual_material.py` (extend)

**Interfaces:**
- Produces: `_apply_tier_articulation(reuse_i, env_idx, tier, plan)` operating on `self._reuse_state[reuse_i]` (a `Dict[str, List[ReuseSegmentState]]`).

- [ ] **Step 1: Write the failing tests**

Append:

```python
def _art_asset(uid="art"):
    from embodichain.lab.sim.objects.articulation import Articulation

    class _A(Articulation):
        def __init__(self):
            self._entities = []
            self._all_indices = [0, 1]
            self.uid = uid
            self.num_instances = 2
            self.is_shared_visual_material = False
            self.link_names = ["link0"]

    obj = _A()
    seg = _seg(0, MagicMock(name="orig"), MagicMock(name="tmpl"))
    obj.get_existing_visual_material = MagicMock(return_value=[{"link0": [seg]}])
    obj.apply_render_material_inst = MagicMock()
    return obj


def test_new_init_articulation_reuse():
    env = _MockEnv()
    env.sim.asset_uids = ["art"]
    env.sim.get_asset = lambda uid: _art_asset()
    cfg = _make_cfg({"entity_cfg": SceneEntityCfg(uid="art", link_names=["link0"])})

    functor = randomize_visual_material(cfg, env)

    assert functor._new_mode is True
    assert env.sim.created_visual_materials == []


def test_new_call_articulation_swaps_per_link():
    env = _MockEnv()
    art = _art_asset()
    env.sim.get_asset = lambda uid: art
    env.sim.asset_uids = ["art"]
    cfg = _make_cfg({"entity_cfg": SceneEntityCfg(uid="art", link_names=["link0"])})
    functor = randomize_visual_material(cfg, env)
    functor._p_original, functor._p_library, functor._p_solid = 1.0, 0.0, 0.0

    functor(env, torch.arange(env.num_envs))

    art.apply_render_material_inst.assert_called()


def test_plane_new_mode_uses_legacy_inplace_no_clean():
    env = _MockEnv()
    env.sim.asset_uids = ["default_plane"]
    cfg = _make_cfg({"entity_cfg": SceneEntityCfg(uid="default_plane")})

    functor = randomize_visual_material(cfg, env)
    assert functor._new_mode is False  # plane never uses swap path

    env.sim.env.clean_materials.reset_mock()
    functor(env, torch.arange(env.num_envs))
    env.sim.env.clean_materials.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/gym/envs/managers/test_randomize_visual_material.py -v`
Expected: FAIL (articulation test hits `NotImplementedError`; plane test calls `clean_materials`).

- [ ] **Step 3: Implement `_apply_tier_articulation`**

Replace the `NotImplementedError` stub in `visual.py` with:

```python
    def _apply_tier_articulation(self, reuse_i, env_idx, tier, plan) -> None:
        link_map = self._reuse_state[reuse_i]  # Dict[str, List[ReuseSegmentState]]
        for link_name, segments in link_map.items():
            for seg in segments:
                if tier == 0:  # original
                    self._apply_inst(env_idx, seg.original_inst, seg.mesh_id, link_name)
                    continue
                if tier == 1 and self._library_textures:  # library
                    self._apply_library_tier(seg, env_idx, plan, reuse_i, link_name)
                else:  # solid
                    self._apply_solid_tier(seg, env_idx, link_name)
```

Also ensure the plane path in `_call_legacy` does not call `clean_materials` when invoked from new mode. The `clean` flag already controls this; `__call__` passes `clean=self._fallback_to_new` (False for plane in new mode). Verify the plane branch in `_call_legacy` respects `clean` (it does, per Task 4).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/gym/envs/managers/test_randomize_visual_material.py -v`
Expected: PASS (all 8 tests).

- [ ] **Step 5: Commit**

```bash
black embodichain/lab/gym/envs/managers/randomization/visual.py tests/gym/envs/managers/test_randomize_visual_material.py
git add embodichain/lab/gym/envs/managers/randomization/visual.py tests/gym/envs/managers/test_randomize_visual_material.py
git commit -m "feat(visual): reuse path for Articulation; plane stays in-place without clean"
```

---

## Task 7: Edge cases & backward-compat tests

**Files:**
- Test: `tests/gym/envs/managers/test_randomize_visual_material.py` (extend only)

- [ ] **Step 1: Write the edge-case tests**

Append:

```python
def test_tier_probs_backward_compat_derivation():
    env = _MockEnv()
    obj = env.sim.get_asset("obj")
    obj.get_existing_visual_material = MagicMock(return_value=[[_seg(0, MagicMock(), MagicMock())]])
    cfg = _make_cfg({"entity_cfg": SceneEntityCfg(uid="obj"), "random_texture_prob": 0.3})
    functor = randomize_visual_material(cfg, env)

    # With a non-empty library, backward-compat derivation gives p_library=0.3, p_solid=0.7.
    functor._library_textures = [MagicMock(name="Texture")]
    functor._resolve_tier_probs()

    assert functor._p_original == 0.0
    assert pytest.approx(functor._p_library) == 0.3
    assert pytest.approx(functor._p_solid) == 0.7


def test_tier_probs_explicit_normalize():
    env = _MockEnv()
    obj = env.sim.get_asset("obj")
    obj.get_existing_visual_material = MagicMock(return_value=[[_seg(0, MagicMock(), MagicMock())]])
    cfg = _make_cfg({
        "entity_cfg": SceneEntityCfg(uid="obj"),
        "p_original": 1.0, "p_library": 1.0, "p_solid": 2.0,
    })
    functor = randomize_visual_material(cfg, env)
    functor._library_textures = [MagicMock(name="Texture")]
    functor._resolve_tier_probs()

    assert pytest.approx(functor._p_original) == 0.25
    assert pytest.approx(functor._p_library) == 0.25
    assert pytest.approx(functor._p_solid) == 0.5


def test_empty_library_folds_into_solid():
    env = _MockEnv()
    obj = env.sim.get_asset("obj")
    obj.get_existing_visual_material = MagicMock(return_value=[[_seg(0, MagicMock(), MagicMock())]])
    cfg = _make_cfg({
        "entity_cfg": SceneEntityCfg(uid="obj"),
        "p_original": 0.0, "p_library": 0.5, "p_solid": 0.5,
    })  # no texture_path -> empty library

    functor = randomize_visual_material(cfg, env)

    assert functor._p_library == 0.0
    assert pytest.approx(functor._p_solid) == 1.0


def test_library_textures_cached_across_functors():
    env = _MockEnv()
    obj = env.sim.get_asset("obj")
    obj.get_existing_visual_material = MagicMock(return_value=[[_seg(0, MagicMock(), MagicMock())]])
    cfg = _make_cfg({"entity_cfg": SceneEntityCfg(uid="obj")})
    functor = randomize_visual_material(cfg, env)

    # Simulate a non-empty texture library and run _build_library_textures directly.
    fake_tex = torch.zeros((2, 2, 4), dtype=torch.uint8)
    functor.textures = [fake_tex]
    functor._texture_key = "texA"
    functor._build_library_textures(env)
    assert env.sim.get_env().create_color_texture.call_count == 1

    # A second functor with the same key reuses the cached Textures (no new upload).
    functor2 = randomize_visual_material(cfg, env)
    functor2.textures = [fake_tex]
    functor2._texture_key = "texA"
    functor2._build_library_textures(env)
    assert env.sim.get_env().create_color_texture.call_count == 1


def test_fallback_to_new_preserves_legacy():
    env = _MockEnv()
    obj = env.sim.get_asset("obj")
    obj.get_existing_visual_material = MagicMock(return_value=[[_seg(0, MagicMock(), MagicMock())]])
    cfg = _make_cfg({"entity_cfg": SceneEntityCfg(uid="obj"), "fallback_to_new": True})

    functor = randomize_visual_material(cfg, env)
    assert functor._new_mode is False
    assert env.sim.created_visual_materials  # legacy created a material

    env.sim.env.clean_materials.reset_mock()
    functor(env, torch.arange(env.num_envs))
    env.sim.env.clean_materials.assert_called_once()  # legacy cleans


def test_multi_segment_all_swapped():
    env = _MockEnv()
    obj = env.sim.get_asset("obj")
    segs = [_seg(0, MagicMock(), MagicMock()), _seg(1, MagicMock(), MagicMock())]
    obj.get_existing_visual_material = MagicMock(return_value=[segs])
    cfg = _make_cfg({"entity_cfg": SceneEntityCfg(uid="obj")})
    functor = randomize_visual_material(cfg, env)
    functor._p_original, functor._p_library, functor._p_solid = 1.0, 0.0, 0.0

    functor(env, torch.arange(env.num_envs))

    # two segments -> two apply calls; mesh_id is the 3rd positional arg
    mesh_ids = {call.args[2] for call in obj.apply_render_material_inst.call_args_list}
    assert mesh_ids == {0, 1}
```

- [ ] **Step 2: Run the full test suite for the feature**

Run: `pytest tests/gym/envs/managers/test_randomize_visual_material.py tests/sim/objects/test_rigid_object_reuse_material.py tests/sim/objects/test_articulation_reuse_material.py tests/sim/test_material_texture_obj.py -v`
Expected: PASS (all tests across all tasks).

- [ ] **Step 3: Run pre-commit check**

Run: `/pre-commit-check` (or `black . && pytest` if the skill is unavailable)
Expected: no formatting/CI violations.

- [ ] **Step 4: Commit**

```bash
black tests/gym/envs/managers/test_randomize_visual_material.py
git add tests/gym/envs/managers/test_randomize_visual_material.py
git commit -m "test(visual): edge cases and backward-compat for reuse-existing-material"
```

---

## Runtime Verification (manual, after all tasks pass)

These cannot be verified from stubs; run with a real dexsim/GPU env if available:

- [ ] `Material.create_inst(name)` produces a usable working instance (blank or copy both work).
- [ ] `RenderBody.set_material(mesh_id, inst)` per-reset is cheap and changes rendered appearance immediately.
- [ ] `set_base_color_map(pre-created Texture)` renders correctly.
- [ ] Per-env material instances: confirm whether shared or independent (swap model is robust to either).
- [ ] Visual smoke test: a textured object randomized with `p_original>0` shows its original texture on the original tier and random textures on other tiers across repeated resets.

## Out of scope (v1)

- `default_plane` original tier and pre-created-texture optimization (plane uses in-place two-tier without `clean_materials`).
- Per-segment independent tiers (v1 tier is per-object).
- Pre-generated solid-color palette (v1 uses per-reset 2×2 upload).
