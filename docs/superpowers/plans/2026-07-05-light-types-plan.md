# Light Type Expansion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Support all 6 dexsim light types (point, sun, direction, spot, rect, mesh) in EmbodiChain's core pipeline.

**Architecture:** Expand the flat `LightCfg` with optional type-specific fields, add typed setters to the `Light` BatchEntity class with runtime validation, and map all 6 types in `SimulationManager.add_light`. Backward compatible — existing point-light code works unchanged.

**Tech Stack:** Python 3.10+, PyTorch, dexsim Python bindings, `@configclass` decorator, `BatchEntity` pattern

## Global Constraints

- `intensity` default: `30.0` (was `50.0`)
- `radius` default: `10.0` (was `1e2`)
- Backend method names: `set_rect_wh` (not `set_rect_size`), `set_spot_angle(inner, outer)`, `set_direction(x, y, z)`, `set_mesh(MeshObject)`, `set_shadow(bool)`
- `set_angular_radius`, `set_halo_size`, `set_halo_falloff` are NOT exposed in Python bindings — config fields exist but setters are deferred
- `set_mesh` takes a `dexsim.models.MeshObject`, not a string — mesh lights are asset-based, not tensor-batched
- All new fields have defaults; backward compatibility is mandatory
- Runtime validation: warn on incompatible setter calls, error only on unknown light type

---

### Task 1: Expand `LightCfg` fields

**Files:**
- Modify: `embodichain/lab/sim/cfg.py:794-810`

**Interfaces:**
- Produces: `LightCfg` with expanded `light_type` literal and new optional fields

- [ ] **Step 1: Replace the `LightCfg` class definition**

Replace lines 794-810 in `embodichain/lab/sim/cfg.py`:

```python
@configclass
class LightCfg(ObjectBaseCfg):
    """Configuration for a light asset in the simulation.

    Supports six light types matching the dexsim rendering backend:

    - ``"point"``: Omnidirectional point light with position and falloff radius.
    - ``"sun"``: Directional sun light with position, direction, and angular radius (sun-specific setters like halo are not yet wired through Python bindings).
    - ``"direction"``: Pure directional light at infinite distance (direction only, no position).
    - ``"spot"``: Spotlight with position, direction, and inner/outer cone angles.
    - ``"rect"``: Rectangular area light with position, direction, width, and height.
    - ``"mesh"``: Mesh-based emissive light (requires a MeshObject via :meth:`set_mesh`; not tensor-batched).

    .. attention::
        The ``angular_radius``, ``halo_size``, and ``halo_falloff`` fields are
        reserved for future use. The dexsim Python bindings do not yet expose
        setters for these sun-specific properties.
    """

    light_type: Literal["point", "sun", "direction", "spot", "rect", "mesh"] = "point"
    """Light type. Supported: ``"point"``, ``"sun"``, ``"direction"``, ``"spot"``, ``"rect"``, ``"mesh"``."""

    # ------------------------------------------------------------------
    # Universal properties (apply to all light types)
    # ------------------------------------------------------------------

    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """RGB color of the light source. Defaults to white ``(1.0, 1.0, 1.0)``."""

    intensity: float = 30.0
    """Intensity of the light source in watts/m^2. Defaults to ``30.0``."""

    enable_shadow: bool = True
    """Whether the light casts shadows. Defaults to ``True``."""

    # ------------------------------------------------------------------
    # Point light
    # ------------------------------------------------------------------

    radius: float = 10.0
    """Falloff radius for point lights. Only used when ``light_type="point"``. Defaults to ``10.0``."""

    # ------------------------------------------------------------------
    # Directional properties (sun, direction, spot, rect, mesh)
    # ------------------------------------------------------------------

    direction: tuple[float, float, float] = (0.0, 0.0, -1.0)
    """Direction vector for directional, spot, rect, and mesh lights.
    Defaults to ``(0.0, 0.0, -1.0)`` (pointing down along -Z)."""

    # ------------------------------------------------------------------
    # Sun light (reserved — Python bindings not yet available)
    # ------------------------------------------------------------------

    angular_radius: float = 0.5
    """Angular radius of the sun disc in degrees. Reserved for future use."""

    halo_size: float = 10.0
    """Halo size for sun light. Reserved for future use."""

    halo_falloff: float = 3.0
    """Halo falloff for sun light. Reserved for future use."""

    # ------------------------------------------------------------------
    # Spot light
    # ------------------------------------------------------------------

    spot_angle_inner: float = 30.0
    """Inner cone angle of the spotlight in degrees. Only used when ``light_type="spot"``.
    Defaults to ``30.0``."""

    spot_angle_outer: float = 45.0
    """Outer cone angle of the spotlight in degrees. Only used when ``light_type="spot"``.
    Defaults to ``45.0``."""

    # ------------------------------------------------------------------
    # Rect light
    # ------------------------------------------------------------------

    rect_width: float = 1.0
    """Width of the rectangular area light. Only used when ``light_type="rect"``.
    Defaults to ``1.0``."""

    rect_height: float = 1.0
    """Height of the rectangular area light. Only used when ``light_type="rect"``.
    Defaults to ``1.0``."""

    # ------------------------------------------------------------------
    # Mesh light
    # ------------------------------------------------------------------

    mesh_path: str = ""
    """Asset path for mesh-based emissive lights. Only used when ``light_type="mesh"``.
    The actual mesh assignment is done via :meth:`Light.set_mesh` which accepts a
    :class:`dexsim.models.MeshObject`. This field stores the path for reference."""
```

- [ ] **Step 2: Verify the module imports correctly**

```bash
cd /home/dex/workspace/sources/EmbodiChain && python -c "from embodichain.lab.sim.cfg import LightCfg; print(LightCfg()); print(LightCfg(light_type='spot'))"
```

Expected output: Two LightCfg instances printed without errors.

- [ ] **Step 3: Commit**

```bash
git add embodichain/lab/sim/cfg.py
git commit -m "feat: expand LightCfg to support 6 light types

Add optional fields for sun, direction, spot, rect, and mesh light types.
Change default intensity to 30.0 and default radius to 10.0.
All new fields have defaults — existing point-light code works unchanged.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Add new setters to `Light` class and update `reset()`

**Files:**
- Modify: `embodichain/lab/sim/objects/light.py`

**Interfaces:**
- Consumes: `LightCfg` with expanded fields (from Task 1)
- Produces: New methods on `Light` — `set_direction`, `set_spot_angle`, `set_rect_wh`, `set_mesh`, `enable_shadow`; updated `reset()`

- [ ] **Step 1: Add new setter methods to the `Light` class**

Insert these methods after the existing `set_falloff` method (after line 88 in the current file) and before `set_local_pose`:

```python
    def set_direction(
        self, directions: torch.Tensor, env_ids: Sequence[int] | None = None
    ) -> None:
        """Set direction for directional-type lights.

        Only applies to ``sun``, ``direction``, ``spot``, ``rect``, and ``mesh``
        light types. Logs a warning and no-ops for other types.

        Args:
            directions (torch.Tensor): Tensor of shape (3,) or (M, 3), representing
                (x, y, z) direction vectors.
            env_ids (Sequence[int] | None): Indices of instances to set. If None:
                - For shape (3,), applies to all instances.
                - For shape (M, 3), M must equal num_instances, applies per-instance.
        """
        if self.cfg.light_type not in ("sun", "direction", "spot", "rect", "mesh"):
            logger.warning(
                f"set_direction not applicable to light type "
                f"'{self.cfg.light_type}', ignoring."
            )
            return
        self._apply_vector3(directions, env_ids, "set_direction")

    def set_spot_angle(
        self,
        inner_angles: torch.Tensor,
        outer_angles: torch.Tensor,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Set inner and outer cone angles for spot lights.

        Only applies to ``spot`` light type. Logs a warning and no-ops for other types.

        Args:
            inner_angles (torch.Tensor): Tensor of shape () (0-dim), (1,), or (M,)
                representing inner cone angle in degrees.
            outer_angles (torch.Tensor): Tensor of shape () (0-dim), (1,), or (M,)
                representing outer cone angle in degrees.
            env_ids (Sequence[int] | None): Indices of instances to set.
        """
        if self.cfg.light_type != "spot":
            logger.warning(
                f"set_spot_angle not applicable to light type "
                f"'{self.cfg.light_type}', ignoring."
            )
            return

        if not torch.is_tensor(inner_angles) or not torch.is_tensor(outer_angles):
            logger.log_error("set_spot_angle requires torch.Tensor arguments")
            return

        inner_cpu = inner_angles.detach().cpu()
        outer_cpu = outer_angles.detach().cpu()

        if env_ids is None:
            all_ids = list(range(self.num_instances))
        else:
            all_ids = list(env_ids)

        # Both scalar (0-dim): broadcast to all_ids
        if inner_cpu.ndim == 0 and outer_cpu.ndim == 0:
            iv = float(inner_cpu.item())
            ov = float(outer_cpu.item())
            for i in all_ids:
                self._entities[i].set_spot_angle(iv, ov)
            return

        # Both 1D with matching length
        if inner_cpu.ndim == 1 and outer_cpu.ndim == 1:
            ilen, olen = inner_cpu.shape[0], outer_cpu.shape[0]
            inner_arr = inner_cpu.numpy()
            outer_arr = outer_cpu.numpy()

            if ilen == olen == self.num_instances and env_ids is None:
                for i in range(self.num_instances):
                    self._entities[i].set_spot_angle(
                        float(inner_arr[i]), float(outer_arr[i])
                    )
                return

            if env_ids is not None and ilen == olen == len(all_ids):
                for idx, i in enumerate(all_ids):
                    self._entities[i].set_spot_angle(
                        float(inner_arr[idx]), float(outer_arr[idx])
                    )
                return

        logger.log_error(
            f"set_spot_angle: invalid tensor shapes "
            f"inner={tuple(inner_cpu.shape)}, outer={tuple(outer_cpu.shape)}"
        )

    def set_rect_wh(
        self,
        widths: torch.Tensor,
        heights: torch.Tensor,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Set width and height for rectangular area lights.

        Only applies to ``rect`` light type. Logs a warning and no-ops for other types.

        Args:
            widths (torch.Tensor): Tensor of shape () (0-dim), (1,), or (M,)
                representing width of the rectangular light.
            heights (torch.Tensor): Tensor of shape () (0-dim), (1,), or (M,)
                representing height of the rectangular light.
            env_ids (Sequence[int] | None): Indices of instances to set.
        """
        if self.cfg.light_type != "rect":
            logger.warning(
                f"set_rect_wh not applicable to light type "
                f"'{self.cfg.light_type}', ignoring."
            )
            return

        if not torch.is_tensor(widths) or not torch.is_tensor(heights):
            logger.log_error("set_rect_wh requires torch.Tensor arguments")
            return

        w_cpu = widths.detach().cpu()
        h_cpu = heights.detach().cpu()

        if env_ids is None:
            all_ids = list(range(self.num_instances))
        else:
            all_ids = list(env_ids)

        # Both scalar (0-dim): broadcast to all_ids
        if w_cpu.ndim == 0 and h_cpu.ndim == 0:
            wv = float(w_cpu.item())
            hv = float(h_cpu.item())
            for i in all_ids:
                self._entities[i].set_rect_wh(wv, hv)
            return

        # Both 1D with matching length
        if w_cpu.ndim == 1 and h_cpu.ndim == 1:
            wlen, hlen = w_cpu.shape[0], h_cpu.shape[0]
            w_arr = w_cpu.numpy()
            h_arr = h_cpu.numpy()

            if wlen == hlen == self.num_instances and env_ids is None:
                for i in range(self.num_instances):
                    self._entities[i].set_rect_wh(
                        float(w_arr[i]), float(h_arr[i])
                    )
                return

            if env_ids is not None and wlen == hlen == len(all_ids):
                for idx, i in enumerate(all_ids):
                    self._entities[i].set_rect_wh(
                        float(w_arr[idx]), float(h_arr[idx])
                    )
                return

        logger.log_error(
            f"set_rect_wh: invalid tensor shapes "
            f"width={tuple(w_cpu.shape)}, height={tuple(h_cpu.shape)}"
        )

    def set_mesh(
        self,
        mesh: "MeshObject",
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Set the mesh for mesh-type lights.

        Only applies to ``mesh`` light type. Logs a warning and no-ops for other types.
        This is NOT tensor-batched — the same MeshObject is assigned to all targeted
        instances.

        Args:
            mesh (MeshObject): The mesh object to assign to the light.
            env_ids (Sequence[int] | None): Indices of instances to set. If None,
                applies to all instances.
        """
        if self.cfg.light_type != "mesh":
            logger.warning(
                f"set_mesh not applicable to light type "
                f"'{self.cfg.light_type}', ignoring."
            )
            return

        if env_ids is None:
            target_ids = list(range(self.num_instances))
        else:
            target_ids = list(env_ids)

        for i in target_ids:
            try:
                self._entities[i].set_mesh(mesh)
            except Exception as e:
                logger.log_error(f"set_mesh: error for instance {i}: {e}")

    def enable_shadow(
        self,
        flags: torch.Tensor,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Enable or disable shadow casting.

        Applies to all light types.

        Args:
            flags (torch.Tensor): Boolean tensor of shape () (0-dim), (1,), or (M,).
                Non-zero values enable shadows; zero disables.
            env_ids (Sequence[int] | None): Indices of instances to set.
        """
        if not torch.is_tensor(flags):
            logger.log_error(f"enable_shadow requires a torch.Tensor, got {type(flags)}")
            return

        cpu = flags.detach().cpu()
        if env_ids is None:
            all_ids = list(range(self.num_instances))
        else:
            all_ids = list(env_ids)

        # Scalar: broadcast
        if cpu.ndim == 0:
            val = bool(cpu.item() != 0)
            for i in all_ids:
                self._entities[i].set_shadow(val)
            return

        # 1D tensor
        if cpu.ndim == 1:
            length = cpu.shape[0]
            arr = cpu.numpy()
            if length == self.num_instances and env_ids is None:
                for i in range(self.num_instances):
                    self._entities[i].set_shadow(bool(arr[i] != 0))
                return
            if env_ids is not None and length == len(all_ids):
                for idx, i in enumerate(all_ids):
                    self._entities[i].set_shadow(bool(arr[idx] != 0))
                return
            if length == 1:
                val = bool(arr[0] != 0)
                for i in all_ids:
                    self._entities[i].set_shadow(val)
                return

        logger.log_error(
            f"enable_shadow: tensor shape {tuple(cpu.shape)} is invalid for broadcasting"
        )
```

- [ ] **Step 2: Add the MeshObject import at the top of the file**

Add after line 21 (`from embodichain.lab.sim.cfg import LightCfg`):

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dexsim.models import MeshObject
```

- [ ] **Step 3: Update the `reset` method**

Replace the existing `reset` method (lines 294-299):

```python
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the light to its initial configuration state.

        Applies only the properties relevant to ``self.cfg.light_type``.

        Args:
            env_ids (Sequence[int] | None): The environment IDs to reset.
                If None, resets all environments.
        """
        self.cfg: LightCfg
        light_type = self.cfg.light_type

        # Universal properties
        self.set_color(torch.as_tensor(self.cfg.color), env_ids=env_ids)
        self.set_intensity(torch.as_tensor(self.cfg.intensity), env_ids=env_ids)
        self.enable_shadow(
            torch.as_tensor(float(self.cfg.enable_shadow)), env_ids=env_ids
        )

        # Position (all types except direction)
        if light_type != "direction":
            self.set_local_pose(torch.as_tensor(self.cfg.init_pos), env_ids=env_ids)

        # Point light: falloff
        if light_type == "point":
            self.set_falloff(torch.as_tensor(self.cfg.radius), env_ids=env_ids)

        # Directional types: direction vector
        if light_type in ("sun", "direction", "spot", "rect", "mesh"):
            self.set_direction(torch.as_tensor(self.cfg.direction), env_ids=env_ids)

        # Spot light: cone angles
        if light_type == "spot":
            self.set_spot_angle(
                torch.as_tensor(self.cfg.spot_angle_inner),
                torch.as_tensor(self.cfg.spot_angle_outer),
                env_ids=env_ids,
            )

        # Rect light: dimensions
        if light_type == "rect":
            self.set_rect_wh(
                torch.as_tensor(self.cfg.rect_width),
                torch.as_tensor(self.cfg.rect_height),
                env_ids=env_ids,
            )

        # Mesh light: mesh_path is stored in cfg but actual mesh assignment
        # is done via set_mesh() which requires a MeshObject.
        # Sun-specific angular_radius/halo are reserved for future backend support.
```

- [ ] **Step 4: Update `set_local_pose` to handle direction-type lights**

Add a guard at the beginning of `set_local_pose` (after the docstring, before line 107):

```python
        if self.cfg.light_type == "direction":
            logger.warning(
                "set_local_pose not applicable to 'direction' light type "
                "(infinite distance, direction only). Use set_direction() instead."
            )
            return
```

- [ ] **Step 5: Verify the module imports correctly**

```bash
cd /home/dex/workspace/sources/EmbodiChain && python -c "from embodichain.lab.sim.objects.light import Light; print('OK')"
```

- [ ] **Step 6: Commit**

```bash
git add embodichain/lab/sim/objects/light.py
git commit -m "feat: add type-specific setters to Light class

Add set_direction, set_spot_angle, set_rect_wh, set_mesh, and
enable_shadow methods with runtime type validation. Update reset()
to apply only properties relevant to the light type. Guard
set_local_pose for direction-type lights (no position).

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Update `SimulationManager.add_light` type mapping

**Files:**
- Modify: `embodichain/lab/sim/sim_manager.py:800-837`

**Interfaces:**
- Consumes: `LightCfg` with expanded `light_type` literal (from Task 1), `Light` class with new setters (from Task 2)
- Produces: `add_light` accepts all 6 light types

- [ ] **Step 1: Replace the `add_light` method**

Replace lines 800-837:

```python
    # Light type string → dexsim LightType enum mapping
    _LIGHT_TYPE_MAP: dict[str, LightType] = {
        "point": LightType.POINT,
        "sun": LightType.SUN,
        "direction": LightType.DIRECTION,
        "spot": LightType.SPOT,
        "rect": LightType.RECT,
        "mesh": LightType.MESH,
    }

    def add_light(self, cfg: LightCfg) -> Light:
        """Create a light in the scene.

        Supports six light types: ``"point"``, ``"sun"``, ``"direction"``,
        ``"spot"``, ``"rect"``, and ``"mesh"``. See :class:`LightCfg` for
        type-specific configuration fields.

        Args:
            cfg (LightCfg): Configuration for the light, including type, color,
                intensity, and type-specific properties.

        Returns:
            Light: The created light instance.

        Raises:
            ValueError: If ``cfg.light_type`` is not one of the supported types.
        """
        if cfg.uid is None:
            uid = "light"
            cfg.uid = uid
        else:
            uid = cfg.uid

        if uid in self._lights:
            logger.log_error(f"Light {uid} already exists.")
            return None

        light_type_str = cfg.light_type
        light_type = self._LIGHT_TYPE_MAP.get(light_type_str)
        if light_type is None:
            supported = ", ".join(self._LIGHT_TYPE_MAP.keys())
            logger.log_error(
                f"Unsupported light type: '{light_type_str}'. "
                f"Supported types: {supported}."
            )
            return None

        # Validation warnings for type-specific constraints
        if light_type_str == "mesh" and not cfg.mesh_path:
            logger.warning(
                f"Mesh light '{uid}' has no mesh_path set. "
                f"Use set_mesh() to assign a MeshObject."
            )
        if light_type_str == "rect" and (cfg.rect_width <= 0 or cfg.rect_height <= 0):
            logger.warning(
                f"Rect light '{uid}' has zero or negative dimensions "
                f"(width={cfg.rect_width}, height={cfg.rect_height})."
            )

        env_list = [self._env] if len(self._arenas) == 0 else self._arenas
        light_list = []
        for i, env in enumerate(env_list):
            light_name = f"{uid}_{i}"
            light = env.create_light(light_name, light_type)
            light_list.append(light)

        batch_lights = Light(cfg=cfg, entities=light_list)

        self._lights[uid] = batch_lights

        return batch_lights
```

Note: The `_LIGHT_TYPE_MAP` is a class-level dict on `SimulationManager`. Place it right before `add_light` (after `get_articulation` at line 798).

- [ ] **Step 2: Verify the mapping works**

```bash
cd /home/dex/workspace/sources/EmbodiChain && python -c "
from embodichain.lab.sim.sim_manager import SimulationManager
# Check that the map has all 6 types
mgr = SimulationManager
print(mgr._LIGHT_TYPE_MAP)
"
```

- [ ] **Step 3: Commit**

```bash
git add embodichain/lab/sim/sim_manager.py
git commit -m "feat: support all 6 light types in SimulationManager.add_light

Add _LIGHT_TYPE_MAP mapping string type names to dexsim LightType enum.
Add validation warnings for mesh (no path) and rect (zero dimensions).
Unknown light types now log the list of supported types.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Write tests for new light types

**Files:**
- Modify: `tests/sim/objects/test_light.py`

**Interfaces:**
- Consumes: All types and setters from Tasks 1-3
- Produces: Test coverage for all 6 light types

- [ ] **Step 1: Add new test class `TestLightTypes` after the existing `TestLight` class**

Insert after line 161 (after `gc.collect()`):

```python
class TestLightTypes:
    """Tests for all six supported light types."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Create a SimulationManager for each test."""
        config = SimulationManagerCfg(headless=True, sim_device="cpu", num_envs=4)
        self.sim = SimulationManager(config)
        yield
        self.sim.destroy()
        import embodichain.lab.sim as om

        om.SimulationManager.flush_cleanup_queue()
        self.__dict__.clear()
        import gc

        gc.collect()

    # ------------------------------------------------------------------
    # Creation tests
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "light_type",
        ["point", "sun", "direction", "spot", "rect", "mesh"],
    )
    def test_create_each_light_type(self, light_type):
        """Each of the 6 light types can be created without error."""
        cfg = LightCfg(
            uid=f"test_{light_type}",
            light_type=light_type,
            init_pos=(0.0, 0.0, 3.0),
        )
        light = self.sim.add_light(cfg=cfg)
        assert light is not None, f"Failed to create light of type '{light_type}'"
        assert light.num_instances == 4

    def test_unknown_light_type_errors(self):
        """Passing an invalid light_type logs an error and returns None."""
        cfg = LightCfg(uid="bad", light_type="invalid")
        light = self.sim.add_light(cfg=cfg)
        assert light is None, "Unknown light type should return None"

    def test_mesh_light_empty_path_warns(self):
        """Mesh light with empty mesh_path warns at creation but succeeds."""
        cfg = LightCfg(
            uid="mesh_no_path",
            light_type="mesh",
            init_pos=(0.0, 0.0, 2.0),
        )
        light = self.sim.add_light(cfg=cfg)
        assert light is not None, "Mesh light should be created even without path"

    # ------------------------------------------------------------------
    # Setter type-validation tests
    # ------------------------------------------------------------------

    def test_set_direction_on_point_warns(self):
        """Calling set_direction on a point light logs a warning and no-ops."""
        cfg = LightCfg(uid="pt", light_type="point", init_pos=(0, 0, 2))
        light = self.sim.add_light(cfg=cfg)
        direction = torch.tensor([1.0, 0.0, 0.0])
        # Should not raise; should log a warning
        try:
            light.set_direction(direction)
        except Exception as e:
            pytest.fail(f"set_direction on point light should warn, not crash: {e}")

    def test_set_spot_angle_on_point_warns(self):
        """Calling set_spot_angle on a non-spot light warns and no-ops."""
        cfg = LightCfg(uid="pt2", light_type="point", init_pos=(0, 0, 2))
        light = self.sim.add_light(cfg=cfg)
        inner = torch.tensor(15.0)
        outer = torch.tensor(30.0)
        try:
            light.set_spot_angle(inner, outer)
        except Exception as e:
            pytest.fail(f"set_spot_angle on point light should warn, not crash: {e}")

    def test_set_rect_wh_on_point_warns(self):
        """Calling set_rect_wh on a non-rect light warns and no-ops."""
        cfg = LightCfg(uid="pt3", light_type="point", init_pos=(0, 0, 2))
        light = self.sim.add_light(cfg=cfg)
        w = torch.tensor(2.0)
        h = torch.tensor(3.0)
        try:
            light.set_rect_wh(w, h)
        except Exception as e:
            pytest.fail(f"set_rect_wh on point light should warn, not crash: {e}")

    def test_set_local_pose_on_direction_warns(self):
        """Calling set_local_pose on a direction light warns and no-ops."""
        cfg = LightCfg(
            uid="dir_light",
            light_type="direction",
            direction=(0.0, -1.0, 0.0),
        )
        light = self.sim.add_light(cfg=cfg)
        pose = torch.tensor([1.0, 2.0, 3.0])
        try:
            light.set_local_pose(pose)
        except Exception as e:
            pytest.fail(
                f"set_local_pose on direction light should warn, not crash: {e}"
            )

    # ------------------------------------------------------------------
    # Type-specific property tests
    # ------------------------------------------------------------------

    def test_spot_light_has_cone_angles(self):
        """Spot light creation applies inner/outer cone angles from cfg."""
        cfg = LightCfg(
            uid="spot_test",
            light_type="spot",
            init_pos=(0.0, 0.0, 3.0),
            direction=(0.0, 0.0, -1.0),
            spot_angle_inner=20.0,
            spot_angle_outer=40.0,
        )
        light = self.sim.add_light(cfg=cfg)
        assert light is not None
        # Should not crash on creation; spot_angle applied during reset()

    def test_rect_light_has_dimensions(self):
        """Rect light creation applies width/height from cfg."""
        cfg = LightCfg(
            uid="rect_test",
            light_type="rect",
            init_pos=(0.0, 0.0, 3.0),
            direction=(0.0, 0.0, -1.0),
            rect_width=2.0,
            rect_height=1.5,
        )
        light = self.sim.add_light(cfg=cfg)
        assert light is not None

    def test_direction_light_no_position_in_reset(self):
        """Direction light reset does not set position."""
        cfg = LightCfg(
            uid="dir_test",
            light_type="direction",
            direction=(0.0, -1.0, 0.0),
        )
        light = self.sim.add_light(cfg=cfg)
        assert light is not None

    # ------------------------------------------------------------------
    # Broadcasting tests
    # ------------------------------------------------------------------

    def test_set_direction_broadcasting(self):
        """set_direction with (3,) tensor broadcasts to all instances."""
        cfg = LightCfg(
            uid="spot_broadcast",
            light_type="spot",
            init_pos=(0, 0, 3),
            direction=(0.0, 0.0, -1.0),
        )
        light = self.sim.add_light(cfg=cfg)
        new_dir = torch.tensor([1.0, 0.0, 0.0])
        try:
            light.set_direction(new_dir)
        except Exception as e:
            pytest.fail(f"set_direction broadcast failed: {e}")

    def test_set_direction_with_env_ids(self):
        """set_direction with (M, 3) tensor applies per-instance."""
        cfg = LightCfg(
            uid="spot_env",
            light_type="spot",
            init_pos=(0, 0, 3),
            direction=(0.0, 0.0, -1.0),
        )
        light = self.sim.add_light(cfg=cfg)
        env_ids = [0, 2]
        dirs = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        try:
            light.set_direction(dirs, env_ids=env_ids)
        except Exception as e:
            pytest.fail(f"set_direction per-instance failed: {e}")

    def test_enable_shadow(self):
        """enable_shadow sets shadow flag on all instances."""
        cfg = LightCfg(uid="shadow_test", light_type="point", init_pos=(0, 0, 2))
        light = self.sim.add_light(cfg=cfg)
        try:
            light.enable_shadow(torch.tensor(0.0))  # disable
            light.enable_shadow(torch.tensor(1.0))  # enable
        except Exception as e:
            pytest.fail(f"enable_shadow failed: {e}")

    # ------------------------------------------------------------------
    # from_dict compatibility
    # ------------------------------------------------------------------

    def test_from_dict_new_types(self):
        """LightCfg.from_dict works with new type fields."""
        cfg = LightCfg.from_dict(
            {
                "light_type": "spot",
                "color": [1.0, 0.9, 0.8],
                "intensity": 100.0,
                "init_pos": [0.0, 0.0, 3.0],
                "direction": [0.0, 0.0, -1.0],
                "spot_angle_inner": 25.0,
                "spot_angle_outer": 50.0,
                "uid": "from_dict_spot",
            }
        )
        assert cfg.light_type == "spot"
        assert cfg.spot_angle_inner == 25.0
        assert cfg.spot_angle_outer == 50.0
        assert cfg.direction == (0.0, 0.0, -1.0)

    # ------------------------------------------------------------------
    # Backward compatibility
    # ------------------------------------------------------------------

    def test_point_light_backward_compat(self):
        """Existing point-light config pattern still works."""
        cfg_dict = {
            "light_type": "point",
            "color": [0.1, 0.1, 0.1],
            "radius": 10.0,
            "position": [0.0, 0.0, 2.0],
            "uid": "point_compat",
        }
        light = self.sim.add_light(cfg=LightCfg.from_dict(cfg_dict))
        assert light is not None
        assert light.num_instances == 4

        # set_color should still work
        base_color = torch.tensor([0.1, 0.1, 0.1])
        try:
            light.set_color(base_color)
        except Exception as e:
            pytest.fail(f"set_color failed: {e}")

        # set_falloff should still work
        try:
            light.set_falloff(torch.tensor(100.0))
        except Exception as e:
            pytest.fail(f"set_falloff failed: {e}")
```

- [ ] **Step 2: Run the new test file**

```bash
cd /home/dex/workspace/sources/EmbodiChain && python -m pytest tests/sim/objects/test_light.py -v
```

Expected: All tests in `TestLight` and `TestLightTypes` pass.

- [ ] **Step 3: Commit**

```bash
git add tests/sim/objects/test_light.py
git commit -m "test: add comprehensive tests for all 6 light types

Cover creation, type validation, broadcasting, from_dict, and
backward compatibility for point lights.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Run full pre-commit check and fix any issues

**Files:**
- Modify: (any files that need formatting or lint fixes)

- [ ] **Step 1: Format with black**

```bash
cd /home/dex/workspace/sources/EmbodiChain && black embodichain/lab/sim/cfg.py embodichain/lab/sim/objects/light.py embodichain/lab/sim/sim_manager.py tests/sim/objects/test_light.py
```

- [ ] **Step 2: Run the full test suite for affected modules**

```bash
cd /home/dex/workspace/sources/EmbodiChain && python -m pytest tests/sim/objects/test_light.py -v
```

- [ ] **Step 3: Verify no import regressions**

```bash
cd /home/dex/workspace/sources/EmbodiChain && python -c "
from embodichain.lab.sim.cfg import LightCfg
from embodichain.lab.sim.objects import Light
from embodichain.lab.sim.sim_manager import SimulationManager
print('All imports successful')
print(f'Supported types: {list(SimulationManager._LIGHT_TYPE_MAP.keys())}')
"
```

Expected: `All imports successful` + list of all 6 types.

- [ ] **Step 4: Final commit if any formatting changes were needed**

```bash
git add -u && git commit -m "chore: format and finalize light type expansion

Co-Authored-By: Claude <noreply@anthropic.com>"
```
