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

"""Rendering configuration for window and offscreen image processing."""

from __future__ import annotations

import math

from dataclasses import field, fields
from numbers import Real
from typing import Literal

import dexsim
from dexsim.types import RTRenderMode, Renderer, ToneMappingType

from embodichain.utils import configclass, logger

__all__ = [
    "DenoisingCfg",
    "DenoisingMode",
    "DLSSCfg",
    "NRDCfg",
    "RenderCfg",
]

DenoisingMode = Literal["off", "optix", "dlss", "nrd"]
"""Public ray-tracing denoising and reconstruction mode."""

_DENOISING_MODES: dict[DenoisingMode, RTRenderMode] = {
    "off": RTRenderMode.RAW,
    "optix": RTRenderMode.OPTIX_DENOISE,
    "dlss": RTRenderMode.DLSS_RR,
    "nrd": RTRenderMode.NRD_RELAX,
}


@configclass
class DenoisingCfg:
    """Select denoising/reconstruction independently by rendering scope.

    The public values describe complete image-processing paths. Native NRD
    method variants remain an implementation detail instead of becoming
    additional user-facing modes.
    """

    window: DenoisingMode = "dlss"
    """Pipeline used by the interactive window."""

    offscreen: DenoisingMode = "dlss"
    """Pipeline used by offscreen camera targets."""

    def __post_init__(self) -> None:
        """Reject values outside the stable public mode contract."""
        for name in ("window", "offscreen"):
            value = getattr(self, name)
            if type(value) is not str or value not in _DENOISING_MODES:
                choices = ", ".join(repr(item) for item in _DENOISING_MODES)
                raise ValueError(
                    f"DenoisingCfg.{name} must be one of {choices}; got {value!r}."
                )

    def to_dexsim_modes(self) -> tuple[RTRenderMode, RTRenderMode]:
        """Convert window and offscreen selections to DexSim modes.

        Returns:
            Window and offscreen modes, in that order.

        Raises:
            ValueError: If mutable settings no longer contain valid modes.
        """
        self.__post_init__()
        return _DENOISING_MODES[self.window], _DENOISING_MODES[self.offscreen]


@configclass
class NRDCfg:
    """Algorithm settings shared by targets that use the NRD path.

    Pipeline selection belongs to :class:`DenoisingCfg`. This class only owns
    NRD tuning values and intentionally does not expose RELAX/REBLUR selection.

    The fields are grouped by how directly they affect the current public
    ``nrd`` path (standalone NRD RELAX): history/disocclusion controls and the
    post-NRD TAA controls are active-path settings. Reblur-specific blur,
    stabilization, and anti-lag controls are retained for native parity but
    can be deferred from a smaller public configuration surface because the
    current EmbodiChain mode does not select NRD REBLUR. Validation is a debug
    feature, and SH/SG mode only has an effect on RELAX with FastRT or OfflineRT.
    """

    max_indirect_bounces: int = 1
    """Number of indirect continuation rays after the primary surface.

    Higher values increase lighting detail and ray cost; NRD's real-time
    default is one continuation bounce.
    """

    denoising_range: float = 500000.0
    """Maximum world-space range considered by NRD's denoising passes."""

    disocclusion_threshold: float = 0.01
    """Base view-depth threshold used to reject invalid temporal history."""

    disocclusion_threshold_alternate: float = 0.05
    """Alternate depth threshold for pixels selecting the mixed threshold."""

    disocclusion_threshold_mix_enabled: bool = True
    """Use the per-pixel threshold-mix input for thin or transparent geometry."""

    history_confidence_enabled: bool = True
    """Generate paired-history confidence from matched probe renders."""

    history_confidence_probe_stride: int = 5
    """Full-resolution spacing of paired-history probe rays; native range is 1–8."""

    history_confidence_sigma_scale: float = 2.0
    """Sigma scale used when converting history gradients into confidence."""

    history_confidence_sensitivity: float = 1.0
    """Sensitivity used when converting history gradients into confidence."""

    max_accumulated_frame_num: int = 30
    """Maximum slow temporal history length for RELAX and REBLUR."""

    max_fast_accumulated_frame_num: int = 6
    """Maximum fast temporal history length used while a pixel is changing."""

    history_fix_frame_num: int = 3
    """Number of frames used to repair history after a disocclusion."""

    diffuse_prepass_blur_radius: float = 30.0
    """Diffuse prepass blur radius; larger values reduce noise and soften detail."""

    specular_prepass_blur_radius: float = 50.0
    """Specular prepass blur radius; mainly an advanced REBLUR tuning knob."""

    anti_firefly_enabled: bool = False
    """Suppress isolated bright history samples; may darken sparse highlights."""

    sh_mode_enabled: bool = False
    """Enable RELAX directional SH/SG signals on FastRT and OfflineRT paths."""

    validation_enabled: bool = False
    """Enable NRD validation diagnostics; intended for development builds."""

    max_stabilized_frame_num: int = 63
    """Maximum stabilized history for REBLUR; inactive for standalone RELAX."""

    min_blur_radius: float = 1.0
    """Minimum adaptive blur radius for REBLUR; inactive for standalone RELAX."""

    max_blur_radius: float = 30.0
    """Maximum adaptive blur radius for REBLUR; inactive for standalone RELAX."""

    antilag_luminance_sigma_scale: float = 2.0
    """REBLUR Anti-Lag luminance sigma scale; inactive for standalone RELAX."""

    antilag_luminance_sensitivity: float = 3.0
    """REBLUR Anti-Lag luminance sensitivity; inactive for standalone RELAX."""

    taa_min_current_weight: float = 1.0 / 16.0
    """Minimum current-frame weight in the post-NRD temporal resolve."""

    taa_sigma_scale: float = 2.0
    """Sigma scale used by the post-NRD temporal resolve."""

    taa_depth_rejection_enabled: bool = True
    """Reject post-NRD history when reprojected view depth does not match."""

    taa_tone_mapping_enabled: bool = False
    """Apply NRD's optional tone mapping before the final EmbodiChain output."""

    def __post_init__(self) -> None:
        """Validate scalar types, numeric ranges, and radius ordering."""
        boolean_fields = (
            "disocclusion_threshold_mix_enabled",
            "history_confidence_enabled",
            "anti_firefly_enabled",
            "sh_mode_enabled",
            "validation_enabled",
            "taa_depth_rejection_enabled",
            "taa_tone_mapping_enabled",
        )
        for name in boolean_fields:
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"NRDCfg.{name} must be a boolean.")

        non_negative_integer_fields = (
            "max_indirect_bounces",
            "max_accumulated_frame_num",
            "max_fast_accumulated_frame_num",
            "history_fix_frame_num",
            "max_stabilized_frame_num",
        )
        for name in non_negative_integer_fields:
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"NRDCfg.{name} must be a non-negative integer.")
        if (
            type(self.history_confidence_probe_stride) is not int
            or self.history_confidence_probe_stride < 1
        ):
            raise ValueError(
                "NRDCfg.history_confidence_probe_stride must be a positive integer."
            )

        positive_fields = (
            "denoising_range",
            "history_confidence_sigma_scale",
            "taa_sigma_scale",
        )
        non_negative_fields = (
            "disocclusion_threshold",
            "disocclusion_threshold_alternate",
            "history_confidence_sensitivity",
            "diffuse_prepass_blur_radius",
            "specular_prepass_blur_radius",
            "min_blur_radius",
            "max_blur_radius",
            "antilag_luminance_sigma_scale",
            "antilag_luminance_sensitivity",
        )
        for name in positive_fields + non_negative_fields:
            value = getattr(self, name)
            minimum = 0.0
            invalid = (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value < minimum
                or (name in positive_fields and value == minimum)
            )
            if invalid:
                qualifier = "positive" if name in positive_fields else "non-negative"
                raise ValueError(f"NRDCfg.{name} must be a {qualifier}, finite number.")
        if (
            isinstance(self.taa_min_current_weight, bool)
            or not isinstance(self.taa_min_current_weight, Real)
            or not math.isfinite(self.taa_min_current_weight)
            or not 0.0 <= self.taa_min_current_weight <= 1.0
        ):
            raise ValueError(
                "NRDCfg.taa_min_current_weight must be a finite number from 0.0 to 1.0."
            )
        if self.min_blur_radius > self.max_blur_radius:
            raise ValueError(
                "NRDCfg.min_blur_radius must not exceed NRDCfg.max_blur_radius."
            )

    def to_dexsim_cfg(self) -> dexsim.NRDConfig:
        """Convert the settings to DexSim's native NRD configuration.

        Returns:
            Populated :class:`dexsim.NRDConfig`.

        Raises:
            ValueError: If mutable settings no longer contain valid values.
        """
        self.__post_init__()
        nrd = dexsim.NRDConfig()
        for item in fields(self):
            setattr(nrd, item.name, getattr(self, item.name))
        return nrd


@configclass
class DLSSCfg:
    """DLSS settings for the public ``dlss`` rendering path.

    Pipeline selection belongs to :class:`DenoisingCfg`. These controls apply
    when a target selects ``"dlss"``.

    .. attention::
        DLSS requires a Vulkan render device, a compatible NVIDIA GPU/driver,
        and a DexSim build with the NGX runtime. Initialization is deferred
        until rendering; configuration conversion alone cannot verify support.
        Each enabled offscreen camera needs its own temporal history and
        Vulkan exchange images, increasing GPU memory use.
    """

    dlss_quality: int = 2
    """Quality mode and derived internal scale: ``-1`` auto (58%), ``0`` Ultra
    Performance (~33%), ``1`` Performance (50%), ``2`` Balanced (58%),
    ``3`` Quality (~67%), ``4`` Ultra Quality (77%), ``5`` DLAA (100%)."""

    upsample_ratio: float | None = None
    """Optional window target/render ratio, at least 1.0. None leaves zero
    render dimensions for DexSim to derive from quality. When specified,
    computes each unset render dimension from the actual window size. Only
    FastRT/OfflineRT windows honor these overrides; hybrid and offscreen
    targets derive their internal resolution from quality."""

    render_width: int = 0
    """Internal FastRT/OfflineRT window width; zero derives it from quality."""

    render_height: int = 0
    """Internal FastRT/OfflineRT window height; zero derives it from quality."""

    target_width: int = 0
    """DexSim compatibility field. Set the actual window or camera width instead."""

    target_height: int = 0
    """DexSim compatibility field. Set the actual window or camera height instead."""

    tiled_enabled: bool = True
    """Evaluate compatible multi-camera targets through one tiled DLSS atlas."""

    exposure_compensation: float = 1.0
    """Positive, finite exposure multiplier used by the RR bridge."""

    def __post_init__(self) -> None:
        """Validate scalar types and the ranges of numeric settings."""
        if type(self.tiled_enabled) is not bool:
            raise ValueError("DLSSCfg.tiled_enabled must be a boolean.")
        if type(self.dlss_quality) is not int or not -1 <= self.dlss_quality <= 5:
            raise ValueError("DLSSCfg.dlss_quality must be an integer from -1 to 5.")
        for name in (
            "render_width",
            "render_height",
            "target_width",
            "target_height",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"DLSSCfg.{name} must be a non-negative integer.")
        if self.upsample_ratio is not None and (
            isinstance(self.upsample_ratio, bool)
            or not isinstance(self.upsample_ratio, Real)
            or not math.isfinite(self.upsample_ratio)
            or self.upsample_ratio < 1.0
        ):
            raise ValueError(
                "DLSSCfg.upsample_ratio must be a finite number of at least 1.0."
            )
        if (
            isinstance(self.exposure_compensation, bool)
            or not isinstance(self.exposure_compensation, Real)
            or not math.isfinite(self.exposure_compensation)
            or self.exposure_compensation <= 0.0
        ):
            raise ValueError(
                "DLSSCfg.exposure_compensation must be a positive, finite number."
            )

    def to_dexsim_cfg(self, window_width: int, window_height: int) -> dexsim.DLSSConfig:
        """Convert settings without changing the window or camera output size.

        Args:
            window_width: Window width in pixels.
            window_height: Window height in pixels.

        Returns:
            Populated :class:`dexsim.DLSSConfig` instance ready to assign to
            ``world_config.dlss_config``.

        Raises:
            ValueError: If the configuration contains invalid values.
        """
        self.__post_init__()
        dlss = dexsim.DLSSConfig()
        dlss.dlss_quality = self.dlss_quality
        dlss.render_width = self.render_width
        dlss.render_height = self.render_height
        if self.upsample_ratio is not None:
            if self.render_width == 0:
                dlss.render_width = max(1, int(window_width / self.upsample_ratio))
            if self.render_height == 0:
                dlss.render_height = max(1, int(window_height / self.upsample_ratio))
        dlss.target_width = self.target_width
        dlss.target_height = self.target_height
        dlss.tiled_enabled = self.tiled_enabled
        dlss.exposure_compensation = self.exposure_compensation
        return dlss


@configclass
class RenderCfg:
    renderer: Literal["auto", "no-render", "hybrid", "fast-rt", "rt"] = "auto"
    """Renderer backend to use for the simulation.

    Note:
    - 'no-render' selects DexSim's NoRender backend and requires headless mode.
        Native camera sensors require 'hybrid', 'fast-rt', or 'rt'.
    - 'auto' lets task environments select 'no-render' before World creation when
        they have no enabled cameras, recording cameras, window, Viser or explicit
        rendering demand. Custom camera/visual code must explicitly select a native
        renderer or reserve rendering through its environment's demand hook.
        Standalone simulations without a demand declaration retain GPU-based selection.
        Explicit renderer selections and global defaults retain precedence.
    - GPU-based auto selection: RTX-series cards use
        'hybrid', while datacenter cards (A100/A800, H100/H800/H200/H20) use 'fast-rt'.
        If no CUDA device is available or the GPU is unknown, it falls back to 'hybrid'.
    - 'hybrid' uses ray tracing for shadows and reflections while keeping rasterization for primary rendering,
        providing a balance between performance and visual quality.
    - 'fast-rt' is a fully ray-traced renderer for maximum visual fidelity, but may have higher computational cost.
    - 'rt' is an offline ray-traced renderer for maximum visual fidelity, suitable for high-quality rendering tasks.
    """

    spp: int = 1
    """Samples per pixel for ray tracing rendering. This parameter is only valid when renderer is 'hybrid', 'fast-rt' or 'rt'."""

    denoising: DenoisingCfg = field(default_factory=DenoisingCfg)
    """Window and offscreen denoising/reconstruction pipeline selection."""

    dlss: DLSSCfg = field(default_factory=DLSSCfg)
    """DLSS settings used when a target selects ``dlss``."""

    nrd: NRDCfg = field(default_factory=NRDCfg)
    """NRD settings shared by targets using standalone NRD."""

    tone_mapping_enabled: bool = False
    """Whether to map HDR RGB output with the modified Reinhard curve."""

    tone_mapping_exposure: float = 1.0
    """Fixed linear exposure multiplier applied before tone mapping."""

    min_bounces: int = 4
    """Path depth at which Russian Roulette termination may begin.

    Rays can still terminate earlier on a miss or absorption. The primary
    segment has depth zero; this value does not guarantee a minimum path length.
    """

    max_bounces: int = 8
    """Maximum traced path depth, counting the primary segment.

    This bounds ordinary ray-tracing paths and NRD transparent continuations.
    NRD's indirect-lighting budget is controlled separately by
    :attr:`NRDCfg.max_indirect_bounces`.
    """

    def __post_init__(self) -> None:
        """Validate rendering parameters."""
        if self.spp < 1:
            logger.log_error("RenderCfg.spp must be at least 1.", ValueError)
        if self.tone_mapping_exposure < 0.0:
            logger.log_error(
                "RenderCfg.tone_mapping_exposure must be non-negative.", ValueError
            )
        if type(self.min_bounces) is not int or self.min_bounces < 0:
            raise ValueError("RenderCfg.min_bounces must be a non-negative integer.")
        if type(self.max_bounces) is not int or self.max_bounces < 1:
            raise ValueError("RenderCfg.max_bounces must be a positive integer.")
        if self.min_bounces > self.max_bounces:
            raise ValueError(
                "RenderCfg.min_bounces must not exceed RenderCfg.max_bounces."
            )

    def to_dexsim_flags(self) -> Renderer:
        """Convert the renderer name to DexSim's renderer enum."""
        if self.renderer == "no-render":
            return Renderer.NORENDER
        elif self.renderer == "hybrid":
            return Renderer.HYBRID
        elif self.renderer == "fast-rt":
            return Renderer.FASTRT
        elif self.renderer == "rt":
            return Renderer.OFFLINERT
        elif self.renderer == "auto":
            # 'auto' is normally resolved by the SimulationManager before this is
            # called. If it reaches here (e.g. used standalone), fall back safely.
            logger.log_warning(
                "Renderer 'auto' was not resolved before converting to dexsim flags. "
                "Falling back to 'hybrid'."
            )
            return Renderer.HYBRID
        else:
            logger.log_error(
                f"Invalid renderer type '{self.renderer}' specified. Must be one of 'auto', 'no-render', 'hybrid', 'fast-rt', or 'rt'."
            )

    def apply_to_dexsim_config(self, world_config: dexsim.WorldConfig) -> None:
        """Apply rendering settings to a DexSim world configuration.

        Args:
            world_config: DexSim world configuration to update in place.

        Raises:
            ValueError: If rendering settings contain invalid values, including
                bounce limits changed after construction.
        """
        self.__post_init__()
        world_config.renderer = self.to_dexsim_flags()
        if self.renderer == "no-render":
            return
        window_mode, offscreen_mode = self.denoising.to_dexsim_modes()
        set_rt_render_modes = getattr(world_config, "set_rt_render_modes", None)
        if callable(set_rt_render_modes):
            set_rt_render_modes(window=window_mode, offscreen=offscreen_mode)
        else:
            # DexSim 0.5.0 exposed the pipeline modes as nested fields.  Keep
            # that fallback for older native configs and lightweight test
            # doubles that only model the fields used by a specific test.
            pipeline = getattr(world_config, "rt_pipeline_config", None)
            if pipeline is not None:
                pipeline.window.mode = window_mode
                pipeline.offscreen.mode = offscreen_mode
        world_config.dlss_config = self.dlss.to_dexsim_cfg(
            window_width=world_config.win_config.width,
            window_height=world_config.win_config.height,
        )
        world_config.nrd_config = self.nrd.to_dexsim_cfg()
        world_config.raytrace_config.render_iterations_per_frame = self.spp
        world_config.raytrace_config.min_bounces = self.min_bounces
        world_config.raytrace_config.max_bounces = self.max_bounces
        world_config.postprocess_config.tone_mapping_enabled = self.tone_mapping_enabled
        world_config.postprocess_config.tone_mapping_type = (
            ToneMappingType.MODIFIED_REINHARD
        )
        world_config.postprocess_config.tone_mapping_exposure = (
            self.tone_mapping_exposure
        )
