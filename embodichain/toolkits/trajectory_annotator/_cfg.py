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

"""Resource limits for the local trajectory annotation tool."""

from __future__ import annotations

from embodichain.utils import configclass

__all__ = ["AnnotationCfg"]


@configclass
class AnnotationCfg:
    """Configure CPU decoding, preview quality and bounded editing resources.

    Attributes:
        preview_edge: Maximum preview width or height in pixels.
        jpeg_quality: OpenCV JPEG quality, between 1 and 100.
        cache_bytes: Maximum retained JPEG payload bytes; zero disables caching.
        index_bytes: Maximum packed timestamp and keyframe-index bytes.
        decoder_threads: FFmpeg threads per decoder; the owner is one worker.
        autosave_ms: Debounce interval for completed annotation edits.
        document_bytes: Maximum JSON request or annotation file size.
    """

    preview_edge: int = 1280
    jpeg_quality: int = 85
    cache_bytes: int = 64 * 1024**2
    index_bytes: int = 64 * 1024**2
    decoder_threads: int = 2
    autosave_ms: int = 1000
    document_bytes: int = 8 * 1024**2

    def __post_init__(self) -> None:
        for name in (
            "preview_edge",
            "index_bytes",
            "decoder_threads",
            "autosave_ms",
            "document_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if type(self.cache_bytes) is not int or self.cache_bytes < 0:
            raise ValueError("cache_bytes must be a non-negative integer.")
        if type(self.jpeg_quality) is not int or not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be an integer between 1 and 100.")
