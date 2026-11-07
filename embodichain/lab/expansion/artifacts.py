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

"""Typed, content-verified artifact references rooted inside a portable bundle."""

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path, PurePosixPath
import json
import re
from embodichain.data_pipeline.recording.media import _digest

__all__ = ["ArtifactReference", "EpisodeArtifactManifest", "resolve_artifact_path"]


def resolve_artifact_path(
    root: str | Path, path: str, *, legacy_absolute: bool = False
) -> Path:
    """Resolve a contained bundle reference; legacy absolute addresses are opt-in."""
    root = Path(root).resolve()
    value = PurePosixPath(path)
    if value.is_absolute():
        if legacy_absolute:
            return Path(path)
        raise ValueError("artifact references must be relative to their bundle")
    if not path or ".." in value.parts or "\\" in path:
        raise ValueError("invalid artifact address")
    candidate = root / Path(path)
    if not candidate.resolve().is_relative_to(root):
        raise ValueError("artifact reference escapes its bundle")
    return candidate


@dataclass(frozen=True)
class ArtifactReference:
    """A relative address, explicit role and SHA-256 for one immutable file."""

    path: str
    sha256: str
    role: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.path, str)
            or PurePosixPath(self.path).is_absolute()
            or ".." in PurePosixPath(self.path).parts
            or "\\" in self.path
        ):
            raise ValueError("artifact path must be a contained relative address")
        if (
            not re.fullmatch(r"[0-9a-f]{64}", self.sha256)
            or not self.role
            or not self.path
        ):
            raise ValueError("artifact needs a role and SHA-256")

    def verify(self, root: str | Path) -> Path:
        """Check content at the current bundle location, including after relocation."""
        path = resolve_artifact_path(root, self.path)
        if not path.is_file() or _digest(path) != self.sha256:
            raise ValueError(f"artifact content changed: {self.path}")
        return path


@dataclass(frozen=True)
class EpisodeArtifactManifest:
    """Compose numerical evidence, media/calibration and source attachments.

    Numerical arrays retain ExpertEpisode's T/T+1 contract. A media manifest
    contains calibrated timed chunks; sources retain their original content pins.
    """

    episode_id: str
    commit_id: str
    recipe_id: str | None
    attempt_id: int
    numerical: ArtifactReference
    media: ArtifactReference | None = None
    sources: tuple[ArtifactReference, ...] = ()
    parent_episode_id: str | None = None
    schema_version: int = 2
    source_ancestors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            self.schema_version != 2
            or not self.episode_id
            or not self.commit_id
            or type(self.attempt_id) is not int
            or self.attempt_id < 0
        ):
            raise ValueError("invalid episode artifact identity")
        object.__setattr__(self, "sources", tuple(self.sources))
        ancestors = tuple(self.source_ancestors)
        if len(set(ancestors)) != len(ancestors) or any(
            not isinstance(value, str) or not value for value in ancestors
        ):
            raise ValueError("invalid source ancestry")
        object.__setattr__(self, "source_ancestors", ancestors)
        if self.numerical.path != "episode.npz" or self.numerical.role != "numerical":
            raise ValueError("numerical artifact must identify episode.npz")
        if self.media is not None and self.media.role != "media":
            raise ValueError("media reference has an invalid role")
        refs = [self.numerical, *self.sources, *((self.media,) if self.media else ())]
        if any(not isinstance(ref, ArtifactReference) for ref in refs):
            raise TypeError("artifact entries must be typed references")
        if len({ref.path for ref in refs}) != len(refs):
            raise ValueError("duplicate artifact addresses")

    def to_dict(self) -> dict[str, object]:
        """Return an owned manifest ready for atomic publication."""
        return asdict(self)

    @classmethod
    def read(cls, path: str | Path) -> EpisodeArtifactManifest:
        """Decode a manifest without trusting its referenced content."""
        value = json.loads(Path(path).read_text())
        value["numerical"] = ArtifactReference(**value["numerical"])
        if value.get("media"):
            value["media"] = ArtifactReference(**value["media"])
        value["sources"] = tuple(
            ArtifactReference(**ref) for ref in value.get("sources", ())
        )
        return cls(**value)

    def verify(self, root: str | Path) -> None:
        """Verify all referenced files; media chunk validation remains schema-aware."""
        self.numerical.verify(root)
        if self.media:
            self.media.verify(root)
        for reference in self.sources:
            reference.verify(root)
