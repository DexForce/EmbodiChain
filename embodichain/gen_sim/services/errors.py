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

"""Safe, classified service failures without provider response bodies."""

from __future__ import annotations

__all__: list[str] = []


class ServiceError(RuntimeError):
    """Base service error with a stable classification and attempt count."""

    code = "service_error"

    def __init__(self, message: str, *, attempts: int = 0) -> None:
        super().__init__(message)
        self.attempts = attempts


class ServiceConfigurationError(ServiceError, ValueError):
    """A selected service configuration is incomplete or invalid."""

    code = "configuration"


class CapabilityError(ServiceError):
    """An adapter does not provide the selected capability."""

    code = "capability"


class ServiceTimeout(ServiceError):
    """A request or polling budget expired."""

    code = "timeout"


class ServiceTransportError(ServiceError):
    """The service could not be reached or rejected the transport request."""

    code = "transport"


class ServiceResponseError(ServiceError, ValueError):
    """Invalid model output, recoverable by the owner's bounded value repair.

    Transport and timeout failures remain separate from semantic repair.
    """

    code = "response"


class SubmissionUnknown(ServiceError):
    """A generation submission may have been accepted; do not resubmit."""

    code = "submission_unknown"
