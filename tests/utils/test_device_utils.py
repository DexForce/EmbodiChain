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

from __future__ import annotations

import pytest
import torch

from embodichain.utils.device_utils import standardize_device_string


class TestStandardizeDeviceString:
    def test_cpu_string(self):
        assert standardize_device_string("cpu") == "cpu"

    def test_cpu_torch_device(self):
        assert standardize_device_string(torch.device("cpu")) == "cpu"

    def test_cuda_without_index(self):
        assert standardize_device_string("cuda") == "cuda:0"

    def test_cuda_without_index_torch_device(self):
        assert standardize_device_string(torch.device("cuda")) == "cuda:0"

    def test_cuda_zero(self):
        assert standardize_device_string("cuda:0") == "cuda:0"

    def test_cuda_zero_torch_device(self):
        assert standardize_device_string(torch.device("cuda:0")) == "cuda:0"

    def test_cuda_non_zero_index(self):
        assert standardize_device_string("cuda:4") == "cuda:4"

    def test_cuda_non_zero_index_torch_device(self):
        assert standardize_device_string(torch.device("cuda:4")) == "cuda:4"
