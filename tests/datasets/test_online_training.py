# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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

import unittest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from common import UnittestMetaclass, OrderedTestLoader
from embodichain.data.data_engine.online.engine import OnlineEngine
import numpy as np
from tqdm import tqdm
import time


class TestDataDictExtractor(unittest.TestCase, metaclass=UnittestMetaclass):
    datacenter_backup = Path("/tmp/datacenter_test")

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_online_generation(
        self,
    ):
        from embodichain.utils.logger import log_warning
        from embodichain.data.data_engine.online.online_generator import (
            OnlineGenerator,
        )

        log_warning("Start online data generation.")

        online_config = {
            "episode_limit": 4,
            "max_sample_num": 100,
            "port": 5566,
            "buffer_size": 4,
            "max_limit_gb": 5,
        }
        online_callback = OnlineGenerator(**online_config)
        generator_func = lambda **kwargs: [{"data": np.random.randn(1000, 1000)}]
        online_callback.generator(generator_func, loop_times=2)
        online_callback.empty_memory()

    def test_sample_data(self):

        from embodichain.utils.logger import log_warning
        import threading
        from embodichain.data.data_engine.online.online_generator import (
            OnlineGenerator,
        )

        log_warning("Start online data generation.")

        online_config = {
            "episode_limit": 4,
            "max_sample_num": 100,
            "port": 7788,
            "buffer_size": 4,
            "max_limit_gb": 5,
        }
        online_callback = OnlineGenerator(**online_config)
        data_o = np.random.randn(1000, 1000)
        generator_func = lambda **kwargs: [{"data": data_o}]

        thread = threading.Thread(
            target=online_callback.generator,
            kwargs={"generate_func": generator_func, "loop_times": 2},
            daemon=True,
        )
        thread.start()
        time.sleep(1.0)

        callback = OnlineEngine(**online_config)
        callback.start()
        time.sleep(1.0)
        for i in tqdm(range(5)):
            data = callback.sample_data()
            assert data.sum() == data_o.sum()


if __name__ == "__main__":
    # `unittest.main()` is the standard usage to start testing, here we use a customed
    # TestLoader to keep executing order of functions the same as their writing order

    unittest.main(testLoader=OrderedTestLoader())
