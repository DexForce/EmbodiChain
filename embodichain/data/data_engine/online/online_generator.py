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

import torch
import time
import zmq
import random
from multiprocessing import shared_memory
import pickle
from collections import deque
from typing import List
from threading import Thread
import multiprocessing as mp
import traceback
from embodichain.utils.logger import log_info, log_warning, log_error, log_debug
from embodichain.data.data_engine.online.enum import (
    ConsumerTeleEnum,
    ProducerTeleEnum,
)

torch._C._cuda_init()


class OnlineGenerator:
    """Callback collection for online training mode."""

    def __init__(
        self, port: int, max_limit_gb: int = 50, multiprocess: bool = False, **kwargs
    ) -> None:
        self.shm_val = None
        max_limit = max_limit_gb * 1024**3
        self._context = mp.get_context("forkserver")
        self.port = port
        self.socket = self.init_context(self.port, multiprocess)
        self._duration = 0.01
        self.queue = deque()
        self.queue_memroy = deque()
        self.max_limit = max_limit

        self.validation_config = kwargs.get("validation", {})

    def get_validation_config(self):
        return self.validation_config

    def init_context(self, port, multiprocess: bool = False):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        if multiprocess:
            socket.connect(f"tcp://127.0.0.1:{port}")
        else:
            socket.bind(f"tcp://*:{port}")

        return socket

    def generator(self, generate_func, loop_times: int = -1, **kwargs):
        self.signal = self._context.Value("b", True)

        self._zmq_send = Thread(
            target=self.zmq_send, args=(self.queue, self.signal), daemon=True
        )
        self._zmq_send.start()
        log_debug("Start zmq sending.")
        scene_id = 0

        # -1 means infinite loop
        while scene_id < loop_times or loop_times == -1:
            if self.signal.value == 1:
                first_time = True
                try:
                    t0 = time.time()
                    return_list = generate_func(
                        time_id=scene_id, **self.validation_config
                    )

                    # TODO: support multiple trajectories for each scene.
                    if len(return_list) > 1:
                        log_error(
                            "Only support one trajectory for each scene in online generation mode."
                        )

                    data_dict_list = [return_list[0]["data"]]

                    if (
                        scene_id == 0
                        and self.validation_config.get("num_samples", 0) > 0
                        and "data_path" in return_list[0]
                    ):
                        # create shared memory to store the validation dataset path, which will be accessed by training process.
                        import sys

                        data_path = return_list[0]["data_path"]

                        shared_name = self.validation_config.get(
                            "dataset_name", "val_data_path"
                        )
                        log_info(
                            f"Create shared memory for validation data path: {shared_name}",
                            color="green",
                        )
                        self.shm_val = shared_memory.SharedMemory(
                            name=shared_name,
                            create=True,
                            size=len(data_path.encode()) + sys.getsizeof(""),
                        )
                        self.shm_val.buf[: len(data_path.encode())] = data_path.encode()
                        log_info(
                            f"Craete shared memory for validation data path: {data_path}"
                        )

                    log_info(
                        f"Generate scene {scene_id + 1} time cost: {time.time() - t0}"
                    )
                    serialized_data = pickle.dumps(data_dict_list)
                    shm = shared_memory.SharedMemory(
                        create=True, size=len(serialized_data)
                    )
                    self.queue.append(shm.name)
                    self.queue_memroy.append(
                        {"name": shm.name, "size": len(serialized_data)}
                    )
                    shm.buf[: len(serialized_data)] = serialized_data
                except Exception as e:
                    log_error(f"Error in data generation process: {e}.")
                    traceback.print_exc()
                    self._zmq_send.join()
                    break
                scene_id += 1
                self.empty_memory()
            elif self.signal.value == 0:
                if first_time:
                    log_warning("zmq recive full signal, wait generator signal")
                    first_time = False
                log_warning("Signal value is 0.")
                time.sleep(self._duration)
                continue
            else:
                log_error("Unknown signal, data generator stop")
                break

    def zmq_send(self, queue, signal):
        while True:
            try:
                message = self.socket.recv_string()
                if message == ConsumerTeleEnum.SHAKEHAND.value:
                    if len(queue) > 0:
                        log_warning(
                            "Recieve {} and send [data] to consumer.".format(message)
                        )
                        self.socket.send(pickle.dumps(queue))
                        queue.clear()
                    else:
                        self.socket.send(ProducerTeleEnum.NOREADY.value.encode())
                    signal.value = 1
            except Exception as e:
                print(e)
                traceback.print_exc()
                break

    def empty_memory(self):
        total_size = sum([x["size"] for x in self.queue_memroy])
        log_info(f"share memory size is {total_size/(1024**3)} GB")
        while total_size >= self.max_limit:
            shm_name = self.queue_memroy.popleft()
            if shm_name["name"] in self.queue:
                log_info(f"remove {shm_name['name']} from queue")
                self.queue.remove(shm_name["name"])
            try:
                shm = shared_memory.SharedMemory(shm_name["name"])
            except:
                continue
            shm.close()
            shm.unlink()
            total_size = sum([x["size"] for x in self.queue_memroy])

    def __del__(self):
        if self.shm_val:
            self.shm_val.close()
            self.shm_val.unlink()
