# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

import time
import sys
import numpy as np
from threading import Thread
from typing import Dict, Tuple, Any, List, Callable, Optional
from copy import deepcopy
import threading
from embodichain.data.data_engine.online.enum import (
    ConsumerTeleEnum,
    ProducerTeleEnum,
)

import torch
import torch.multiprocessing as mp
import copy
from embodichain.utils.logger import (
    log_info,
    log_warning,
    decorate_str_color,
    log_debug,
)

# Must call cuda init to prevent cuda error in subprocess.
torch._C._cuda_init()

from dexsim.utility import NumpyRNG

import threading
from multiprocessing import shared_memory
import pickle
from datetime import datetime
import zmq

__all__ = ["MaiDataEngine"]

rng = NumpyRNG.get_rng()

log_info_produce = lambda x: log_info(decorate_str_color(x, "cyan"))
log_info_consume = lambda x: log_info(decorate_str_color(x, "orange"))

MAX_LOOP_TIMES = 40000


def init_context(port):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:{}".format(port))
    return socket


class DataPoolCont:
    data: Any
    count: int = 0
    tag: str

    @staticmethod
    def from_list(data_pool: List[Dict]) -> List["DataPoolCont"]:
        ret = []
        for data in data_pool:
            dcnt = DataPoolCont()
            dcnt.data = data
            dcnt.count = 0
            dcnt.tag = str(datetime.now()).split(".")[0]
            ret.append(dcnt)
        return ret

    @staticmethod
    def clean_data_pool_in_place(
        data_pool: List["DataPoolCont"], clean_indices: List[int]
    ):
        if clean_indices is None:
            data_pool = []
        else:
            if len(clean_indices) > 0:
                log_debug(
                    "Clean data pool with data indices {}, counts {}.".format(
                        clean_indices,
                        [data_pool[index].count for index in clean_indices],
                    ),
                    color="purple",
                )
                for i in list(np.sort(clean_indices)[::-1]):
                    data_pool.pop(i)


def fetch_data(
    queue_data: mp.Queue, data_pool: List[DataPoolCont], worker_info, debug: bool = True
) -> bool:
    start_time = time.time()
    try:
        existing_shm = queue_data.get(timeout=5)
    except Exception as error:
        log_debug("Timeout! {}.".format(str(error)), color="red")
        return False
    log_debug(
        "[Thread {}][Worker {}][Get] Cost {}s.".format(
            threading.current_thread().ident,
            worker_info.id,
            time.time() - start_time,
        )
    )
    start_time = time.time()
    scene_data = pickle.loads(existing_shm.buf[:])
    log_debug(
        "[Thread {}][Worker {}][Pickle] Cost {}s.".format(
            threading.current_thread().ident,
            worker_info.id,
            time.time() - start_time,
        )
    )

    if np.random.random() > 0.5 or queue_data.qsize() == 0:
        start_time = time.time()
        queue_data.put(existing_shm)  # put back
        log_debug(
            "[Thread {}][Worker {}][Put] Cost {}s.".format(
                threading.current_thread().ident,
                worker_info.id,
                time.time() - start_time,
            )
        )

    assert isinstance(scene_data, list), "Invalid data format {}.".format(
        type(scene_data)
    )
    start_time = time.time()
    data = DataPoolCont.from_list(scene_data)
    data_pool.extend(data)

    log_debug(
        "[Thread {}][Worker {}][Other] Cost {}s.".format(
            threading.current_thread().ident,
            worker_info.id,
            time.time() - start_time,
        )
    )
    return True


class RestockCriterion:
    def __init__(self, data_pool_limit: int, buffer_size: int, max_sample_num: int):
        self.data_pool_limit = data_pool_limit
        self.buffer_size = buffer_size
        self.max_sample_num = max_sample_num

    def restock_condition(self, data_pool: List, queue: mp.Queue) -> bool:
        return len(data_pool) < self.data_pool_limit

    def expired_condition(
        self, data_pool: List[DataPoolCont], inverse: bool = False
    ) -> List[bool]:

        if len(data_pool) == 0:
            return []

        if inverse:
            return [data.count <= self.max_sample_num for data in data_pool]
        else:
            return [data.count > self.max_sample_num for data in data_pool]


class OnlineEngine:
    """Data manager for online data production and training.

        The objectives of this class are:
        - Manage the fetch data in a separate thread.
        - Perform data synchronization between the data production process and
            the training process (main process).
        - Provide data sampling interface for the training process, which is designed
            to return a batch of synthetic data with the different scene id.
        - Data lifecycle management.

        To achieve the above objectives, the following functions should be implemented:
        - from_shm_thread (static method)

    Args:
        insight_config (List[CfgNode]): The config of insight pipeline.
        episode_limit (int, optional): The maximum number of frames in the data pool. Defaults to 24.
        max_sample_num (int, optional): The maximum number of times that a data can be sampled.
            Defaults to 2.
        target_device (torch.device, optional): The target device of the data. Defaults to torch.device('cpu').
        annos_param (Dict[str, Any], optional): The parameters of the annotations. Defaults to None.
        data_gen_func (Callable, optional): The data generation function. Defaults to None.
        unique_scene_frame (int, optional): The number of unique scene frame to be sampled. Defaults to None.
        port (int, optional): The ZeroMQ socket port. Defaults to 5555.
        buffer_size(int, optional): The number of max data queue size. Defaults to 10.
    """

    def __init__(
        self,
        episode_limit: int = 24,
        max_sample_num: int = 2,
        port: int = 5555,
        buffer_size: int = 10,
        multiprocess: bool = False,
        **kwargs,
    ) -> None:

        self.episode_limit = episode_limit
        self._max_sample_num = max_sample_num
        self.port = port

        self._data_pool = []

        self._duration = 0.01

        self._context = mp.get_context("forkserver")

        self._queue_data = self._context.Queue()
        self._queue_data.cancel_join_thread()

        self.buffer_size = buffer_size

        self._data_gen_proc = None
        self._fetch_data_thread = None
        self._restock_data_pool = None

        self._is_started = False
        self._is_restocked = False
        self._socket = init_context(port + 1 if multiprocess else port)

        self._restock_criterion = RestockCriterion(
            data_pool_limit=episode_limit,
            buffer_size=buffer_size,
            max_sample_num=max_sample_num,
        )
        self._lock = threading.RLock()

    def start(
        self,
    ) -> None:
        """Start the data production process and the data synchronization thread.

        Args:
            wait_for_limit (bool, optional): Whether to wait for the data pool to reach
                the frame limit. Defaults to False.
        """

        self._signal_gen = self._context.Value("b", True)
        self._signal_fetch = self._context.Value("b", True)

        self._fetch_data_thread = Thread(
            target=self.from_shm_thread,
            args=(
                self._socket,
                self._queue_data,
                self._duration,
                self.buffer_size,
            ),
            daemon=True,
        )
        self._fetch_data_thread.start()
        self._is_started = True
        log_info(
            "Now start the thread to fetch data from share memory.", color="purple"
        )

    def start_restock(self, static: bool = False):
        if static:
            self._restock_data_pool = Thread(
                target=self.restock_data_pool_static,
                args=(
                    self._data_pool,
                    self._queue_data,
                    self._duration,
                    self._restock_criterion,
                    self._context,
                    self._lock,
                ),
                daemon=True,
            )
        else:
            self._restock_data_pool = Thread(
                target=self.restock_data_pool,
                daemon=True,
            )

        self._restock_data_pool.start()
        self._is_restocked = True

    def stop(self) -> None:
        if self.is_started:
            self._is_started = False
            self._signal_fetch.value = 2
            self._fetch_data_thread.join()
            self.empty_queue(self._queue_data, self._context)
            self.clean_data_pool_in_place()
            self._signal_gen.value = 2
        else:
            log_info(
                "The data generation process has not been started.", color="purple"
            )

    @property
    def is_started(self) -> bool:
        return self._is_started

    @property
    def data_size(self) -> int:
        with self._lock:
            return len(self._data_pool)

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    @property
    def unique_scene_frame(self) -> int:
        return self._unique_scene_frame

    @staticmethod
    def empty_queue(queue: mp.Queue, context: mp) -> None:
        while queue.qsize() > 0:
            try:
                queue.get()
            except Exception as e:
                log_info("queue put invaild data format")
                queue.close()
                queue.join_thread()
                queue = context.Queue()
                break
        return queue

    @staticmethod
    def empty_share_memory(queue: mp.Queue) -> None:
        while queue.qsize() > 0:
            shm_name = queue.get()
            shm = shared_memory.SharedMemory(shm_name)
            shm.close()
            shm.unlink()

    def restock_data_pool(self):
        return OnlineEngine.restock_data_pool_static(
            self._data_pool,
            self._queue_data,
            self._duration,
            self._restock_criterion,
            self._context,
            self._lock,
        )

    @staticmethod
    def restock_data_pool_static(
        data_pool: List[DataPoolCont],
        queue_data: mp.Queue,
        duration: float,
        restock_criterion: RestockCriterion,
        context,
        thread_lock,
    ):
        counts = 0

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:

            class FakeWorkerInfo:
                num_workers = 1
                id = 0

            worker_info = FakeWorkerInfo()

        while True:
            time.sleep(duration)
            # always clean the data pool first.

            start_time = time.time()
            with thread_lock:
                # delete
                clean_indices = list(
                    np.argwhere(restock_criterion.expired_condition(data_pool)).reshape(
                        -1
                    )
                )
                DataPoolCont.clean_data_pool_in_place(
                    data_pool,
                    clean_indices,
                )
            if len(clean_indices) > 0:
                log_debug(
                    "[Thread {}][Delete][Cost {}s]".format(
                        threading.current_thread().ident, time.time() - start_time
                    )
                )

            # after clean, we check whether to restock data.
            while restock_criterion.restock_condition(data_pool, queue_data):

                prev_data_size = len(data_pool)
                should_fetch = False
                for i in range(worker_info.num_workers):
                    if queue_data.qsize() > 0 and worker_info.id == i:
                        should_fetch = True
                if should_fetch:
                    start_time = time.time()
                    with thread_lock:
                        # add
                        fetch_data(
                            data_pool=data_pool,
                            queue_data=queue_data,
                            worker_info=worker_info,
                        )
                    log_debug(
                        "[Thread {}][Worker {}][ToDataPool] Produce data: {}->{}. Cost {}s.".format(
                            threading.current_thread().ident,
                            worker_info.id,
                            prev_data_size,
                            len(data_pool),
                            time.time() - start_time,
                        )
                    )
                    counts = 0
                else:
                    counts += 1

            if counts % MAX_LOOP_TIMES == 0 and counts != 0:
                log_info("Can not find the shm after {} times.".format(counts))
            # queue_data = OnlineEngine.empty_queue(queue_data, context)

    @staticmethod
    def from_shm_thread(
        socket,
        queue_data: mp.Queue,
        duration: float = 0.001,
        buffer_size: int = 10,
    ) -> None:
        """The data fetching thread for data synchronization.

            The queue_data_size is used to control the data fetching thread.
            If  queue_data_size < buffer_size, the data fetching thread will fetch data from the queue.
            If queue_data_size >= buffer_size, the data fetching thread will stop fetch data.

        Args:
            socket (zmq.Context): The socket send signal for connect fetch and generator.
            queue_data (mp.Queue): This queue contains information about shared memory.
            duration (float, optional): _description_. Defaults to 0.001.
            port (int, optional): The ZeroMQ socket port. Defaults to 5555.
            buffer_size(int, optional): The number of max data queue size. Defaults to 10.
        """
        counts = 0
        while True:
            time.sleep(duration)
            counts += 1
            if queue_data.qsize() < buffer_size:
                socket.send_string(ConsumerTeleEnum.SHAKEHAND.value)
                message = socket.recv()
                try:
                    message_str = message.decode()
                except Exception as e:
                    log_debug(str(e), color="red")
                    message_str = ""
                if message_str != ProducerTeleEnum.NOREADY.value:
                    log_debug("Receive data.", color="purple")
                    shm_name = pickle.loads(message).popleft()
                    existing_shm = shared_memory.SharedMemory(name=shm_name)
                    queue_data.put(existing_shm)
                    log_debug(
                        "[FromShmThread] Produce queue: {}->{};".format(
                            queue_data.qsize() - 1, queue_data.qsize()
                        )
                    )
            else:
                if counts % MAX_LOOP_TIMES == 0:
                    log_debug("Queue is full. Skip this stage.", "purple")

    def sample_data(
        self,
    ):

        if self._is_restocked:
            pass
        else:
            log_debug("Now start the thread to restock data.", color="purple")
            self.start_restock(static=False)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:

            class FakeWorkerInfo:
                num_workers = 1
                id = 0

            worker_info = FakeWorkerInfo()

        counts = 0
        while True:
            time.sleep(self._duration)
            if len(self._data_pool) > 0:
                start_time = time.time()
                with self._lock:
                    index = rng.integers(0, len(self._data_pool))
                    data = self._data_pool[index]
                    self._data_pool[index].count += 1
                log_debug(
                    "[SampleData, worker {}] Consume data {}: index {}; times: {}->{}; Show queue size: {}; Cost time: {}s.".format(
                        worker_info.id,
                        data.tag,
                        index,
                        data.count,
                        data.count + 1,
                        self._queue_data.qsize(),
                        np.round(time.time() - start_time, 4),
                    )
                )
                counts = 0
                return data.data
            else:
                counts += 1
            if counts % MAX_LOOP_TIMES == 0:
                log_info("Data pool is always empty after {} times.".format(counts))
