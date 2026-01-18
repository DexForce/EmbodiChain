from abc import ABCMeta, abstractmethod
import os
import cv2
from embodichain.utils.utility import load_json


class ToolkitsBase(metaclass=ABCMeta):
    @classmethod
    def from_config(cls, path: str):
        assert (
            os.path.basename(path).split(".")[-1] == "json"
        ), "only json file is supported."
        config = load_json(path)
        return config["ToolKits"][cls.__name__]

    @abstractmethod
    def call(self, **kwargs):
        pass
