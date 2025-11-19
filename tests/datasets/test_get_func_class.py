from embodichain.lab.gym.envs.action_bank.configurable_action import (
    tag_node,
    get_func_tag,
)


class A:
    def __init__(self) -> None:
        pass

    @staticmethod
    @tag_node
    def whatever():
        pass


assert "A" in list(get_func_tag("node").functions.keys())
