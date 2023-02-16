from torch import nn


class GeneralFusor(nn.Module):
    def __init__(self, name) -> None:
        self.name = name
        super().__init__()


class NoActionFusor(GeneralFusor):
    def __init__(self) -> None:
        super().__init__("fusor-no_action")

    def forward(self, x):
        assert len(
            x.keys()) == 1, "should only have one element in no action fusor"

        out = x[list(x.keys())[0]]
        return {"z": out}

class ElementwiseSumFusor(GeneralFusor):
    def __init__(self) -> None:
        super().__init__("fusor-elementwise")

    def forward(self, x):
        return {"z": sum(list(x.values()))}
