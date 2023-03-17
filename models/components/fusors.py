from torch import nn
import torch

from models.components.general import Conv2dBNReLu


class GeneralFusor(nn.Module):
    def __init__(self, name, out_channel) -> None:
        self.name = name
        self.out_channel = out_channel
        super().__init__()


class NoActionFusor(GeneralFusor):
    def __init__(self, out_channel) -> None:
        super().__init__(
            name="fusor-no_action",
            out_channel=out_channel,
        )

    def forward(self, x):
        assert len(x.keys()) == 1, "should only have one element in no action fusor"

        out = x[list(x.keys())[0]]
        return {"z": out}


class ElementwiseSumFusor(GeneralFusor):
    def __init__(self, out_channel) -> None:
        super().__init__("fusor-elementwise_sum", out_channel=out_channel)

    def forward(self, x):
        return {"z": sum(list(x.values()))}


class HadamardProductFusor(GeneralFusor):
    def __init__(self, out_channel) -> None:
        super().__init__("fusor-hadamard_product", out_channel)

    def forward(self, x):
        output = torch.ones(list(x.values())[0].shape)

        for v in x.values():
            output *= v

        return {"z": output}


class ConcatenationFusor(GeneralFusor):
    def __init__(self, in_channels, out_channel) -> None:
        super().__init__("fusor_concatenation", out_channel)
        self.model = Conv2dBNReLu(in_channels, out_channel)

    def forward(self, x):
        return {"z": self.model(torch.concat(list(x.values()), axis=1))}
