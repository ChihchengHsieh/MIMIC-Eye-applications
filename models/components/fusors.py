from torch import nn
import torch

from models.components.general import Conv2dBNReLu

def get_fusing_tensor_and_pass_through(x: dict[dict: torch.Tensor]):
    fusing_tensors = []
    pass_through = {}
    for k, v in x.items():
        for vk, vv in v.items():
            if vk.endswith("_f"):
                # print(f"fusing {k}_{vk}") # ensuring the
                fusing_tensors.append(vv)
            else:
                pass_through[f"{k}_{vk}"] = vv
    return fusing_tensors, pass_through


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

    def forward(self, x: dict[dict: torch.Tensor]):
        assert len(x.keys()) == 1, "should only have one element in no action fusor"
        fusing_tensor, pass_through= get_fusing_tensor_and_pass_through(x)
        return {"z": fusing_tensor[0], **pass_through}


class ElementwiseSumFusor(GeneralFusor):
    def __init__(self, out_channel) -> None:
        super().__init__("fusor-elementwise_sum", out_channel=out_channel)

    def forward(self, x):
        fusing_tensor, pass_through= get_fusing_tensor_and_pass_through(x)
        return {"z": sum(fusing_tensor), **pass_through}


class HadamardProductFusor(GeneralFusor):
    def __init__(self, out_channel) -> None:
        super().__init__("fusor-hadamard_product", out_channel)

    def forward(self, x):
        fusing_tensor, pass_through= get_fusing_tensor_and_pass_through(x)

        output = torch.ones(fusing_tensor[0].shape)
        for v in fusing_tensor():
            output *= v

        return {"z": output, **pass_through}

class ConcatenationFusor(GeneralFusor):
    def __init__(self, in_channels, out_channel) -> None:
        super().__init__("fusor_concatenation", out_channel)
        self.model = Conv2dBNReLu(in_channels, out_channel)

    def forward(self, x):
        fusing_tensor, pass_through= get_fusing_tensor_and_pass_through(x)
        return {"z": self.model(torch.concat(fusing_tensor), axis=1), **pass_through}
