from torch import nn
import torch

from models.components.general import Conv2dBNGELU
from models.components.deform import DeformConv2dBlock


def get_fusing_tensor_and_pass_through(x: dict[dict : torch.Tensor]):
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

def no_action_pass_through(x: dict[dict : torch.Tensor]):
    pass_through = {}
    for k, v in x.items():
        for vk, vv in v.items():
                pass_through[f"{k}_{vk}"] = vv
    return pass_through


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

    def forward(self, x: dict[dict : torch.Tensor]):
        # assert len(x.keys()) == 1, "should only have one element in no action fusor"
        pass_through = no_action_pass_through(x)
        return {**pass_through}


class ElementwiseSumFusor(GeneralFusor):
    def __init__(self, out_channel) -> None:
        super().__init__("fusor-elementwise_sum", out_channel=out_channel)

    def forward(self, x):
        self.x = x
        fusing_tensor, pass_through = get_fusing_tensor_and_pass_through(x)
        return {"z": sum(fusing_tensor), **pass_through}


class HadamardProductFusor(GeneralFusor):
    def __init__(self, out_channel) -> None:
        super().__init__("fusor-hadamard_product", out_channel)

    def forward(self, x):
        fusing_tensor, pass_through = get_fusing_tensor_and_pass_through(x)
        output = torch.ones(fusing_tensor[0].shape).cuda()
        for v in fusing_tensor:
            output *= v

        return {"z": output, **pass_through}

class ConcatenationFusor(GeneralFusor):
    def __init__(self, in_channels, out_channel) -> None:
        super().__init__("fusor_concatenation", out_channel)
        self.model = nn.Conv2d(in_channels, out_channel, kernel_size=3, padding=1)

    def forward(self, x):
        fusing_tensor, pass_through = get_fusing_tensor_and_pass_through(x)
        return {"z": self.model(torch.concat(fusing_tensor, axis=1)), **pass_through}
    
class ConcatenationDeformFusor(GeneralFusor):
    def __init__(self, in_channels, out_channel) -> None:
        super().__init__("fusor_concatenation", out_channel)
        self.model = DeformConv2dBlock(in_channels, out_channel, kernel_size=3, padding=1)

    def forward(self, x):
        fusing_tensor, pass_through = get_fusing_tensor_and_pass_through(x)
        return {"z": self.model(torch.concat(fusing_tensor, axis=1)), **pass_through}   
    
class ConcatenationWithBlockFusor(GeneralFusor):
    def __init__(self, in_channels, out_channel) -> None:
        super().__init__("fusor_concatenation", out_channel)
        self.model = Conv2dBNGELU(in_channels, out_channel)

    def forward(self, x):
        fusing_tensor, pass_through = get_fusing_tensor_and_pass_through(x)
        return {"z": self.model(torch.concat(fusing_tensor, axis=1)), **pass_through}

class ConcatenationWithTokenMixer(GeneralFusor):
    def __init__(self, in_channels, out_channel) -> None:
        super().__init__("fusor_concatenation_with_token_mixer", out_channel)
        self.model = nn.Conv2d(
            in_channels, out_channel, kernel_size=1, padding=0, stride=1
        )

    def forward(self, x):
        fusing_tensor, pass_through = get_fusing_tensor_and_pass_through(x)
        return {"z": self.model(torch.concat(fusing_tensor, axis=1)), **pass_through}

class ConcatenationWithBlockTokenMixer(GeneralFusor):
    def __init__(self, in_channels, out_channel, in_dim) -> None:
        super().__init__("fusor_concatenation_with_token_mixer", out_channel)
        self.model = nn.Sequential(
            *[
                nn.Conv2d(in_channels, out_channel, kernel_size=1, padding=0, stride=1),
                nn.LayerNorm([int(out_channel), int(in_dim), int(in_dim)]),
                nn.GELU(),
            ]
        )

    def forward(self, x):
        fusing_tensor, pass_through = get_fusing_tensor_and_pass_through(x)
        return {"z": self.model(torch.concat(fusing_tensor, axis=1)), **pass_through}
