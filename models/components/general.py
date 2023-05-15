from torch import nn
import torch

def get_keys_from_value(d, val):
    return [k for k, v in d.items() if v == val]

def map_inputs(inputs, mapper):
    if mapper:
        print(inputs.keys())
        return {v: inputs[k] for k, v in mapper.items()}
    else:
        return inputs

def map_labels(targets, mapper):
    if mapper:
        return {v: targets[k] for k, v in mapper.items()}
    else:
        return targets


class ArgMax(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.argmax(x, dim=self.dim)


class Clamp(nn.Module):
    def __init__(self, min=0, max=1):
        super().__init__()
        self.min, self.max = min, max

    def forward(self, x):
        return torch.clamp(x, self.min, self.max)

class Activation(nn.Module):
    def __init__(self, name, **params):

        super().__init__()

        if name is None or name == "identity":
            self.activation = nn.Identity(**params)
        elif name == "sigmoid":
            self.activation = nn.Sigmoid()
        elif name == "softmax2d":
            self.activation = nn.Softmax(dim=1, **params)
        elif name == "softmax":
            self.activation = nn.Softmax(**params)
        elif name == "logsoftmax":
            self.activation = nn.LogSoftmax(**params)
        elif name == "tanh":
            self.activation = nn.Tanh()
        elif name == "argmax":
            self.activation = ArgMax(**params)
        elif name == "argmax2d":
            self.activation = ArgMax(dim=1, **params)
        elif name == "clamp":
            self.activation = Clamp(**params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError(
                f"Activation should be callable/sigmoid/softmax/logsoftmax/tanh/"
                f"argmax/argmax2d/clamp/None; got {name}"
            )
        
    def forward(self, x):
        return self.activation(x)
    
class Conv2dBNGELU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, padding=padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.model(x)


class Deconv2dBNReLu(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2,),
            nn.BatchNorm2d((out_channels),),
            nn.GELU(),
        )

    def forward(self, x):
        return self.model(x)