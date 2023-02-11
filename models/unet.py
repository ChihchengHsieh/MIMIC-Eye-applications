import torch.nn as nn
import torch.nn.functional as F


class Conv2dBNReLu(nn.Module):
    def __init__(self, in_channels, out_channels,  kernel_size=3, padding=1) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()

        self.convs = nn.Sequential(
            Conv2dBNReLu(in_channels, out_channels, kernel_size=3, padding=1,),
            Conv2dBNReLu(out_channels, out_channels,
                         kernel_size=3, padding=1,),
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.convs(x)
        return x


class UNetDecoder(nn.Module):
    def __init__(
        self,
        input_channel,
        decoder_channels,
    ):
        super().__init__()

        channels = [input_channel] + decoder_channels

        # remove first skip with same spatial resolution
        # encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        # encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        # head_channels = encoder_channels[0]
        # in_channels = [head_channels] + list(decoder_channels[:-1])
        # skip_channels = list(encoder_channels[1:]) + [0]
        # out_channels = decoder_channels

        # combine decoder keyword arguments
        # kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)

        self.model = nn.Sequential(*[DecoderBlock(channels[i], channels[i+1])
                                     for i in range(len(channels)-1)])

    def forward(self, x):

        # # remove first skip with same spatial resolution
        # features = features[1:]
        # # reverse channels to start from head of encoder
        # features = features[::-1]

        # head = features[0]
        # skips = features[1:]

        # x = self.center(head)

        # for i, decoder_block in enumerate(self.blocks):
        #     skip = skips[i] if i < len(skips) else None
        #     x = decoder_block(x, skip)
        return self.model(x)
