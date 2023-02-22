import torch.nn as nn
import torch.nn.functional as F

from models.components.general import Conv2dBNReLu


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

        decoder_layers = [DecoderBlock(channels[i], channels[i+1])
                          for i in range(len(channels)-1)]
        decoder_layers += [nn.Conv2d(in_channels=channels[-1],
                                     out_channels=1, kernel_size=3, stride=1, padding=1,)]

        self.model = nn.Sequential(*decoder_layers)

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
        x = self.model(x)
        return x
