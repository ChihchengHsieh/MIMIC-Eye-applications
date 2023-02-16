# The extractors here, take into different kinds of modalities, then ouput a 3-D tensor for task perfromer.
# Those 3d tensors from different extractors can then be fused together

from collections import OrderedDict
from typing import Dict
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch

from models.components.general import Conv2dBNReLu, Deconv2dBNReLu, map_inputs





class SpatialisationBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        upsample="interpolate",  # [interpolate, deconv]
    ):
        super().__init__()


        self.upsample = upsample

        if upsample == "deconv":
            self.deconv = Deconv2dBNReLu(in_channels, out_channels)
            self.convs = nn.Sequential(
                Conv2dBNReLu(out_channels, out_channels, kernel_size=3, padding=1,),
                Conv2dBNReLu(out_channels, out_channels, kernel_size=3, padding=1,),
            )
        elif upsample == "interpolate":
            self.convs = nn.Sequential(
                Conv2dBNReLu(in_channels, out_channels, kernel_size=3, padding=1,),
                Conv2dBNReLu(out_channels, out_channels, kernel_size=3, padding=1,),
            )
        else:
            raise ValueError(f"Not supported upsample method: {upsample}")

    def forward(self, x):

        if self.upsample == "deconv":
            x = self.deconv(x)
        elif self.upsample == "interpolate":
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        else:
            raise ValueError(f"Not supported upsample method: {upsample}")

        x = self.convs(x)

        return x


class GeneralFeatureExtractor(nn.Module):
    def __init__(self, name) -> None:
        self.name = name
        super().__init__()


class ImageFeatureExtractor(GeneralFeatureExtractor):
    """
    Extracting features maps from a given image, x.
    """

    def __init__(self, input_name_mapper, backbone) -> None:
        super().__init__("extractor-image")
        self.input_name_mapper = input_name_mapper
        self.backbone = backbone

    def forward(self, x):
        '''
        {
            image_list:
        }
        '''


        x = map_inputs(inputs=x, mapper=self.input_name_mapper)
        return self.backbone(x["image_list"].tensors)

class TabularFeatureExtractor(GeneralFeatureExtractor):
    """
        categorical_col_maps:{
            category_name: number of category
        }
    """

    def __init__(
        self,
        input_name_mapper: Dict,
        all_cols: list,
        categorical_col_maps: Dict,
        embedding_dim: int,
        image_size: int,
        conv_channels,
        out_channels,
        upsample,
    ) -> None:
        super().__init__("extractor-tabular")

        self.input_name_mapper = input_name_mapper

        self.embs = nn.ModuleDict(
            {k: nn.Embedding(v, embedding_dim) for k, v in categorical_col_maps.items()}
        )

        deconv_in_channels = (len(all_cols) - len(categorical_col_maps)) + (
            len(categorical_col_maps) * embedding_dim
        )

        expand_times = np.log2(image_size)

        self.spatialisations = nn.Sequential(
            *(
                [SpatialisationBlock(deconv_in_channels, conv_channels, upsample)]
                + [
                    SpatialisationBlock(conv_channels, conv_channels, upsample)
                    for _ in range(int(expand_times) - 2)
                ]
                + [SpatialisationBlock(conv_channels, out_channels, upsample)]
            )
        )

    def forward(self, x):
        '''
        {
            'tabular_cat' : categorical tabular data,
            'tabular_num' : numerical tabular data,
        }
        '''


        x = map_inputs(inputs=x, mapper=self.input_name_mapper)

        emb_out = OrderedDict({k: self.embs(v) for k, v in x["tabular_cat"]})

        emb_out_cat = torch.concat(list(emb_out.values()), axis=1)

        tabular_input = torch.concat(x["tabular_num"], emb_out_cat)

        output = self.spatialisations(tabular_input[:, :, None, None])

        return output


class SequentialFeatureExtractor(GeneralFeatureExtractor):
    def __init__(self,) -> None:
        super().__init__("extractor-sequential")

    def forward(self, x):
        pass