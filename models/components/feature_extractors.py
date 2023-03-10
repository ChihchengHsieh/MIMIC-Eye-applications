# The extractors here, take into different kinds of modalities, then ouput a 3-D tensor for task perfromer.
# Those 3d tensors from different extractors can then be fused together

from collections import OrderedDict
from typing import Dict
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch
from data.utils import chain_map

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
            raise ValueError(f"Not supported upsample method: {self.upsample}")

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

    def __init__(self, source_name, backbone) -> None:
        super().__init__("extractor-image")
        self.source_name = source_name
        self.backbone = backbone

    def forward(self, x):
        x = torch.stack([x_i[self.source_name]["images"] for x_i in x], dim=0,)
        return self.backbone(x)


class TabularFeatureExtractor(GeneralFeatureExtractor):
    """
        categorical_col_maps:{
            category_name: number of category
        }
    """

    def __init__(
        self,
        source_name: str,
        all_cols: list,
        categorical_col_maps: Dict,
        embedding_dim: int,
        out_dim: int,
        conv_channels,
        out_channels,
        upsample,
    ) -> None:
        super().__init__("extractor-tabular")
        self.source_name = source_name
        self.has_cat = len(categorical_col_maps) > 0
        self.has_num = len(all_cols) - len(categorical_col_maps) > 0

        if self.has_cat:
            self.embs = nn.ModuleDict(
                {
                    k: nn.Embedding(v, embedding_dim)
                    for k, v in categorical_col_maps.items()
                }
            )

        self.all_cols = all_cols
        self.categorical_col_maps = categorical_col_maps
        self.embedding_dim = embedding_dim

        deconv_in_channels = (len(all_cols) - len(categorical_col_maps)) + (
            len(categorical_col_maps) * embedding_dim
        )

        expand_times = np.log2(out_dim)

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
        """
        {
            'cat' : categorical tabular data,
            'num' : numerical tabular data,
        }
        """

        list[dict[str, torch.Tensor]]

        cat_data = [x_i[self.source_name]["cat"] for x_i in x]
        num_data = [x_i[self.source_name]["num"] for x_i in x]

        cat_data = chain_map(cat_data)
        cat_data = {k: torch.stack(v, dim=0) for k, v in cat_data.items()}
        num_data = torch.stack(num_data)
        # x = x[self.source_name]

        if self.has_cat:
            emb_out = OrderedDict({k: self.embs[k](v) for k, v in cat_data.items()})
            emb_out_cat = torch.concat(list(emb_out.values()), axis=1)

            if self.has_num:
                tabular_input = torch.concat([num_data, emb_out_cat], dim=1)
            else:
                tabular_input = emb_out_cat

        else:
            tabular_input = num_data

        output = self.spatialisations(tabular_input[:, :, None, None])

        return output


class SequentialFeatureExtractor(GeneralFeatureExtractor):
    def __init__(self, source_name: str, encoder_type="", **kwargs) -> None:
        super().__init__("extractor-sequential")
        self.source_name = source_name
        if encoder_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
            transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

