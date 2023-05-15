# The extractors here, take into different kinds of modalities, then ouput a 3-D tensor for task perfromer.
# Those 3d tensors from different extractors can then be fused together

from collections import OrderedDict
from typing import Dict
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch
from data.utils import chain_map

from models.components.general import Conv2dBNGELU, Deconv2dBNReLu, map_inputs


class InterpolateLayer(nn.Module):
    def __init__(
        self,
        scale_factor=2,
        mode="nearest",
    ):
        super().__init__()
        self.mode = mode
        self.scale_factor = scale_factor

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class RepeatExpander(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        out_dim,
    ) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels)
        )
        self.out_dim = out_dim

    def forward(self, x):
        output = self.model(x.squeeze())[:, :, None, None].repeat(
            1, 1, int(self.out_dim), int(self.out_dim)
        )
        return output


class SpatialisationBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        upsample="interpolate",  # [interpolate, deconv]
        last_activation=True,
    ):
        super().__init__()
        self.upsample = upsample

        if upsample == "deconv":
            self.upsample_layer = Deconv2dBNReLu(in_channels, out_channels)
            self.convs = nn.Sequential(
                Conv2dBNGELU(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                ),
            )
        elif upsample == "interpolate":
            self.upsample_layer = InterpolateLayer(scale_factor=2, mode="nearest")
            self.convs = nn.Sequential(
                Conv2dBNGELU(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                ),
            )
        else:
            raise ValueError(f"Not supported upsample method: {upsample}")

        if last_activation:
            self.convs.append(
                Conv2dBNGELU(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                ),
            )
        else:
            self.convs.append(
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                )
            )

    def forward(self, x):

        # if self.upsample == "deconv":
        #     x = self.deconv(x)
        # elif self.upsample == "interpolate":
        #     x = F.interpolate(x, scale_factor=2, mode="nearest")
        # else:
        #     raise ValueError(f"Not supported upsample method: {self.upsample}")
        x = self.upsample_layer(x)
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

    def __init__(self, source_name, backbone, fix_backbone=True) -> None:
        super().__init__("extractor-image")
        self.source_name = source_name
        self.backbone = backbone
        self.fix_backbone = fix_backbone
        self.still_have_this = "Yes"

    def forward(self, x):
        x = torch.stack(
            [x_i[self.source_name]["images"] for x_i in x],
            dim=0,
        )

        if self.fix_backbone:
            if len(self.backbone) == 2:
                x = self.backbone[0](x).detach()
                return {"output_f": self.backbone[1](x)}
            else:
                raise NotImplementedError("Not supported weights fixing method.")
        return {"output_f": self.backbone(x)}

    def fix_backbone_weights(self, x):
        self.fix_backbone = x
        print(f"backbone weights fix status: {self.fix_backbone}")


# class TabularFeatureExpander(GeneralFeatureExtractor):
#     def __init__(
#         self,
#         source_name: str,
#         all_cols: list,
#         categorical_col_maps: Dict,
#         embedding_dim: int,
#         out_dim: int,
#         out_channels,
#     ) -> None:

#         """
#         This model don't use the deconv, but just repeat the value.
#         """
#         super().__init__("extractor-tabular-expander")
#         self.source_name = source_name
#         self.has_cat = len(categorical_col_maps) > 0
#         self.has_num = len(all_cols) - len(categorical_col_maps) > 0

#         if self.has_cat:
#             self.embs = nn.ModuleDict(
#                 {
#                     k: nn.Embedding(v, embedding_dim)
#                     for k, v in categorical_col_maps.items()
#                 }
#             )

#         self.all_cols = all_cols
#         self.categorical_col_maps = categorical_col_maps
#         self.embedding_dim = embedding_dim

#         self.deconv_in_channels = (len(all_cols) - len(categorical_col_maps)) + (
#             len(categorical_col_maps) * embedding_dim
#         )

#         self.out_dim = out_dim

#         self.model = nn.Sequential(
#             *[
#                 nn.Linear(self.deconv_in_channels, out_channels),
#                 nn.LayerNorm(out_channels),
#                 nn.GELU(),
#             ],
#         )

#     def forward(
#         self,
#         x,
#     ):
#         """
#         {
#             'cat' : categorical tabular data,
#             'num' : numerical tabular data,
#         }
#         """

#         # raise StopIteration("Tabular Expander is in used.")

#         cat_data = [x_i[self.source_name]["cat"] for x_i in x]
#         num_data = [x_i[self.source_name]["num"] for x_i in x]

#         cat_data = chain_map(cat_data)
#         cat_data = {k: torch.stack(v, dim=0) for k, v in cat_data.items()}
#         num_data = torch.stack(num_data)
#         # x = x[self.source_name]

#         if self.has_cat:
#             emb_out = OrderedDict({k: self.embs[k](v) for k, v in cat_data.items()})
#             emb_out_cat = torch.concat(list(emb_out.values()), axis=1)

#             if self.has_num:
#                 tabular_input = torch.concat([num_data, emb_out_cat], dim=1)
#             else:
#                 tabular_input = emb_out_cat

#         else:
#             tabular_input = num_data

#         output = self.model(tabular_input)[:, :, None, None].repeat(
#             1, 1, int(self.out_dim), int(self.out_dim)
#         )

#         return {"output_f": output, "tabular_input": tabular_input}


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

        self.deconv_in_channels = (len(all_cols) - len(categorical_col_maps)) + (
            len(categorical_col_maps) * embedding_dim
        )

        self.out_dim = out_dim

        self.expand_times = np.log2(out_dim)

        if upsample == "repeat":
            self.spatialisations = RepeatExpander(
                out_dim=out_dim,
                out_channels=out_channels,
                in_channels=self.deconv_in_channels,
            )
        else:
            self.spatialisations = nn.Sequential(
                *(
                    [
                        SpatialisationBlock(
                            in_channels=self.deconv_in_channels,
                            out_channels=conv_channels,
                            upsample=upsample,
                            last_activation=True,
                        )
                    ]
                    + [
                        SpatialisationBlock(
                            in_channels=conv_channels,
                            out_channels=conv_channels,
                            upsample=upsample,
                            last_activation=True,
                        )
                        for _ in range(int(self.expand_times) - 2)
                    ]
                    + [
                        SpatialisationBlock(
                            in_channels=conv_channels,
                            out_channels=out_channels,
                            upsample=upsample,
                            last_activation=False,
                        )
                    ]
                )
            )

    def forward(self, x):
        """
        {
            'cat' : categorical tabular data,
            'num' : numerical tabular data,
        }
        """

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

        return {"output_f": output, "tabular_input": tabular_input}


class SequentialFeatureExtractor(GeneralFeatureExtractor):
    def __init__(self, source_name: str, encoder_type="", **kwargs) -> None:
        super().__init__("extractor-sequential")
        self.source_name = source_name
        if encoder_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
            transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
