# Fuse frameworks that combine components here.

from collections import OrderedDict
from typing import Dict, List
from torch import nn
import torch

from data.helpers import map_target_to_device
from models.components.general import map_labels


class ExtractFusePerform(nn.Module):
    """
    X is the input_dictionary
    """

    def __init__(
        self, feature_extractors: dict, fusor: nn.Module, task_performers: dict,
    ) -> None:  # expect the feature extractor be dictionary
        super().__init__()
        self.feature_extractors = feature_extractors
        self.task_performers = task_performers
        self.fusor = fusor

    def forward(self, x: Dict, targets: List[Dict]):  # expect input to be a dictionary

        # the transform should be done here. (or even higher level)

        # self.__input_checking(x, targets)
        # x, targets = self.prepare(x, targets) # the prepare should be ran outside of this framework to optimise the memory usage.

        # extract feature maps # doesn't allow the feature extractors created but not used.

        feature_maps = OrderedDict(
            {k: self.feature_extractors[k](x) for k in self.feature_extractors.keys()}
        )

        fused = self.fusor(feature_maps)

        outputs = OrderedDict(
            {
                k: self.task_performers[k](x, fused, targets)
                for k in self.task_performers.keys()
            }
        )

        return outputs

    # def __input_checking(self, x: Dict, targets: List[Dict]):
    #     if self.training:
    #         assert targets is not None
    #         for target in targets:
    #             if "boxes" in target.keys():
    #                 boxes = target["boxes"]
    #                 if isinstance(boxes, torch.Tensor):
    #                     if len(boxes【.shape) != 2 or boxes.shape[-1] != 4:
    #                         raise ValueError(
    #                             "Expected target boxes to be a tensor"
    #                             "of shape [N, 4], got {:}.".format(boxes.shape)
    #                         )
    #                 else:
    #                     raise ValueError(
    #                         "Expected target boxes to be of type "
    #                         "Tensor, got {:}.".format(type(boxes))
    #                     )

    def prepare(self, x, targets):
        if "lesion-detection" in self.task_performers.keys():
            x, targets = self.lesion_detetion_prepare(x, targets)

        return x, targets

    def lesion_detetion_prepare(self, x, targets):
        original_image_sizes = []
        for img in x["xrays"]:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        x["xrays_original_image_sizes"] = original_image_sizes

        # need to check how to involve fixations here.
        x["xray_list"], tramsformed_targets = self.task_performers[
            "lesion-detection"
        ].transform(
            x["xrays"],
            map_labels(
                targets,
                self.task_performers["lesion-detection"].params.label_name_mapper,
            ),
        )

        reversed_targets = map_labels(
            targets=tramsformed_targets,
            mapper={
                v: k
                for k, v in self.task_performers[
                    "lesion-detection"
                ].params.label_name_mapper.items()
            },
        )

        for k, v in reversed_targets.items():
            targets[k] = v


        self.task_performers["lesion-detection"].valid_bbox(targets)

        # see if it's okay just keep the image list
        del x["xrays"]

        return x, targets

    def get_all_losses_keys(self,):

        loss_keys = []
        for k, p in self.task_performers.items():
            for l in p.loses:
                loss_keys.append(f"{k}_{l}")

        return loss_keys

    def get_all_params(self, dynamic_loss_weight=None):
        params = [p for p in self.parameters() if p.requires_grad]
        if dynamic_loss_weight:
            params += [p for p in dynamic_loss_weight.parameters() if p.requires_grad]
        return params

