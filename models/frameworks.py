# Fuse frameworks that combine components here.

from collections import OrderedDict
from typing import Dict, List
from torch import nn
import torch

from data.helpers import map_target_to_device
from data.utils import chain_map
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
            {
                k: self.feature_extractors[k](x)
                for k in self.feature_extractors.keys()
            }
        )

        # k is the task name or extractor name.

        fused = self.fusor(feature_maps)
        # fused is a dict: {"z": feature maps}

        outputs = OrderedDict(
            {
                k: self.task_performers[k](fused, targets)
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
        # first, do the mapping

        # for k in self.feature_extractors.keys():
        #     x[k] = chain_map(x[k])

        # for k in self.task_performers.keys():
        #     targets[k] = chain_map(targets[k])

        if "lesion-detection" in self.task_performers.keys():
            x, targets = self.lesion_detetion_prepare(x, targets)

        # for k in [k for k in self.task_performers.keys() if k != "lesion-detection"]:
        #     targets[k] = chain_map(targets[k])

        return x, targets

    def lesion_detetion_prepare(self, x, targets):

        # instead of putting these in the input x, we put it in the target of the object-detection.

        # need to check how to involve fixations here.

        batched_images, targets = self.task_performers["lesion-detection"].transform(
            [i["xrays"]["images"] for i in x], targets
        )

        original_image_sizes = []
        for x_i in x:
            val = x_i['xrays']['images'].shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

    
        # assign image sizes
        for i, (b_i, o_s) in enumerate(zip(batched_images, original_image_sizes)):
            x[i]['xrays']['images'] = b_i
            targets[i]['lesion-detection']['original_image_sizes'] = o_s


        self.task_performers["lesion-detection"].valid_bbox([t["lesion-detection"] for t in targets])

        # targets["lesion-detection"] = chain_map(targets["lesion-detection"])
        # targets["lesion-detection"]["original_image_sizes"] = original_image_sizes
        # targets["lesion-detection"]["image_list_image_sizes"] = image_list.image_sizes
        # targets["lesion-detection"][
        #     "image_list_tensors_shape"
        # ] = image_list.tensors.shape # (batch_size, 3, image_size, image_size)

        return x, targets

    def get_all_losses_keys(self,):

        loss_keys = []
        for k, p in self.task_performers.items():
            for l in p.loses:
                loss_keys.append(f"{k}_{p.name}_{l}")

        return loss_keys

    def get_all_params(self, dynamic_loss_weight=None):
        params = [p for p in self.parameters() if p.requires_grad]
        if dynamic_loss_weight:
            params += [p for p in dynamic_loss_weight.parameters() if p.requires_grad]
        return params

