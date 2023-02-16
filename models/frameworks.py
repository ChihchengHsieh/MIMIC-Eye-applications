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
            {k: self.feature_extractors[k](x[k]) for k in self.feature_extractors.keys()}
        )
      
      
        # k is the task name or extractor name.
         
        fused = self.fusor(feature_maps)
        #fused is a dict: {"z": feature maps}


        outputs = OrderedDict(
            {
                k: self.task_performers[k](fused, targets[k])
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
        
        # instead of putting these in the input x, we put it in the target of the object-detection.
        
        # need to check how to involve fixations here.
        image_list, targets = self.task_performers['object-detection'].transform(x["images"], targets)
        
        self.task_performers['object-detection'].valid_bbox(targets)

        x['images']  =  image_list.tensors

        targets["original_image_sizes"] = original_image_sizes
        targets['image_list_image_sizes'] = image_list.image_sizes
        targets['image_list_tensors_shape'] = image_list.tensors.shape

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

