# Fuse frameworks that combine components here.

from torch import nn
import torch

class ExtractFusePerform(nn.Module):
    def __init__(self, feature_extractors: dict,  fusor: nn.Module, task_performers: dict,) -> None: # expect the feature extractor be dictionary
        super().__init__()
        self.feature_extractors = feature_extractors
        self.task_performers = task_performers
        self.fusor = fusor
    
    def forward(self, x: dict, targets): # expect input to be a dictionary

        # the transform should be done here. (or even higher level)

        self.__input_checking(x, targets)
        self.prepare(x, targets)


        # extract feature maps # doesn't allow the feature extractors created but not used.
        feature_maps = { k: self.feature_extractors[k](x) for k in self.feature_extractors.keys()}

        fused = self.fusor(feature_maps)

        outputs = { k: self.task_performers[k](x, fused, targets[k]) for k in self.task_performers.keys()}


        return outputs

    def __input_checking(self, x, targets):
        if self.training:
            assert targets is not None
            if "object-detection" in self.task_performers.keys():
                for target in targets["object-detection"]:
                    boxes = target["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                            raise ValueError(
                                "Expected target boxes to be a tensor"
                                "of shape [N, 4], got {:}.".format(boxes.shape)
                            )
                    else:
                        raise ValueError(
                            "Expected target boxes to be of type "
                            "Tensor, got {:}.".format(type(boxes))
                        )

    def prepare(self, x, targets):
        if "object-detection" in self.task_performers.keys():
            self.object_detetion_prepare(x, targets)

    def object_detetion_prepare(self, x, targets):
        original_image_sizes = []
        for img in x["image"]:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))
        
        x["original_image_sizes"] = original_image_sizes

        x['image_list'], targets['object-detection'] = self.task_performers['object-detection'].transform(x["image"], targets['object-detection'])
        self.task_performers['object-detection'].valid_bbox(targets['object-detection'])

        return x, targets





        

