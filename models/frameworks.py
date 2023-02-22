# Fuse frameworks that combine components here.

from collections import OrderedDict
from typing import Dict, List
from torch import nn
import torch

from data.helpers import map_target_to_device
from data.strs import SourceStrs, TaskStrs
from data.utils import chain_map
from models.components.general import map_labels
from models.components.rcnn import EyeImageRCNNTransform, EyeObjectDetectionRCNNTransform
from models.setup import ModelSetup


class ExtractFusePerform(nn.Module):
    """
    X is the input_dictionary
    """

    def __init__(
        self,
        setup: ModelSetup,
        feature_extractors: dict,
        fusor: nn.Module,
        task_performers: dict,
    ) -> None:  # expect the feature extractor be dictionary
        super().__init__()
        self.setup = setup
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
        if SourceStrs.XRAYS in self.feature_extractors:
            # then we have to do something to the image.
            has_transformed = False
            if TaskStrs.LESION_DETECTION in self.task_performers:
                x, targets = self.lesion_detetion_prepare(x, targets)
                has_transformed = True
            
            if TaskStrs.FIXATION_GENERATION in self.task_performers:
                x, targets = self.fixation_generation_prepare(x, targets)
                has_transformed = True

            if not has_transformed :
                image_mean = [0.485, 0.456, 0.406]
                image_std = [0.229, 0.224, 0.225]
                eye_transform = EyeImageRCNNTransform(
                    obj_det_task_name=None,
                    image_mean=image_mean,
                    image_std=image_std,
                    fixed_size=[self.setup.image_size,self.setup.image_size],
                )
                x, targets = self.image_transform(x, targets, eye_transform)
                
        return x, targets

    def image_transform(self, x, targets, eye_transform):
        batched_images, _ = eye_transform(
            [i["xrays"]["images"] for i in x], targets
        )

        for i, b_i in enumerate(batched_images):
            x[i]["xrays"]["images"] = b_i

        return x, targets


    def lesion_detetion_prepare(self, x, targets):

        # instead of putting these in the input x, we put it in the target of the object-detection.

        # need to check how to involve fixations here.

        batched_images, targets = self.task_performers[
            TaskStrs.LESION_DETECTION
        ].transform([i["xrays"]["images"] for i in x], targets)

        original_image_sizes = []
        for x_i in x:
            val = x_i["xrays"]["images"].shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        # assign image sizes
        for i, (b_i, o_s) in enumerate(zip(batched_images, original_image_sizes)):
            x[i]["xrays"]["images"] = b_i
            targets[i][TaskStrs.LESION_DETECTION]["original_image_sizes"] = o_s

        self.task_performers[TaskStrs.LESION_DETECTION].valid_bbox(
            [t[TaskStrs.LESION_DETECTION] for t in targets]
        )
        # targets[TaskStrs.LESION_DETECTION] = chain_map(targets[TaskStrs.LESION_DETECTION])
        # targets[TaskStrs.LESION_DETECTION]["original_image_sizes"] = original_image_sizes
        # targets[TaskStrs.LESION_DETECTION]["image_list_image_sizes"] = image_list.image_sizes
        # targets[TaskStrs.LESION_DETECTION][
        #     "image_list_tensors_shape"
        # ] = image_list.tensors.shape # (batch_size, 3, image_size, image_size)

        return x, targets
    
    def fixation_generation_prepare(self, x, targets):

        # instead of putting these in the input x, we put it in the target of the object-detection.
        # need to check how to involve fixations here.

        batched_images, targets = self.task_performers[
            TaskStrs.FIXATION_GENERATION
        ].transform([i["xrays"]["images"] for i in x], targets)

        # original_image_sizes = []
        # for x_i in x:
        #     val = x_i["xrays"]["images"].shape[-2:]
        #     assert len(val) == 2
        #     original_image_sizes.append((val[0], val[1]))

        # # assign image sizes
        for i, b_i in enumerate(batched_images):
            x[i]["xrays"]["images"] = b_i

        # self.task_performers[TaskStrs.LESION_DETECTION].valid_bbox(
        #     [t[TaskStrs.LESION_DETECTION] for t in targets]
        # )
        # targets[TaskStrs.LESION_DETECTION] = chain_map(targets[TaskStrs.LESION_DETECTION])
        # targets[TaskStrs.LESION_DETECTION]["original_image_sizes"] = original_image_sizes
        # targets[TaskStrs.LESION_DETECTION]["image_list_image_sizes"] = image_list.image_sizes
        # targets[TaskStrs.LESION_DETECTION][
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

