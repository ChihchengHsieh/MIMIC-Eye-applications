import random
import torch

from typing import Callable, Dict, Tuple
from PIL import Image
from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image: Image.Image, target: Dict):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


# class RandomHorizontalFlip(object):
#     def __init__(self, prob: float):
#         self.prob: float = prob

#     def __call__(self, image: torch.Tensor, target: Dict) -> Tuple[torch.Tensor, Dict]:
#         if random.random() < self.prob:
#             _, width = image.shape[-2:]
#             image = image.flip(-1)
#             bbox = target["boxes"]
#             bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
#             target["boxes"] = bbox
#             # if "masks" in target:
#             #     target["masks"] = target["masks"].flip(-1)
#         return image, target


class HorizontalFlipTransform(object):
    def __init__(self, prob: float):
        self.prob: float = prob

    def __call__(self, image: torch.Tensor, target: Dict) -> Tuple[torch.Tensor, Dict]:
        if random.random() < self.prob:
            _, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target['lesion-detection']["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target['lesion-detection']["boxes"] = bbox

            # if "masks" in target:
            #     target["masks"] = target["masks"].flip(-1)

            if "fixation-generation" in target:
                target["fixation-generation"]["heatmaps"] =target["fixation-generation"]["heatmaps"].flip(-1)

        return (image, target)


class ToTensor(object):
    def __call__(self, image: Image.Image, target: Dict) -> Tuple[torch.Tensor, Dict]:
        image = F.to_tensor(image)
        if "fixation-generation" in target:
            target["fixation-generation"]["heatmaps"] = F.to_tensor(target["fixation-generation"]["heatmaps"])
        return image, target


def get_tensorise_h_flip_transform(
    train: bool,
) -> Callable[[Image.Image, Dict], Tuple[torch.Tensor, Dict]]:
    transforms = []
    transforms.append(ToTensor())
    if train:
        transforms.append(HorizontalFlipTransform(0.5))
    return Compose(transforms)
