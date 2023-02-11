import random
import torch

from typing import Callable, Dict, Tuple
from PIL import Image
from torchvision.transforms import functional as F


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image: Image.Image, target: Dict):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob: float):
        self.prob: float = prob

    def __call__(self, image: torch.Tensor, target: Dict) -> Tuple[torch.Tensor, Dict]:
        if random.random() < self.prob:
            _, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                raise StopIteration()
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class HorizontalFlipTransform(object):
    def __init__(self, prob: float):
        self.prob: float = prob

    def __call__(self, image: torch.Tensor, target: Dict) -> Tuple[torch.Tensor, Dict]:
        if random.random() < self.prob:
            _, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)

            if "fixations" in target:
                target['fixations'] = target["fixations"].flip(-1)

            if "keypoints" in target:
                raise StopIteration()
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints

            # if not fixation is None:
            #     fixation = fixation.flip(-1)

        return (image, target)


class ToTensor(object):
    def __call__(self, image: Image.Image, target: Dict) -> Tuple[torch.Tensor, Dict]:
        image = F.to_tensor(image)
        if "fixations" in target:
            target["fixations"] = F.to_tensor(target['fixations'])
        return image, target


def get_tensorise_h_flip_transform(
    train: bool,
) -> Callable[[Image.Image, Dict], Tuple[torch.Tensor, Dict]]:
    transforms = []
    transforms.append(ToTensor())
    if train:
        transforms.append(HorizontalFlipTransform(0.5))
    return Compose(transforms)
