from cProfile import label
import torch, torchvision

from typing import List
import torch.nn as nn
from ..setup import ModelSetup


class NoAction(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x):
        return x


def get_normal_backbone(
    setup: ModelSetup, pretrained_backbone=True,
):
    if setup.backbone == "resnet18":
        backbone = _to_feature_extract_backbone(
            torchvision.models.resnet18(pretrained=pretrained_backbone)
        )
        backbone.out_channels = 512
        backbone.out_dim = 1
    elif setup.backbone == "resnet50":
        backbone = _to_feature_extract_backbone(
            torchvision.models.resnet50(pretrained=pretrained_backbone)
        )
        backbone.out_channels = 2048
        backbone.out_dim = 1
    elif setup.backbone == "mobilenet_v3":
        backbone = _remove_last(
            torchvision.models.mobilenet_v3_small(pretrained=pretrained_backbone)
        )
        backbone.out_channels = 576
        backbone.out_dim = setup.image_size / 32
    else:
        raise Exception(f"Unsupported backbone {setup.backbone}")

    if setup.backbone_out_channels:
        out_dim = backbone.out_dim
        backbone = nn.Sequential(
            backbone,
            nn.Conv2d(backbone.out_channels, setup.backbone_out_channels, 3, 1, 1),
        )
        backbone.out_channels = setup.backbone_out_channels
        backbone.out_dim = out_dim

    if pretrained_backbone:
        print(f"Using pretrained backbone. {setup.backbone}")
    else:
        print("Not using pretrained backbone.")

    return backbone


def _to_feature_extract_backbone(resnet):
    return nn.Sequential(
        resnet.conv1,
        resnet.bn1,
        resnet.relu,
        resnet.maxpool,
        resnet.layer1,
        resnet.layer2,
        resnet.layer3,
        resnet.layer4,
    )


def _remove_last(model):
    if hasattr(model, "features"):
        return model.features

    elif hasattr(model, "fc"):
        model.fc = NoAction()
        if hasattr(model, "avgpool"):
            model.avgpool = NoAction()
        return model

    elif hasattr(model, "classifier"):
        model.classifier = NoAction()
        if hasattr(model, "avgpool"):
            model.avgpool = NoAction()
        return model
