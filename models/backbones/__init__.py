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
    '''
    input: torch.randn(1,3,512,512)
    | [resnet18] | Size: [11,176,512] | Out: [torch.Size([1, 512])] | Type: [fc]
    | [resnet50] | Size: [23,508,032] | Out: [torch.Size([1, 2048])] | Type: [fc]
    | [alexnet] | Size: [61,100,840] | Out: [torch.Size([1, 256, 15, 15])] | Type: [features]
    | [vgg16] | Size: [138,357,544] | Out: [torch.Size([1, 512, 16, 16])] | Type: [features]
    | [squeezenet1_0] | Size: [1,248,424] | Out: [torch.Size([1, 512, 31, 31])] | Type: [features]
    | [densenet161] | Size: [28,681,000] | Out: [torch.Size([1, 2208, 16, 16])] | Type: [features]
    | [shufflenet_v2_x1_0] | Size: [1,253,604] | Out: [torch.Size([1, 1024])] | Type: [fc]
    | [mobilenet_v2] | Size: [3,504,872] | Out: [torch.Size([1, 1280, 16, 16])] | Type: [features]
    | [mobilenet_v3_large] | Size: [5,483,032] | Out: [torch.Size([1, 960, 16, 16])] | Type: [features]
    | [mobilenet_v3_small] | Size: [2,542,856] | Out: [torch.Size([1, 576, 16, 16])] | Type: [features]
    | [resnext50_32x4d] | Size: [22,979,904] | Out: [torch.Size([1, 2048])] | Type: [fc]
    | [wide_resnet50_2] | Size: [66,834,240] | Out: [torch.Size([1, 2048])] | Type: [fc]
    | [mnasnet1_0] | Size: [3,102,312] | Out: [torch.Size([1, 1280])] | Type: [classifier]
    | [efficientnet_b0] | Size: [5,288,548] | Out: [torch.Size([1, 1280, 16, 16])] | Type: [features]
    | [efficientnet_b1] | Size: [7,794,184] | Out: [torch.Size([1, 1280, 16, 16])] | Type: [features]
    | [efficientnet_b2] | Size: [9,109,994] | Out: [torch.Size([1, 1408, 16, 16])] | Type: [features]
    | [efficientnet_b3] | Size: [12,233,232] | Out: [torch.Size([1, 1536, 16, 16])] | Type: [features]
    | [efficientnet_b4] | Size: [19,341,616] | Out: [torch.Size([1, 1792, 16, 16])] | Type: [features]
    | [efficientnet_b5] | Size: [30,389,784] | Out: [torch.Size([1, 2048, 16, 16])] | Type: [features]
    | [efficientnet_b6] | Size: [43,040,704] | Out: [torch.Size([1, 2304, 16, 16])] | Type: [features]
    | [efficientnet_b7] | Size: [66,347,960] | Out: [torch.Size([1, 2560, 16, 16])] | Type: [features]
    | [regnet_y_400mf] | Size: [3,903,144] | Out: [torch.Size([1, 440])] | Type: [fc]
    | [regnet_y_800mf] | Size: [5,647,512] | Out: [torch.Size([1, 784])] | Type: [fc]
    | [regnet_y_1_6gf] | Size: [10,313,430] | Out: [torch.Size([1, 888])] | Type: [fc]
    | [regnet_y_3_2gf] | Size: [17,923,338] | Out: [torch.Size([1, 1512])] | Type: [fc]
    | [regnet_y_8gf] | Size: [37,364,472] | Out: [torch.Size([1, 2016])] | Type: [fc]
    | [regnet_y_16gf] | Size: [80,565,140] | Out: [torch.Size([1, 3024])] | Type: [fc]
    | [regnet_y_32gf] | Size: [141,333,770] | Out: [torch.Size([1, 3712])] | Type: [fc]
    | [regnet_y_128gf] | Size: [637,419,894] | Out: [torch.Size([1, 7392])] | Type: [fc]
    | [regnet_x_400mf] | Size: [5,094,976] | Out: [torch.Size([1, 400])] | Type: [fc]
    | [regnet_x_800mf] | Size: [6,586,656] | Out: [torch.Size([1, 672])] | Type: [fc]
    | [regnet_x_1_6gf] | Size: [8,277,136] | Out: [torch.Size([1, 912])] | Type: [fc]
    | [regnet_x_3_2gf] | Size: [14,287,552] | Out: [torch.Size([1, 1008])] | Type: [fc]
    | [regnet_x_8gf] | Size: [37,651,648] | Out: [torch.Size([1, 1920])] | Type: [fc]
    | [regnet_x_16gf] | Size: [52,229,536] | Out: [torch.Size([1, 2048])] | Type: [fc]
    | [regnet_x_32gf] | Size: [105,290,560] | Out: [torch.Size([1, 2520])] | Type: [fc]
    | [vit_b_16] | Size: [86,567,656] | Out: [Unknown] | Type: [unknown]
    | [vit_b_32] | Size: [88,224,232] | Out: [Unknown] | Type: [unknown]
    | [vit_l_16] | Size: [304,326,632] | Out: [Unknown] | Type: [unknown]
    | [vit_l_32] | Size: [306,535,400] | Out: [Unknown] | Type: [unknown]
    | [convnext_tiny] | Size: [28,589,128] | Out: [torch.Size([1, 768, 16, 16])] | Type: [features]
    | [convnext_small] | Size: [50,223,688] | Out: [torch.Size([1, 768, 16, 16])] | Type: [features]
    | [convnext_base] | Size: [88,591,464] | Out: [torch.Size([1, 1024, 16, 16])] | Type: [features]
    | [convnext_large] | Size: [197,767,336] | Out: [torch.Size([1, 1536, 16, 16])] | Type: [features]
    '''

    if setup.backbone == "resnet18":
        backbone = _to_resnet_feature_extract_backbone(
            torchvision.models.resnet18(pretrained=pretrained_backbone)
        )
        backbone.out_channels = 512
        backbone.out_dim = 1
    elif setup.backbone == "resnet50":
        backbone = _to_resnet_feature_extract_backbone(
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

    elif setup.backbone == "mobilenet_v3_large":
        backbone = _remove_last(
            torchvision.models.mobilenet_v3_large(pretrained=pretrained_backbone)
        )
        backbone.out_channels = 960
        backbone.out_dim = setup.image_size / 32

    elif setup.backbone == "vgg16":
        backbone = _remove_last(
            torchvision.models.vgg16(pretrained=pretrained_backbone)
        )
        backbone.out_channels = 512
        backbone.out_dim = setup.image_size / 32

    elif setup.backbone == "densenet161":
        backbone = _remove_last(
            torchvision.models.densenet161(pretrained=pretrained_backbone)
        )
        backbone.out_channels = 2208
        backbone.out_dim = setup.image_size / 32

    elif setup.backbone == "efficientnet_b0":
        backbone = _remove_last(
            torchvision.models.efficientnet_b0(pretrained=pretrained_backbone)
        )
        backbone.out_channels = 1280
        backbone.out_dim = setup.image_size / 32

    elif setup.backbone == "convnext_base":
        backbone = _remove_last(
            torchvision.models.convnext_base(pretrained=pretrained_backbone)
        )
        backbone.out_channels = 1024
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


def _to_resnet_feature_extract_backbone(resnet):
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
