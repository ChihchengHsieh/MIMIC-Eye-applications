import torch.nn as nn

from data.constants import DEFAULT_REFLACX_LABEL_COLS
from .backbones import get_normal_backbone
from .setup import ModelSetup
from .components.feature_extractors import *
from .components.task_performers import *
from .components.fusors import *
from .frameworks import *


def create_model_from_setup(setup: ModelSetup):

    feature_extractors = nn.ModuleDict()

    if "image" in setup.sources:
        backbone = get_normal_backbone(setup)
        feature_extractors = nn.ModuleDict()
        image_extractor = ImageFeatureExtractor(backbone)
        feature_extractors.update({"image": image_extractor})

    fusor = NoActionFusor()

    task_performers = nn.ModuleDict()

    if "object-detection" in setup.tasks:
        obj_params = ObjectDetectionParameters(image_size=setup.image_size)
        obj_performer = ObjectDetectionPerformer(
            obj_params, image_extractor.backbone.out_channels, len(setup.label_cols) + 1,
        )
        task_performers.update({"object-detection": obj_performer})

    if "heatmap-generation" in setup.tasks:
        fix_params = HeatmapGeneratorParameters(
            input_channel=backbone.out_channels, decoder_channels=setup.decoder_channels
        )  # the output should be just one channel.
        fix_performer = HeatmapGenerator(params=fix_params,)
        task_performers.update({"heatmap-generation": fix_performer})

    model = ExtractFusePerform(
        feature_extractors=nn.ModuleDict(feature_extractors),
        fusor=fusor,
        task_performers=nn.ModuleDict(task_performers),
    )

    return model

