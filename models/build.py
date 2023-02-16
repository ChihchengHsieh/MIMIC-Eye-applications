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

    xray_input_mapper = {
        "xrays": "images",
        "xrays_original_image_sizes": "original_image_sizes",
        "xray_list": "image_list",
    }

    if "chest x-ray" in setup.sources:
        backbone = get_normal_backbone(setup)
        image_extractor = ImageFeatureExtractor(
            input_name_mapper=xray_input_mapper, backbone=backbone,
        )
        feature_extractors.update({"chest x-ray": image_extractor})

    if "clinical" in setup.sources:
        TabularFeatureExtractor(
            input_name_mapper={
                "clinical_cat": "tabular_cat",
                "clinical_num": "tabular_num",
            },
            all_cols=setup.clinical_cat + setup.clinical_num,
            categorical_col_maps=setup.categorical_col_maps,
            embedding_dim=setup.clinical_cat_emb_dim,
            image_size=setup.image_size,
            conv_channels=setup.clinical_conv_channels,
            out_channels=setup.backbone_out_channels,
            upsample=setup.clinical_upsample,
        )

    if setup.fusor == "no-action":
        fusor = NoActionFusor(out_channel=setup.backbone_out_channels)
    elif setup.fusor == "element-wise sum":
        fusor = ElementwiseSumFusor(out_channel=setup.backbone_out_channels)
    elif setup.fusor == "hadamard product":
        fusor = HadamardProductFusor(out_channel=setup.backbone_out_channels)
    else:
        fusor = ConcatenationFusor(
            in_channels=setup.backbone_out_channels * len(feature_extractors),
            out_channel=setup.backbone_out_channels,
        )

    task_performers = nn.ModuleDict()

    lesion_detection_task_name = "lesion-detection"
    if lesion_detection_task_name in setup.tasks:
        lesion_params = ObjectDetectionParameters(
            task_name=lesion_detection_task_name,
            input_name_mapper=xray_input_mapper,
            label_name_mapper={
                "lesion_boxes": "boxes",
                "lesion_labels": "labels",
                "lesion_image_id": "image_id",
                "lesion_iscrowd": "iscrowd",
            },
            out_channels=image_extractor.backbone.out_channels,
            num_classes=len(setup.lesion_label_cols) + 1,
            image_size=setup.image_size,
        )
        lesion_performer = ObjectDetectionPerformer(lesion_params)
        task_performers.update({lesion_params.task_name: lesion_performer})

    fixation_generation_task_name = "fixation-generation"
    if fixation_generation_task_name in setup.tasks:
        fix_params = HeatmapGenerationParameters(
            task_name=fixation_generation_task_name,
            label_name_mapper={"fixations": "heatmaps"},
            input_channel=backbone.out_channels,
            decoder_channels=setup.decoder_channels,
        )  # the output should be just one channel.
        fix_performer = HeatmapGenerationPerformer(params=fix_params,)
        task_performers.update({fix_params.task_name: fix_performer})

    chexpert_classification_task_name = "chexpert-classification"
    if chexpert_classification_task_name in setup.tasks:
        chexpert_clf_params = ImageClassificationParameters(
            task_name=chexpert_classification_task_name,
            label_name_mapper={"chexpert_classifications": "classifications"},
            input_channel=fusor.out_channel,
            num_classes=len(setup.chexpert_label_cols),
        )
        chexpert_clf = ImageClassificationPerformer(params=chexpert_clf_params,)
        task_performers.update({chexpert_classification_task_name: chexpert_clf})

    negbio_classification_task_name = "negbio-classification"
    if negbio_classification_task_name in setup.tasks:
        negbio_clf_params = ImageClassificationParameters(
            task_name=negbio_classification_task_name,
            label_name_mapper={"negbio_classifications": "classifications"},
            input_channel=fusor.out_channel,
            num_classes=len(setup.negbio_label_cols),
        )
        negbio_clf = ImageClassificationPerformer(params=negbio_clf_params,)
        task_performers.update({negbio_classification_task_name: negbio_clf})

    model = ExtractFusePerform(
        feature_extractors=nn.ModuleDict(feature_extractors),
        fusor=fusor,
        task_performers=nn.ModuleDict(task_performers),
    )

    return model

