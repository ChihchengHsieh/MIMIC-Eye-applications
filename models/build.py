import torch.nn as nn

from data.constants import DEFAULT_REFLACX_LABEL_COLS
from data.strs import FusionStrs, SourceStrs, TaskStrs
from .backbones import get_normal_backbone
from .setup import ModelSetup
from .components.feature_extractors import *
from .components.task_performers import *
from .components.fusors import *
from .frameworks import *


def create_model_from_setup(setup: ModelSetup):
    feature_extractors = nn.ModuleDict()

    feature_map_dim = None

    xrays_extractor_name = SourceStrs.XRAYS
    if xrays_extractor_name in setup.sources:
        backbone = get_normal_backbone(setup)
        xray_extractor = ImageFeatureExtractor(
            source_name=xrays_extractor_name, backbone=backbone,
        )
        feature_extractors.update({xrays_extractor_name: xray_extractor})
        feature_map_dim = [backbone.out_channels, backbone.out_dim, backbone.out_dim]

    clinical_extractor_name = SourceStrs.CLINICAL
    if clinical_extractor_name in setup.sources:
        clnical_extractor = TabularFeatureExtractor(
            source_name=clinical_extractor_name,
            all_cols=setup.clinical_cat + setup.clinical_num,
            categorical_col_maps=setup.categorical_col_maps,
            embedding_dim=setup.clinical_cat_emb_dim,
            out_dim=feature_map_dim[-1],
            conv_channels=setup.clinical_conv_channels,
            out_channels=setup.backbone_out_channels,
            upsample=setup.clinical_upsample,
        )
        feature_extractors.update({clinical_extractor_name: clnical_extractor})

    fixation_extractor_name = SourceStrs.FIXATIONS
    if fixation_extractor_name in setup.sources:
        backbone = get_normal_backbone(setup)
        fixations_extractor = ImageFeatureExtractor(
            source_name=fixation_extractor_name, backbone=backbone,
        )
        feature_extractors.update({fixation_extractor_name: fixations_extractor})
        feature_map_dim = [backbone.out_channels, backbone.out_dim, backbone.out_dim]

    if setup.fusor == FusionStrs.NO_ACTION:
        fusor = NoActionFusor(out_channel=setup.backbone_out_channels)
    elif setup.fusor == FusionStrs.ElEMENTWISE_SUM:
        fusor = ElementwiseSumFusor(out_channel=setup.backbone_out_channels)
    elif setup.fusor == FusionStrs.HADAMARD_PRODUCT:
        fusor = HadamardProductFusor(out_channel=setup.backbone_out_channels)
    elif setup.fusor == FusionStrs.CONCAT:
        fusor = ConcatenationFusor(
            in_channels=setup.backbone_out_channels * len(feature_extractors),
            out_channel=setup.backbone_out_channels,
        )
    else:
        ValueError(f"Unsupported fusion method: [{setup.fusor}]")

    task_performers = nn.ModuleDict()

    ########## clinical predictors 
    age_regression_task_name = TaskStrs.AGE_REGRESSION
    if age_regression_task_name in setup.tasks:
        age_regression_params = RegressionParameters(
            task_name=age_regression_task_name,
            input_channel=fusor.out_channel,
        )
        age_regression_clf = RegressionPerformer(params=age_regression_params)
        task_performers.update({age_regression_task_name: age_regression_clf})

    temperature_regression_task_name = TaskStrs.TEMPERATURE_REGRESSION
    if temperature_regression_task_name in setup.tasks:
        temperature_regression_params = RegressionParameters(
            task_name=temperature_regression_task_name,
            input_channel=fusor.out_channel,
        )
        temperature_regression_clf = RegressionPerformer(params=temperature_regression_params)
        task_performers.update({temperature_regression_task_name: temperature_regression_clf})

    heartrate_regression_task_name = TaskStrs.HEARTRATE_REGRESSION
    if heartrate_regression_task_name in setup.tasks:
        heartrate_regression_params = RegressionParameters(
            task_name=heartrate_regression_task_name,
            input_channel=fusor.out_channel,
        )
        heartrate_regression_clf = RegressionPerformer(params=heartrate_regression_params)
        task_performers.update({heartrate_regression_task_name: heartrate_regression_clf})

    resprate_regression_task_name = TaskStrs.RESPRATE_REGRESSION
    if resprate_regression_task_name in setup.tasks:
        resprate_regression_params = RegressionParameters(
            task_name=resprate_regression_task_name,
            input_channel=fusor.out_channel,
        )
        resprate_regression_clf = RegressionPerformer(params=resprate_regression_params)
        task_performers.update({resprate_regression_task_name: resprate_regression_clf})

    o2sat_regression_task_name = TaskStrs.O2SAT_REGRESSION
    if o2sat_regression_task_name in setup.tasks:
        o2sat_regression_params = RegressionParameters(
            task_name=o2sat_regression_task_name,
            input_channel=fusor.out_channel,
        )
        o2sat_regression_clf = RegressionPerformer(params=o2sat_regression_params)
        task_performers.update({o2sat_regression_task_name: o2sat_regression_clf})

    sbp_regression_task_name = TaskStrs.SBP_REGRESSION
    if sbp_regression_task_name in setup.tasks:
        sbp_regression_params = RegressionParameters(
            task_name=sbp_regression_task_name,
            input_channel=fusor.out_channel,
        )
        sbp_regression_clf = RegressionPerformer(params=sbp_regression_params)
        task_performers.update({sbp_regression_task_name: sbp_regression_clf})

    dbp_regression_task_name = TaskStrs.DBP_REGRESSION
    if dbp_regression_task_name in setup.tasks:
        dbp_regression_params = RegressionParameters(
            task_name=dbp_regression_task_name,
            input_channel=fusor.out_channel,
        )
        dbp_regression_clf = RegressionPerformer(params=dbp_regression_params)
        task_performers.update({dbp_regression_task_name: dbp_regression_clf})

    acuity_regression_task_name = TaskStrs.TEMPERATURE_REGRESSION
    if acuity_regression_task_name in setup.tasks:
        acuity_regression_params = RegressionParameters(
            task_name=acuity_regression_task_name,
            input_channel=fusor.out_channel,
        )
        acuity_regression_clf = RegressionPerformer(params=acuity_regression_params)
        task_performers.update({acuity_regression_task_name: acuity_regression_clf})
    
    gender_classification_task_name = TaskStrs.GENDER_CLASSIFICATION
    if gender_classification_task_name in setup.tasks:
        gender_clf_params = MultiBinaryClassificationParameters(
            task_name=gender_classification_task_name,
            input_channel=fusor.out_channel,
            num_classes=1,
        )
        gender_clf = MultiBinaryClassificationPerformer(params=gender_clf_params,)
        task_performers.update({gender_classification_task_name: gender_clf})

    # make the dataset, and add the tasks in model setup.

    lesion_detection_task_name = TaskStrs.LESION_DETECTION
    if lesion_detection_task_name in setup.tasks:
        lesion_params = ObjectDetectionParameters(
            task_name=lesion_detection_task_name,
            out_channels=xray_extractor.backbone.out_channels,
            num_classes=len(setup.lesion_label_cols) + 1,
            image_size=setup.image_size,
        )
        lesion_performer = ObjectDetectionPerformer(lesion_params)
        task_performers.update({lesion_params.task_name: lesion_performer})

    fixation_generation_task_name = TaskStrs.FIXATION_GENERATION
    if fixation_generation_task_name in setup.tasks:
        fix_params = HeatmapGenerationParameters(
            task_name=fixation_generation_task_name,
            input_channel=backbone.out_channels,
            decoder_channels=setup.decoder_channels,
            image_size= setup.image_size,
        )  # the output should be just one channel.
        fix_performer = HeatmapGenerationPerformer(params=fix_params,)
        task_performers.update({fix_params.task_name: fix_performer})

    chexpert_classification_task_name = TaskStrs.CHEXPERT_CLASSIFICATION
    if chexpert_classification_task_name in setup.tasks:
        chexpert_clf_params = MultiBinaryClassificationParameters(
            task_name=chexpert_classification_task_name,
            input_channel=fusor.out_channel,
            num_classes=len(setup.chexpert_label_cols),
        )
        chexpert_clf = MultiBinaryClassificationPerformer(params=chexpert_clf_params,)
        task_performers.update({chexpert_classification_task_name: chexpert_clf})

    negbio_classification_task_name = TaskStrs.NEGBIO_CLASSIFICATION
    if negbio_classification_task_name in setup.tasks:
        negbio_clf_params = MultiBinaryClassificationParameters(
            task_name=negbio_classification_task_name,
            input_channel=fusor.out_channel,
            num_classes=len(setup.negbio_label_cols),
        )
        negbio_clf = MultiBinaryClassificationPerformer(params=negbio_clf_params,)
        task_performers.update({negbio_classification_task_name: negbio_clf})

    model = ExtractFusePerform(
        setup=setup,
        feature_extractors=nn.ModuleDict(feature_extractors),
        fusor=fusor,
        task_performers=nn.ModuleDict(task_performers),
    )

    return model

