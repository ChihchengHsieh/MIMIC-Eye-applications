from ast import Dict
from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class ModelSetup:

    # which mode of dataset is used during training.
    # [normal] - the images with same dicom_id will be seen as different instances.
    # [unified] - the images with same dicom_id will be seen as only one instance.

    # name of the model.
    name: str = None

    sources: List[str] = field(default_factory=lambda: ["chest x-ray", "clinical"])
    tasks: List[Dict] = field(
        default_factory=lambda: [
            "lesion-detection",
            "fixation-generation",
            "chexpert-classification",
            "negbio-classification",
        ]
    )

    fusor: str = "element-wise sum"

    # heatmap generation
    decoder_channels: List[int] = field(default_factory=lambda: [64, 32, 16, 8, 1])

    lesion_label_cols: List[str] = field(
        default_factory=lambda: [
            # "Fibrosis",
            # "Quality issue",
            # "Wide mediastinum",
            # "Fracture",
            # "Airway wall thickening",
            ######################
            # "Hiatal hernia",
            # "Acute fracture",
            # "Interstitial lung disease",
            # "Enlarged hilum",
            # "Abnormal mediastinal contour",
            # "High lung volume / emphysema",
            # "Pneumothorax",
            # "Lung nodule or mass",
            # "Groundglass opacity",
            ######################
            "Pulmonary edema",
            "Enlarged cardiac silhouette",
            "Consolidation",
            "Atelectasis",
            "Pleural abnormality",
            # "Support devices",
        ]
    )

    # setting up is heatmap mask will passed into the model.
    # use_heatmaps: bool = False
    # with_fixations: bool = False
    # with_pupil: bool = False
    # with_1st_third_fixations: bool = False
    # with_2nd_third_fixations: bool = False
    # with_rad_silence: bool = False
    # with_rad_speaking: bool = False

    # this will save the model with best validation performance across each epochs.
    save_early_stop_model: bool = True

    # Will the training process will be recorded. (The TrainInfo instance will be saved with the weights of model.)
    record_training_performance: bool = True

    # define the backbone used in the model.
    # If fixation is used, then both image and fixation backbones will use this architecture.
    backbone: str = "mobilenet_v3"  # [resnet18, resnet50, swin, mobilenet_v3]

    # optimiser for training the model, SGD is default for training CNN.
    optimiser: str = "sgd"  # [adamw, sgd]

    # learning rate.
    lr: float = 0.0005

    # L2 regulariser
    weight_decay: float = 0.05

    #####################
    # Pretrained setup.
    #####################

    # if the image backbone is pretrained.
    image_backbone_pretrained: bool = True
    # if the fixation backbone is pretrained.
    heatmap_backbone_pretrained: bool = False

    image_size: int = 256
    backbone_out_channels: int = 64
    batch_size: int = 16
    warmup_epochs: int = 0

    lr_scheduler: str = "ReduceLROnPlateau"  # [ReduceLROnPlateau, MultiStepLR]

    reduceLROnPlateau_factor: float = 0.1
    reduceLROnPlateau_patience: int = 3
    reduceLROnPlateau_full_stop: bool = False

    multiStepLR_milestones: List[int] = field(default_factory=lambda: [30, 50, 70, 90])
    multiStepLR_gamma: float = 0.1

    # For warming up the training, but found not useful in our case.
    # warmup_epoch: int = 10
    # warmup_factor: float = 1.0 / 1000;

    #######################
    # Model related params
    #######################

    representation_size: int = 64
    mask_hidden_layers: int = 64

    # the fpn is only implemented for ResNet and SwinTranformer.
    using_fpn: bool = False
    use_mask: bool = True

    fuse_conv_channels: int = 32

    box_head_dropout_rate: float = 0
    fuse_depth: int = 4

    fusion_strategy: str = "concat"  # ["add", "concat", "multiply"]
    fusion_residule: bool = False

    gt_in_train_till: int = 20

    measure_test: bool = True

    eval_freq: int = 10

    use_iobb: bool = True
    iou_thrs: np.array = field(default_factory=lambda: np.array([0.5]))

    fiaxtions_mode: str = "normal"  # [normal, reporting, silent]

    clinical_num: List[str] = field(
        default_factory=lambda: [
            "age",
            "temperature",
            "heartrate",
            "resprate",
            "o2sat",
            "sbp",
            "dbp",
            # "pain",
            "acuity",
        ]
    )

    clinical_cat: List[str] = field(default_factory=lambda: ["gender"])
    categorical_col_maps: dict[str, int] = field(
        default_factory=lambda: {"gender": 2,}
    )

    clinical_cat_emb_dim: int = 32
    clinical_conv_channels: int = 32
    clinical_upsample: str = "deconv"

    chexpert_label_cols: List[str] = field(
        default_factory=lambda: [
            "Atelectasis_chexpert",
            "Cardiomegaly_chexpert",
            "Consolidation_chexpert",
            "Edema_chexpert",
            "Enlarged Cardiomediastinum_chexpert",
            "Fracture_chexpert",
            "Lung Lesion_chexpert",
            "Lung Opacity_chexpert",
            "No Finding_chexpert",
            "Pleural Effusion_chexpert",
            "Pleural Other_chexpert",
            "Pneumonia_chexpert",
            "Pneumothorax_chexpert",
            "Support Devices_chexpert",
        ]
    )

    negbio_label_cols: List[str] = field(
        default_factory=lambda: [
            "Atelectasis_negbio",
            "Cardiomegaly_negbio",
            "Consolidation_negbio",
            "Edema_negbio",
            "Enlarged Cardiomediastinum_negbio",
            "Fracture_negbio",
            "Lung Lesion_negbio",
            "Lung Opacity_negbio",
            "No Finding_negbio",
            "Pleural Effusion_negbio",
            "Pleural Other_negbio",
            "Pneumonia_negbio",
            "Pneumothorax_negbio",
            "Support Devices_negbio",
        ]
    )

