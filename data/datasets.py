from collections import OrderedDict
import os
import random
from sklearn.preprocessing import LabelEncoder
import torch
import json

import pandas as pd
import numpy as np
import torch.utils.data as data
import pandas as pd
import numpy as np
import torch.utils.data as data
import sys
import torch.nn.functional as F
from typing import Callable, Dict, List, Tuple, Union
from pathlib import Path
from PIL import Image
from copy import deepcopy
from data.strs import SourceStrs, TaskStrs
from data.utils import chain_map

from models.setup import ModelSetup
from .constants import (
    DEFAULT_REFLACX_BOX_COORD_COLS,
    DEFAULT_REFLACX_BOX_FIX_COLS,
    REFLACX_ALL_LABEL_COLS,
    DEFAULT_REFLACX_LABEL_COLS,
    DEFAULT_REFLACX_PATH_COLS,
    DEFAULT_REFLACX_REPETITIVE_LABEL_MAP,
)
from .paths import SPREADSHEET_FOLDER


from .helpers import map_dict_to_device, map_target_to_device, target_processing
from .fixation import get_fixations_dict_from_fixation_df, get_heatmap

from torchvision.transforms import functional as tvF


def collate_fn(batch: Tuple) -> Tuple:
    return tuple(zip(*batch))


class ReflacxDataset(data.Dataset):
    """
    Class to load the preprocessed REFLACX master sheet. There `.csv` files are required to run this class.

    - `reflacx_for_eyetracking.csv'

    """

    def __init__(
        self,
        MIMIC_EYE_PATH: str,
        clinical_numerical_cols,
        clinical_categorical_cols,
        normalise_clinical_num=False,
        split_str: str = None,
        dataset_mode: str = "normal",
        labels_cols: List[str] = DEFAULT_REFLACX_LABEL_COLS,
        all_disease_cols: List[str] = REFLACX_ALL_LABEL_COLS,
        repetitive_label_map: Dict[
            str, List[str]
        ] = DEFAULT_REFLACX_REPETITIVE_LABEL_MAP,
        box_fix_cols: List[str] = DEFAULT_REFLACX_BOX_FIX_COLS,
        box_coord_cols: List[str] = DEFAULT_REFLACX_BOX_COORD_COLS,
        path_cols: List[str] = DEFAULT_REFLACX_PATH_COLS,
        spreadsheets_folder=SPREADSHEET_FOLDER,
        ### input & label fields ###
        with_xrays_input: bool = True,
        with_clincal_input: bool = True,
        with_bboxes_label: bool = True,
        with_fixations_label: bool = True,
        with_fixations_input: bool = True,
        fixations_mode_label="reporting",  # [silent, reporting, all]
        fixations_mode_input="reporting",
        with_chexpert_label: bool = True,
        with_negbio_label: bool = True,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        image_size=512,
        random_flip=True,
    ):
        # Data loading selections

        self.split_str: str = split_str
        self.random_flip = random_flip
        # image related params.
        self.image_size = image_size
        self.image_mean = image_mean
        self.image_std = image_std

        self.MIMIC_EYE_PATH = MIMIC_EYE_PATH
        self.path_cols = path_cols
        self.path_cols: List[str] = path_cols

        # Labels
        self.labels_cols: List[str] = labels_cols
        self.all_disease_cols: List[str] = all_disease_cols
        self.repetitive_label_map: Dict[str, List[str]] = repetitive_label_map
        self.dataset_mode: str = dataset_mode

        self.with_clinical = with_clincal_input
        if self.with_clinical:
            self.clinical_numerical_cols = clinical_numerical_cols
            self.normalise_clinical_num = normalise_clinical_num
            self.clinical_categorical_cols = clinical_categorical_cols

        self.with_xrays_input = with_xrays_input

        self.with_chexpert_label = with_chexpert_label
        self.with_negbi_label = with_negbio_label
        self.with_bboxes_label = with_bboxes_label
        if self.with_bboxes_label:
            self.box_fix_cols: List[str] = box_fix_cols
            self.box_coord_cols: List[str] = box_coord_cols

        self.with_fixations_label: bool = with_fixations_label
        if self.with_fixations_label:
            self.fiaxtions_mode_label = fixations_mode_label

        self.with_fixations_input = with_fixations_input
        if self.with_fixations_input:
            self.fiaxtions_mode_input = fixations_mode_input

        # deciding which to df load
        self.df_path = (
            "reflacx_clinical_eye.csv" if self.with_clinical else "reflacx_eye.csv"
        )
        self.df: pd.DataFrame = pd.read_csv(
            os.path.join(spreadsheets_folder, self.df_path), index_col=0
        )

        # get the splited group that we desire.
        if not self.split_str is None:
            self.df: pd.DataFrame = self.df[self.df["split"] == self.split_str]

        self.replace_paths()

        # preprocessing data.
        self.preprocess_label()

        self.chexpert_label_cols = [
            c for c in self.df.columns if c.endswith("_chexpert")
        ]
        self.negbio_label_cols = [c for c in self.df.columns if c.endswith("_negbio")]

        if self.with_clinical:
            self.preprocess_clinical_df()

        super().__init__()

    def preprocess_label(
        self,
    ):
        self.df[self.all_disease_cols] = self.df[self.all_disease_cols].gt(0)

    def replace_paths(
        self,
    ):
        # replace the path with local mimic folder path.
        for p_col in self.path_cols:
            if p_col in self.df.columns:
                if p_col == "bbox_paths":

                    def apply_bbox_paths_transform(input_paths_str: str) -> List[str]:
                        input_paths_list: List[str] = json.loads(input_paths_str)
                        replaced_path_list: List[str] = [
                            p.replace("{XAMI_MIMIC_PATH}", self.MIMIC_EYE_PATH)
                            for p in input_paths_list
                        ]
                        return replaced_path_list

                    apply_fn: Callable[
                        [str], List[str]
                    ] = lambda x: apply_bbox_paths_transform(x)

                else:
                    apply_fn: Callable[[str], str] = lambda x: str(
                        Path(x.replace("{XAMI_MIMIC_PATH}", self.MIMIC_EYE_PATH))
                    )

                self.df[p_col] = self.df[p_col].apply(apply_fn)

    def load_image_array(self, image_path: str) -> np.ndarray:
        return np.asarray(Image.open(image_path))

    def plot_image_from_array(self, image_array: np.ndarray):
        im = Image.fromarray(image_array)
        im.show()

    def negbio_chexpert_disease_to_idx(disease, label_cols):
        if not disease in label_cols:
            raise Exception("This disease is not the label.")

        return label_cols.index(disease)

    def negbio_chexpert_idx_to_disease(idx, label_cols):
        if idx >= len(label_cols):
            return f"exceed label range :{idx}"

        return label_cols[idx]

    def disease_to_idx(self, disease: str) -> int:
        if not disease in self.labels_cols:
            raise Exception("This disease is not the label.")

        if disease == "background":
            return 0

        return self.labels_cols.index(disease) + 1

    def label_idx_to_disease(self, idx: int) -> str:
        if idx == 0:
            return "background"

        if idx > len(self.labels_cols):
            return f"exceed label range :{idx}"

        return self.labels_cols[idx - 1]

    def __len__(self) -> int:
        return len(self.df)

    def generate_bboxes_df(
        self,
        ellipse_df: pd.DataFrame,
    ) -> pd.DataFrame:
        boxes_df = ellipse_df[self.box_fix_cols]

        # relabel repetitive columns.
        for k in self.repetitive_label_map.keys():
            boxes_df.loc[:, k] = ellipse_df[
                [l for l in self.repetitive_label_map[k] if l in ellipse_df.columns]
            ].any(axis=1)

        # filtering out the diseases not in the label_cols
        boxes_df = boxes_df[boxes_df[self.labels_cols].any(axis=1)]
        label_df = boxes_df.loc[:, DEFAULT_REFLACX_LABEL_COLS].reset_index(drop=True)

        labels = [
            list(label_df.loc[i, label_df.any()].index) for i in range(len(label_df))
        ]

        boxes_df["label"] = labels

        new_df_list = []

        if len(boxes_df) > 0:
            for _, instance in boxes_df.iterrows():
                for l in instance["label"]:
                    new_df_list.append(
                        {
                            "xmin": instance["xmin"],
                            "ymin": instance["ymin"],
                            "xmax": instance["xmax"],
                            "ymax": instance["ymax"],
                            "label": l,
                        }
                    )

        return pd.DataFrame(
            new_df_list, columns=["xmin", "ymin", "xmax", "ymax", "label"]
        )

    def normalize(self, image: torch.Tensor) -> torch.Tensor:
        if not image.is_floating_point():
            raise TypeError(
                f"Expected input images to be of floating type (in range [0, 1]), "
                f"but found type {image.dtype} instead"
            )
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]

    def resize_image(self, image: torch.Tensor, size):

        image = torch.nn.functional.interpolate(
            image[None],
            size=size,
            mode="bilinear",
            align_corners=False,
        )[0]

        return image

    def prepare_xray(self, xray):
        xray = self.normalize(xray)
        xray = self.resize_image(image=xray, size=[self.image_size, self.image_size])
        return xray

    def prepare_clinical(self, data):
        clinical_num = None
        if (
            not self.clinical_numerical_cols is None
            and len(self.clinical_numerical_cols) > 0
        ):
            if self.normalise_clinical_num:
                clinical_num = (
                    torch.tensor(
                        self.clinical_num_norm.transform(
                            np.array([data[self.clinical_numerical_cols]])
                        ),
                        dtype=float,
                    )
                    .float()
                    .squeeze()
                )
            else:
                clinical_num = torch.tensor(
                    np.array(data[self.clinical_numerical_cols], dtype=float)
                ).float()

        clinical_cat = None
        if (
            not self.clinical_categorical_cols is None
            and len(self.clinical_categorical_cols) > 0
        ):
            clinical_cat = {
                c: torch.tensor(np.array(data[c], dtype=int))
                for c in self.clinical_categorical_cols
            }

        return clinical_cat, clinical_num

    def get_fixation_image(self, data, mode):

        fiaxtion_df = pd.read_csv(data["fixation_path"])

        if mode != "normal":
            utterance_path = os.path.join(
                os.path.dirname(data["fixation_path"]),
                "timestamps_transcription.csv",
            )
            utterance_df = pd.read_csv(utterance_path)
            report_starting_time = utterance_df.iloc[0]["timestamp_start_word"]
            if mode == "reporting":
                fiaxtion_df = fiaxtion_df[
                    fiaxtion_df["timestamp_start_fixation"] >= report_starting_time
                ]
            elif mode == "silent":
                fiaxtion_df = fiaxtion_df[
                    fiaxtion_df["timestamp_start_fixation"] < report_starting_time
                ]
            else:
                raise ValueError("Not supported fiaxtions mode.")

        fix = get_heatmap(
            get_fixations_dict_from_fixation_df(fiaxtion_df),
            (data["image_size_x"], data["image_size_y"]),
        ).astype(np.float32)

        return fix

    def prepare_fixation(self, data, mode):
        fix = self.get_fixation_image(data, mode)
        fix = tvF.to_tensor(fix)
        fix = self.normalize(fix)
        fix = self.resize_image(image=fix, size=[self.image_size, self.image_size])
        return fix

    def resize_boxes(self,
        boxes: torch.Tensor, original_size: List[int], new_size: List[int]
    ) -> torch.Tensor:
        ratios = [
            torch.tensor(s, dtype=torch.float32, device=boxes.device)
            / torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
            for s, s_orig in zip(new_size, original_size)
        ]
        ratio_height, ratio_width = ratios
        xmin, ymin, xmax, ymax = boxes.unbind(1)

        xmin = xmin * ratio_width
        xmax = xmax * ratio_width
        ymin = ymin * ratio_height
        ymax = ymax * ratio_height
        return torch.stack((xmin, ymin, xmax, ymax), dim=1)

    def get_lesion_detection_labels(self, idx, data, original_size, new_size):
        bboxes_df = self.generate_bboxes_df(pd.read_csv(data["bbox_path"]))
        bboxes = np.array(bboxes_df[self.box_coord_cols], dtype=float)
          # x1, y1, x2, y2
        unsized_boxes = bboxes
        bboxes = torch.tensor(bboxes)
        bboxes = self.resize_boxes(boxes=bboxes, original_size=original_size, new_size=new_size)
        # resize the bb
        # Calculate area of boxes.
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        unsized_area = (unsized_boxes[:, 3] - unsized_boxes[:, 1]) * (unsized_boxes[:, 2] - unsized_boxes[:, 0])
        labels = torch.tensor(
            np.array(bboxes_df["label"].apply(lambda l: self.disease_to_idx(l))).astype(
                int
            ),
            dtype=torch.int64,
        )

        image_id = torch.tensor([idx])
        num_objs = bboxes.shape[0]
        # S suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        # prepare lesion-detection targets
        return {
            "boxes": bboxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
            "dicom_id": data["dicom_id"],
            "image_path": data["image_path"],
            "original_image_sizes": original_size,
            "unsized_boxes": unsized_boxes,
            "unsized_area": unsized_area,
        }

    def __getitem__(
        self, idx: int
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict],
        Tuple[torch.Tensor, Dict],
    ]:
        # find the df
        data: pd.Series = self.df.iloc[idx]

        # it's necesary to load the image, becasue it will be used to run the transform.
        xray: Image = Image.open(data["image_path"]).convert("RGB")
        xray = tvF.to_tensor(xray)
        original_image_size = xray.shape[-2:]

        flip = self.random_flip and random.random() < 0.5
        # contain this one into thhe

        """
        inputs
        """
        input_dict = OrderedDict({})

        if self.with_xrays_input:
            xray = self.prepare_xray(xray)
            if flip:
                xray = xray.flip(-1)
            input_dict.update({SourceStrs.XRAYS: {"images": xray}})

        if self.with_clinical:
            clinical_cat, clinical_num = self.prepare_clinical(data)
            input_dict.update(
                {SourceStrs.CLINICAL: {"cat": clinical_cat, "num": clinical_num}}
            )

        if self.with_fixations_input:
            fix = self.prepare_fixation(data, self.fiaxtions_mode_input)
            if flip:
                fix = fix.flip(-1)

            input_dict.update({SourceStrs.FIXATIONS: {"images": fix}})

        # do bboxes resizing later.
        # if self.obj_det_task_name in target_index:
        #     bbox = target_index[self.obj_det_task_name]["boxes"]
        #     bbox = resize_boxes(bbox, (h, w), image.shape[-2:])
        #     target_index[self.obj_det_task_name]["boxes"] = bbox

        """
        targets
        """

        target = OrderedDict({})

        if self.with_bboxes_label:
            lesion_target = self.get_lesion_detection_labels(
                idx=idx,
                data=data,
                original_size=original_image_size,
                new_size=[self.image_size, self.image_size],
            )

            if flip:
                width = self.image_size
                bbox = lesion_target["boxes"]
                bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
                lesion_target["boxes"] = bbox

            target.update({TaskStrs.LESION_DETECTION: lesion_target})

        if self.with_chexpert_label:
            target.update(
                {
                    TaskStrs.CHEXPERT_CLASSIFICATION: {
                        "classifications": torch.tensor(data[self.chexpert_label_cols])
                        == 1
                    }
                }
            )

        if self.with_negbi_label:
            target.update(
                {
                    TaskStrs.NEGBIO_CLASSIFICATION: {
                        "classifications": torch.tensor(data[self.negbio_label_cols])
                        == 1
                    }
                }
            )

        if self.with_fixations_label:
            fix = self.prepare_fixation(data, self.fiaxtions_mode_label)
            if flip:
                fix = fix.flip(-1)

            target.update({TaskStrs.FIXATION_GENERATION: {"heatmaps": fix}})

        # img_t, target = self.transforms(xray, target)

        # if self.with_xrays:
        # input_dict.update({SourceStrs.XRAYS: {"images": img_t}})

        ## we should perform the preprocessing in here instead of using that transformer in the task_perfromer.
        # record the original size of the image

        return input_dict, target

    def prepare_input_from_data(
        self,
        data: Union[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict],
            Tuple[torch.Tensor, Dict],
        ],
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict],
        Tuple[torch.Tensor, Dict],
    ]:
        inputs, targets = data

        inputs = list(inputs)
        targets = list(targets)

        return inputs, targets

    def get_idxs_from_dicom_id(self, dicom_id: str) -> List[str]:
        return [
            self.df.index.get_loc(i)
            for i in self.df.index[self.df["dicom_id"].eq(dicom_id)]
        ]

    def get_image_path_from_dicom_id(self, dicom_id: str) -> List[str]:
        return self.df[self.df["dicom_id"] == dicom_id].iloc[0]["image_path"]

    def preprocess_clinical_df(
        self,
    ):
        self.encoders_map: Dict[str, LabelEncoder] = {}

        # encode the categorical cols.
        for col in self.clinical_categorical_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.encoders_map[col] = le
