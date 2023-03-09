import os
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
        transforms: Callable[[Image.Image, Dict], Tuple[torch.Tensor, Dict]] = None,
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
        with_xrays: bool = True,
        with_clincal: bool = True,
        with_bboxes: bool = True,
        with_fixations: bool = True,
        with_fixation_as_input: bool = True,
        with_chexpert: bool = True,
        with_negbio: bool = True,
        fiaxtions_mode="reporting",  # [silent, reporting, all]
        input_fiaxtions_mode = "reporting",
    ):
        # Data loading selections

        self.split_str: str = split_str

        self.MIMIC_EYE_PATH = MIMIC_EYE_PATH
        self.path_cols = path_cols

        # Image related
        self.transforms = transforms
        self.path_cols: List[str] = path_cols

        # Labels
        self.labels_cols: List[str] = labels_cols
        self.all_disease_cols: List[str] = all_disease_cols
        self.repetitive_label_map: Dict[str, List[str]] = repetitive_label_map
        self.dataset_mode: str = dataset_mode

        self.with_clinical = with_clincal
        if self.with_clinical:
            self.clinical_numerical_cols = clinical_numerical_cols
            self.normalise_clinical_num = normalise_clinical_num
            self.clinical_categorical_cols = clinical_categorical_cols

        self.with_xrays = with_xrays
        
        self.with_chexpert = with_chexpert
        self.with_negbio = with_negbio
        self.with_bboxes = with_bboxes
        if self.with_bboxes:
            self.box_fix_cols: List[str] = box_fix_cols
            self.box_coord_cols: List[str] = box_coord_cols

        self.with_fixations: bool = with_fixations
        if self.with_fixations:
            self.fiaxtions_mode = fiaxtions_mode

        self.with_fixations_as_input = with_fixation_as_input
        if self.with_fixations_as_input:
            self.input_fiaxtions_mode = input_fiaxtions_mode

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
            boxes_df[k] = ellipse_df[
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

    def __getitem__(
        self, idx: int
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict],
        Tuple[torch.Tensor, Dict],
    ]:
        # find the df
        data: pd.Series = self.df.iloc[idx]

        # convert images to rgb

        # it's necesary to load the image, becasue it will be used to run the transform.
        img: Image = Image.open(data["image_path"]).convert("RGB")

        target = {}

        if self.with_bboxes:

            # Get bounding boxes.
            bboxes_df = self.generate_bboxes_df(pd.read_csv(data["bbox_path"]))

            self.bboxes_df = bboxes_df

            bboxes = torch.tensor(
                np.array(bboxes_df[self.box_coord_cols], dtype=float)
            )  # x1, y1, x2, y2

            # Calculate area of boxes.
            area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])

            labels = torch.tensor(
                np.array(
                    bboxes_df["label"].apply(lambda l: self.disease_to_idx(l))
                ).astype(int),
                dtype=torch.int64,
            )

            image_id = torch.tensor([idx])
            num_objs = bboxes.shape[0]

            # S suppose all instances are not crowd
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

            # prepare lesion-detection targets

            target[TaskStrs.LESION_DETECTION] = {
                "boxes": bboxes,
                "labels": labels,
                "image_id": image_id,
                "area": area,
                "iscrowd": iscrowd,
                "dicom_id": data["dicom_id"],
                "image_path": data["image_path"],
            }

            # this has to be the same name as the task_performer.

        if self.with_chexpert:
            target[TaskStrs.CHEXPERT_CLASSIFICATION] = {
                "classifications": torch.tensor(data[self.chexpert_label_cols]) == 1
            }

        if self.with_negbio:
            target[TaskStrs.NEGBIO_CLASSIFICATION] = {
                "classifications": torch.tensor(data[self.negbio_label_cols]) == 1
            }

        if self.with_fixations:
            # get fixations
            # target["fixation_path"] = data["fixation_path"]
            fiaxtion_df = pd.read_csv(data["fixation_path"])

            if self.fiaxtions_mode != "normal":
                utterance_path = os.path.join(
                    os.path.dirname(data["fixation_path"]),
                    "timestamps_transcription.csv",
                )
                utterance_df = pd.read_csv(utterance_path)
                report_starting_time = utterance_df.iloc[0]["timestamp_start_word"]
                if self.fiaxtions_mode == "reporting":
                    fiaxtion_df = fiaxtion_df[
                        fiaxtion_df["timestamp_start_fixation"] >= report_starting_time
                    ]
                elif self.fiaxtions_mode == "silent":
                    fiaxtion_df = fiaxtion_df[
                        fiaxtion_df["timestamp_start_fixation"] < report_starting_time
                    ]
                else:
                    raise ValueError("Not supported fiaxtions mode.")

            fix = get_heatmap(
                get_fixations_dict_from_fixation_df(fiaxtion_df),
                (data["image_size_x"], data["image_size_y"]),
            ).astype(np.float32)

            target[TaskStrs.FIXATION_GENERATION] = {"heatmaps": fix}

        input_dict = {}

        if self.with_fixations_as_input:
            # get fixations
            # target["fixation_path"] = data["fixation_path"]
            fiaxtion_df = pd.read_csv(data["fixation_path"])

            if self.input_fiaxtions_mode != "normal":
                utterance_path = os.path.join(
                    os.path.dirname(data["fixation_path"]),
                    "timestamps_transcription.csv",
                )
                utterance_df = pd.read_csv(utterance_path)
                report_starting_time = utterance_df.iloc[0]["timestamp_start_word"]
                if self.input_fiaxtions_mode == "reporting":
                    fiaxtion_df = fiaxtion_df[
                        fiaxtion_df["timestamp_start_fixation"] >= report_starting_time
                    ]
                elif self.input_fiaxtions_mode == "silent":
                    fiaxtion_df = fiaxtion_df[
                        fiaxtion_df["timestamp_start_fixation"] < report_starting_time
                    ]
                else:
                    raise ValueError("Not supported fiaxtions mode.")

            fix = get_heatmap(
                get_fixations_dict_from_fixation_df(fiaxtion_df),
                (data["image_size_x"], data["image_size_y"]),
            ).astype(np.float32)

            fix_t, target = self.transforms(fix, target)

            input_dict.update(
                {SourceStrs.FIXATIONS: {"images": fix_t}}
            )
        

        if self.with_clinical:
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

            input_dict.update(
                {SourceStrs.CLINICAL: {"cat": clinical_cat, "num": clinical_num}}
            )

        img_t, target = self.transforms(img, target)

        # if self.with_xrays:
        input_dict.update({SourceStrs.XRAYS: {"images": img_t}})

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
