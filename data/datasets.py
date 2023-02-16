import os
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

from models.setup import ModelSetup
from .constants import (
    DEFAULT_REFLACX_BOX_COORD_COLS,
    DEFAULT_REFLACX_BOX_FIX_COLS,
    REFLACX_ALL_LABEL_COLS,
    DEFAULT_REFLACX_LABEL_COLS,
    DEFAULT_REFLACX_PATH_COLS,
    DEFAULT_REFLACX_REPETITIVE_LABEL_MAP,
)
from .paths import (
    SPREADSHEET_FOLDER
)


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
        split_str: str = None,
        transforms: Callable[[Image.Image, Dict],
                             Tuple[torch.Tensor, Dict]] = None,
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
        with_bboxes: bool = True,
        with_fixations: bool = True,
        fiaxtions_mode='normal',  # [silent, reporting, all]

    ):
        # Data loading selections

        self.split_str: str = split_str

        # Image related
        self.transforms = transforms
        self.path_cols: List[str] = path_cols

        # Labels
        self.labels_cols: List[str] = labels_cols
        self.all_disease_cols: List[str] = all_disease_cols
        self.repetitive_label_map: Dict[str, List[str]] = repetitive_label_map
        self.box_fix_cols: List[str] = box_fix_cols
        self.box_coord_cols: List[str] = box_coord_cols
        self.dataset_mode: str = dataset_mode
        self.with_clinical = with_clincal
        self.with_fixations: bool = with_fixations
        self.fiaxtions_mode = fiaxtions_mode
        self.with_bboxes = with_bboxes

        self.df_path = "reflacx_clinical_eye.csv" if self.with_clinical else "reflacx_eye.csv"

        # load dataframe
        self.df: pd.DataFrame = pd.read_csv(
            os.path.join(spreadsheets_folder, self.df_path), index_col=0
        )

        # get the splited group that we desire.
        if not self.split_str is None:
            self.df: pd.DataFrame = self.df[self.df["split"] == self.split_str]

        # replace the path with local mimic folder path.
        for p_col in path_cols:
            if p_col in self.df.columns:
                if p_col == "bbox_paths":

                    def apply_bbox_paths_transform(input_paths_str: str) -> List[str]:
                        input_paths_list: List[str] = json.loads(
                            input_paths_str)
                        replaced_path_list: List[str] = [
                            p.replace("{XAMI_MIMIC_PATH}", MIMIC_EYE_PATH)
                            for p in input_paths_list
                        ]
                        return replaced_path_list

                    apply_fn: Callable[
                        [str], List[str]
                    ] = lambda x: apply_bbox_paths_transform(x)

                else:
                    apply_fn: Callable[[str], str] = lambda x: str(
                        Path(x.replace("{XAMI_MIMIC_PATH}", MIMIC_EYE_PATH))
                    )

                self.df[p_col] = self.df[p_col].apply(apply_fn)

        # preprocessing data.
        self.preprocess_label()

        self.chexpert_label_cols = [c for c in self.df.columns if c.endswith("_chexpert")]
        self.negbio_label_cols =  [c for c in self.df.columns if c.endswith("_negbio")]

        super().__init__()

    def preprocess_label(self,):
        self.df[self.all_disease_cols] = self.df[self.all_disease_cols].gt(0)

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

    def generate_bboxes_df(self, ellipse_df: pd.DataFrame,) -> pd.DataFrame:
        boxes_df = ellipse_df[self.box_fix_cols]

        # relabel repetitive columns.
        for k in self.repetitive_label_map.keys():
            boxes_df[k] = ellipse_df[
                [l for l in self.repetitive_label_map[k] if l in ellipse_df.columns]
            ].any(axis=1)

        # filtering out the diseases not in the label_cols
        # boxes_df = boxes_df[boxes_df[self.labels_cols].any(axis=1)]

        # ## get labels
        # for index in boxes_df[self.labels_cols].index:
        #     count_labels = 0
        #     for label in self.labels_cols:
        #         if boxes_df[self.labels_cols].at[index, label] and count_labels==0:
        #             count_labels +=1
        #             boxes_df.at[index, "label"] = label
        #         elif boxes_df[self.labels_cols].at[index, label] and count_labels>0:
        #             boxes_df = boxes_df.append({'xmin':boxes_df.at[index,'xmin'], 'ymin':boxes_df.at[index,'ymin'], 'xmax':boxes_df.at[index,'xmax'],
        #                                         'ymax':boxes_df.at[index,'ymax'], 'xmax':boxes_df.at[index,'xmax'],
        #                                         'certainty':boxes_df.at[index,'certainty'], 'label':label},
            # ignore_index=True)

        # filtering out the diseases not in the label_cols
        boxes_df = boxes_df[boxes_df[self.labels_cols].any(axis=1)]
        # get labels
        # instead of doing this version. we then get repeat the bounding boxes with more than one label
        # boxes_df["label"] = boxes_df[self.labels_cols].idxmax(axis=1)

        label_df = boxes_df.loc[:, DEFAULT_REFLACX_LABEL_COLS].reset_index(
            drop=True)

        self.label_df = label_df

        labels = [list(label_df.loc[i, label_df.any()].index)
                  for i in range(len(label_df))]

        boxes_df['label'] = labels

        new_df_list = []

        if len(boxes_df) > 0:
            for _, instance in boxes_df.iterrows():
                for l in instance["label"]:
                    new_df_list.append({
                        "xmin": instance['xmin'],
                        "ymin": instance['ymin'],
                        "xmax": instance['xmax'],
                        "ymax": instance['ymax'],
                        "label": l,
                    })

        return pd.DataFrame(new_df_list, columns=["xmin", "ymin", "xmax", "ymax", "label"])

        # boxes_df = boxes_df[self.box_fix_cols + ["label"]]
        # return boxes_df

    def __getitem__(
        self, idx: int
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict],
        Tuple[torch.Tensor, Dict],
    ]:
        # find the df
        data: pd.Series = self.df.iloc[idx]

        # convert images to rgb
        img: Image = Image.open(data["image_path"]).convert("RGB")

        if self.with_bboxes:

            # Get bounding boxes.
            bboxes_df = self.generate_bboxes_df(
                pd.read_csv(data["bbox_path"])
            )

            self.bboxes_df = bboxes_df

            bboxes = torch.tensor(
                np.array(bboxes_df[self.box_coord_cols], dtype=float)
            )  # x1, y1, x2, y2

            # Calculate area of boxes.
            area = (bboxes[:, 3] - bboxes[:, 1]) * \
                (bboxes[:, 2] - bboxes[:, 0])

            labels = torch.tensor(
                np.array(bboxes_df["label"].apply(
                    lambda l: self.disease_to_idx(l))).astype(int),
                dtype=torch.int64,
            )

            image_id = torch.tensor([idx])
            num_objs = bboxes.shape[0]

            # S suppose all instances are not crowd
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

            # prepare lesion-detection targets

            target = {
                "lesion-detection": {
                    "boxes": bboxes,
                    "labels": labels,
                    "image_id": image_id,
                    "area": area,
                    "iscrowd": iscrowd,
                    "dicom_id": data['dicom_id'],
                    "image_path": data['image_path']
                }
            }
            
            target['chexpert_classifications'] = {"classifications": torch.tensor(data[self.chexpert_label_cols])}
            target['negbio_classifications'] = {"classifications": torch.tensor(data[self.negbio_label_cols])}


        if self.with_fixations:
            # get fixations
            target["fixation_path"] = data["fixation_path"]
            fiaxtion_df = pd.read_csv(data["fixation_path"])
            if self.fiaxtions_mode != "normal":
                utterance_path = os.path.join(os.path.dirname(
                    data["fixation_path"]), "timestamps_transcription.csv")
                utterance_df = pd.read_csv(utterance_path)
                report_starting_time = utterance_df.iloc[0]['timestamp_start_word']
                if self.fiaxtions_mode == "reporting":
                    fiaxtion_df[fiaxtion_df['timestamp_start_fixation']
                                >= report_starting_time]
                elif self.fiaxtions_mode == "silent":
                    fiaxtion_df[fiaxtion_df['timestamp_start_fixation']
                                < report_starting_time]
                else:
                    raise ValueError("Not supported fiaxtions mode.")

            fix = get_heatmap(
                get_fixations_dict_from_fixation_df(fiaxtion_df),
                (data["image_size_x"], data["image_size_y"]),
            ).astype(np.float32)
            
            target['fixation-generation'] = {'fixations':  fix}
            # img_t, target, fix_t = self.transforms(img, target, fix)
            # target["fixations"] = fix_t
        # else:

        img_t, target = self.transforms(img, target)

        return {"image": img_t}, target

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

        images = [t["image"] for t in inputs]

        targets = [target_processing(t) for t in targets]

        return {"xrays": images}, targets

    def get_idxs_from_dicom_id(self, dicom_id: str) -> List[str]:
        return [
            self.df.index.get_loc(i)
            for i in self.df.index[self.df["dicom_id"].eq(dicom_id)]
        ]


    def get_image_path_from_dicom_id(self, dicom_id: str) -> List[str]:
        return self.df[self.df["dicom_id"] == dicom_id].iloc[0]["image_path"]
