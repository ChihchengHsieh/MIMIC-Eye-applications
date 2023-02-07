from typing import Dict
import torch

from data.datasets import ReflacxObjectDetectionDataset

def pred_thrs_check(
    pred: Dict, dataset, score_thres: Dict, device: str
)-> Dict:

    if len(pred["boxes"]) == 0:
        return pred

    select_idx = torch.tensor(
        [
            score > score_thres[dataset.label_idx_to_disease(label)]
            for label, score in zip(pred["labels"], pred["scores"])
        ]
    )
    for k in pred.keys():
        pred[k] = pred[k][select_idx.to(device)]

    return pred
