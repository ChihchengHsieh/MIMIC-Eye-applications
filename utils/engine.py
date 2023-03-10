import numpy as np
import math
import sys
import time
import torch
import torchvision
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    recall_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from typing import Dict, List, Tuple
import torch.nn as nn
from data.strs import TaskStrs
from data.utils import chain_map
from models.components.feature_extractors import ImageFeatureExtractor
from models.components.task_performers import (
    HeatmapGenerationPerformer,
    ImageClassificationPerformer,
    ObjectDetectionPerformer,
)
from models.frameworks import ExtractFusePerform

from models.setup import ModelSetup

from .coco_eval import CocoEvaluator

from . import detect_utils
from data.helpers import map_every_thing_to_device

from .pred import pred_thrs_check
from torch.utils.data import DataLoader, Dataset
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F

cpu_device = torch.device("cpu")


def get_iou_types(model: nn.Module, setup: ModelSetup) -> List[str]:
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if setup.use_mask:
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


def get_iou_score(gt, pred):
    intersection = np.logical_and(gt, pred)
    union = np.logical_or(gt, pred)
    if np.sum(union) == 0:
        return 1
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


class HeatmapGenerationEvaluator:
    def __init__(self) -> None:
        # self.preds = []
        # self.gts = []

        self.ious = []
        self.accuracies = []
        self.f1_scores = []
        self.precisions = []
        self.recalls = []
        self.mses = []
        self.maes = []

    def update(self, outputs, targets):

        for o, t in zip(outputs, targets):
            pred = F.sigmoid(o).to(cpu_device).detach().numpy()
            gt = t["heatmaps"].to(cpu_device).detach().numpy()

            self.ious.append(
                get_iou_score(gt=np.array(gt) > 0.5, pred=np.array(pred) > 0.5)
            )
            self.precisions.append(
                precision_score(
                    np.array(gt).reshape(-1) > 0.5, (np.array(pred) > 0.5).reshape(-1)
                )
            )
            self.accuracies.append(
                accuracy_score(
                    np.array(gt).reshape(-1) > 0.5, (np.array(pred) > 0.5).reshape(-1)
                )
            )
            self.f1_scores.append(
                f1_score(
                    np.array(gt).reshape(-1) > 0.5, (np.array(pred) > 0.5).reshape(-1)
                )
            )
            self.recalls.append(
                recall_score(
                    np.array(gt).reshape(-1) > 0.5, (np.array(pred) > 0.5).reshape(-1)
                )
            )

            self.gt = gt
            self.pred = pred
            
            ### Continues
            self.mses.append(
                mean_squared_error(np.array(gt).reshape(-1), np.array(pred).reshape(-1))
            )

            self.maes.append(
                mean_absolute_error(
                    np.array(gt).reshape(-1), np.array(pred).reshape(-1)
                )
            )

        # for o in outputs:
        #     self.preds.append(F.sigmoid(o).to(cpu_device).detach().numpy())

        # for t in targets:
        #     self.gts.append(t['heatmaps'].to(cpu_device).detach().numpy())

    def get_performance_dict(self,):
        return {
            "iou": np.mean(self.ious),
            "f1": np.mean(self.f1_scores),
            "precision": np.mean(self.precisions),
            "accuracy": np.mean(self.accuracies),
            "recall": np.mean(self.recalls),
            "mse": np.mean(self.mses),
            "mae": np.mean(self.maes),
        }


class ImageClassificationEvaluator:
    def __init__(self) -> None:
        self.preds = []
        self.gts = []

    def update(self, outputs, targets):
        for o in outputs:
            self.preds.append(F.sigmoid(o).to(cpu_device).detach().numpy())

        for t in targets:
            self.gts.append(t["classifications"].to(cpu_device).detach().numpy())

    def get_clf_score(self, clf_score, has_threshold=None):
        if has_threshold:
            return clf_score(
                np.array(self.gts).reshape(-1),
                (np.array(self.preds) > has_threshold).reshape(-1),
            )
        return clf_score(
            np.array(self.gts).reshape(-1), (np.array(self.preds)).reshape(-1)
        )

    def get_performance_dict(self,):
        return {
            "f1": self.get_clf_score(f1_score, has_threshold=0.5),
            "precision": self.get_clf_score(precision_score, has_threshold=0.5),
            "accuracy": self.get_clf_score(accuracy_score, has_threshold=0.5),
            "recall": self.get_clf_score(recall_score, has_threshold=0.5),
            "auc": self.get_clf_score(roc_auc_score, has_threshold=0.5),
        }


def train_one_epoch(
    setup: ModelSetup,
    model: ExtractFusePerform,
    optimizer: Optimizer,
    data_loader: DataLoader,
    device: str,
    epoch: int,
    print_freq: int,
    iou_types: List[str],
    coco: Dataset,
    score_thres: Dict[str, float] = None,
    evaluate_on_run=True,
    params_dict: Dict = None,
    dynamic_loss_weight=None,
) -> Tuple[CocoEvaluator, detect_utils.MetricLogger]:
    model.train()
    metric_logger = detect_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr", detect_utils.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = f"Epoch: [{epoch}]"

    evaluators = {}
    model.evaluators = evaluators

    if evaluate_on_run:
        for k, v in model.task_performers.items():
            if isinstance(v, ObjectDetectionPerformer):
                evaluators[k] = CocoEvaluator(coco, iou_types, params_dict)
            elif isinstance(v, HeatmapGenerationPerformer):
                evaluators[k] = HeatmapGenerationEvaluator()
            elif isinstance(v, ImageClassificationPerformer):
                evaluators[k] = ImageClassificationEvaluator()
            else:
                raise ValueError(f"Task-{k} doesn't have an evaluator.")

    lr_scheduler = None

    for e in model.feature_extractors.values():
        if isinstance(e, ImageFeatureExtractor):
            if epoch < setup.model_warmup_epochs:
                e.fix_backbone_weights(True)
            else:
                e.fix_backbone_weights(False)
            
    for data in metric_logger.log_every(data_loader, print_freq, header):
        inputs, targets = data_loader.dataset.prepare_input_from_data(data)
        # inputs, targets = model.prepare(inputs, targets)
        inputs = map_every_thing_to_device(inputs, device)
        targets = map_every_thing_to_device(targets, device)

        with torch.cuda.amp.autocast(enabled=False):
            outputs = model(inputs, targets=targets)
            # loss_dict = loss_multiplier(loss_dict,epoch)
            # print(outputs['lesion-detection']['outputs'])
            # print(outputs['lesion-detection']['losses'])
            # raise StopIteration()
            all_losses = {}
            for task in outputs.keys():
                all_losses.update(
                    {
                        f"{task}_{model.task_performers[task].name}_{k}": v
                        for k, v in outputs[task]["losses"].items()
                    }
                )

            if dynamic_loss_weight:
                losses = dynamic_loss_weight(all_losses)
            else:
                # enhance the rpn. (maybe lower this value later epoches if trainable.)
                if epoch < setup.loss_warmup_epochs:
                    all_losses['lesion-detection_performer-object_detection_loss_box_reg'] *= 0
                    all_losses['lesion-detection_performer-object_detection_loss_classifier'] *= 0

                # all_losses['lesion-detection_performer-object_detection_loss_box_reg'] *= 0.44
                # all_losses['lesion-detection_performer-object_detection_loss_classifier'] *= 0.47
                # all_losses['lesion-detection_performer-object_detection_loss_objectness'] *= 0.35
                # all_losses['lesion-detection_performer-object_detection_loss_rpn_box_reg'] *= 0.19

                # all_losses['lesion-detection_performer-object_detection_loss_objectness'] *= 1e+2
                losses = sum(loss for loss in all_losses.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = detect_utils.reduce_dict(all_losses)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()

        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if evaluate_on_run:
            for k in model.task_performers.keys():
                if isinstance(evaluators[k], CocoEvaluator):
                    obj_dts = outputs[k]["outputs"]
                    if not score_thres is None:
                        obj_dts = [
                            pred_thrs_check(
                                pred, data_loader.dataset, score_thres, device
                            )
                            for pred in obj_dts
                        ]

                    obj_dts = [
                        {k: v.detach().to(cpu_device) for k, v in t.items()}
                        for t in obj_dts
                    ]

                    # record the trained data from each, (only stored the in the cpu memory, not gpu)

                    res = {
                        img_id.item(): dt
                        for img_id, dt in zip(
                            [t[k]["image_id"] for t in targets], obj_dts
                        )
                    }
                    evaluators[k].update(res)
                else:
                    evaluators[k].update(outputs[k]["outputs"], [t[k] for t in targets])

            # raise StopIteration()

    # tasks to perform evaluation (fixation-generation, negbio-classification, chexpert-classification)
    # segmentation can be used with IoU
    # intersection = np.logical_and(target, prediction)
    # union = np.logical_or(target, prediction)
    # iou_score = np.sum(intersection) / np.sum(union)
    # or accuracy

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    if evaluate_on_run:
        for k, e in evaluators.items():
            if isinstance(e, CocoEvaluator):
                e.synchronize_between_processes()
                e.accumulate()
                e.summarize()

        return evaluators, metric_logger

    return metric_logger


@torch.inference_mode()
def evaluate(
    setup: ModelSetup,
    model: ExtractFusePerform,
    data_loader: DataLoader,
    device: str,
    coco: Dataset,
    iou_types: List[str],
    params_dict: Dict = None,
    score_thres: Dict[str, float] = None,
) -> Tuple[CocoEvaluator, detect_utils.MetricLogger]:

    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)

    model.eval()
    metric_logger = detect_utils.MetricLogger(delimiter="  ")
    header = "Evaluation:"
    # coco_evaluator = CocoEvaluator(coco, iou_types, params_dict)

    evaluators = {}

    for k, v in model.task_performers.items():
        if isinstance(v, ObjectDetectionPerformer):
            evaluators[k] = CocoEvaluator(coco, iou_types, params_dict)
        elif isinstance(v, HeatmapGenerationPerformer):
            evaluators[k] = HeatmapGenerationEvaluator()
        elif isinstance(v, ImageClassificationPerformer):
            evaluators[k] = ImageClassificationEvaluator()
        else:
            raise ValueError(f"Task-{k} doesn't have an evaluator.")

    for data in metric_logger.log_every(data_loader, 100, header):
        inputs, targets = data_loader.dataset.prepare_input_from_data(data)
        # inputs, targets = model.prepare(inputs, targets)
        # inputs = map_dict_elements_to_device(inputs, device)
        # targets = [map_dict_elements_to_device(t, device) for t in targets]
        # clinical cat has a different structure.
        # if "clinical" in inputs:
        #     inputs["clinical"]["cat"] = chain_map(inputs["clinical"]["cat"])
        #     inputs["clinical"]["cat"] = {
        #         k: torch.stack(v) for k, v in inputs["clinical"]["cat"].items()
        #     }

        # inputs = map_2l_nest_dict_to_device(inputs, device)
        # targets = map_2l_nest_dict_to_device(targets, device)

        inputs = map_every_thing_to_device(inputs, device)
        targets = map_every_thing_to_device(targets, device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        model_time = time.time()
        outputs = model(inputs, targets=targets)
        # loss_dict = loss_multiplier(loss_dict)

        all_losses = {}
        for task in outputs.keys():
            all_losses.update(
                {
                    f"{task}_{model.task_performers[task].name}_{k}": v
                    for k, v in outputs[task]["losses"].items()
                }
            )

        loss_dict_reduced = detect_utils.reduce_dict(all_losses)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        for k in model.task_performers.keys():
            if isinstance(evaluators[k], CocoEvaluator):
                obj_dts = outputs[k]["outputs"]
                if not score_thres is None:
                    obj_dts = [
                        pred_thrs_check(pred, data_loader.dataset, score_thres, device)
                        for pred in obj_dts
                    ]

                obj_dts = [
                    {k: v.detach().to(cpu_device) for k, v in t.items()}
                    for t in obj_dts
                ]

                # record the trained data from each, (only stored the in the cpu memory, not gpu)

                res = {
                    img_id.item(): dt
                    for img_id, dt in zip([t[k]["image_id"] for t in targets], obj_dts)
                }
                evaluators[k].update(res)
            else:
                evaluators[k].update(outputs[k]["outputs"], [t[k] for t in targets])

            model.evaluators = evaluators

        evaluator_time = time.time()
        evaluator_time = time.time() - evaluator_time

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    for k, e in evaluators.items():
        if isinstance(e, CocoEvaluator):
            e.synchronize_between_processes()
            e.accumulate()
            e.summarize()

    torch.set_num_threads(n_threads)

    return evaluators, metric_logger
