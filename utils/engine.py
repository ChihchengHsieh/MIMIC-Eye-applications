import math, sys, time, torch, torchvision
from typing import Dict, List, Tuple
import torch.nn as nn

from models.setup import ModelSetup

from .coco_eval import CocoEvaluator

from . import detect_utils
from data.helpers import map_dict_elements_to_device

from .pred import pred_thrs_check
from torch.utils.data import DataLoader, Dataset
from torch.optim.optimizer import Optimizer

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


def train_one_epoch(
    setup: ModelSetup,
    model: nn.Module,
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

    if evaluate_on_run:
        coco_evaluator = CocoEvaluator(coco, iou_types, params_dict)

    lr_scheduler = None

    # if epoch == 1:
    #     print("start wariming up ")
    #     warmup_factor = 1.0 / 1000
    #     warmup_iters = min(1000, len(data_loader) - 1)

    #     lr_scheduler = torch.optim.lr_scheduler.LinearLR(
    #         optimizer, start_factor=warmup_factor, total_iters=warmup_iters
    #     )

    for data in metric_logger.log_every(data_loader, print_freq, header):
        # inputs, targets = data_loader.dataset.prepare_input_from_data(data, device)
        inputs, targets = data_loader.dataset.prepare_input_from_data(data)
        inputs, targets = model.prepare(inputs, targets)
        inputs = map_dict_elements_to_device(inputs, device)
        targets = [map_dict_elements_to_device(t, device) for t in targets]

        with torch.cuda.amp.autocast(enabled=False):
            outputs = model(inputs, targets=targets)
            # loss_dict = loss_multiplier(loss_dict,epoch)

            all_losses = {}
            for task in outputs.keys():
                all_losses.update(
                    {f"{task}_{k}": v for k, v in outputs[task]["losses"].items()}
                )

            if dynamic_loss_weight:
                losses = dynamic_loss_weight(all_losses)
            else:
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
            obj_dts = outputs["object-detection"]["outputs"]
            if not score_thres is None:
                obj_dts = [
                    pred_thrs_check(pred, data_loader.dataset, score_thres, device)
                    for pred in obj_dts
                ]

            obj_dts = [
                {k: v.detach().to(cpu_device) for k, v in t.items()} for t in obj_dts
            ]

            res = {
                target["image_id"].item(): dt for target, dt in zip(targets, obj_dts)
            }
            coco_evaluator.update(res)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    if evaluate_on_run:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        return coco_evaluator, metric_logger

    return metric_logger


@torch.inference_mode()
def evaluate(
    setup: ModelSetup,
    model: nn.Module,
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
    coco_evaluator = CocoEvaluator(coco, iou_types, params_dict)

    for data in metric_logger.log_every(data_loader, 100, header):
        inputs, targets = data_loader.dataset.prepare_input_from_data(data)
        inputs, targets = model.prepare(inputs, targets)
        inputs = map_dict_elements_to_device(inputs, device)
        targets = [map_dict_elements_to_device(t, device) for t in targets]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(inputs, targets=targets)
        # loss_dict = loss_multiplier(loss_dict)

        all_losses = {}
        for task in outputs.keys():
            all_losses.update(
                {f"{task}_{k}": v for k, v in outputs[task]["losses"].items()}
            )

        loss_dict_reduced = detect_utils.reduce_dict(all_losses)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        obj_dts = outputs["object-detection"]["outputs"]
        if not score_thres is None:
            obj_dts = outputs["object-detection"]["outputs"]
            obj_dts = [
                pred_thrs_check(pred, data_loader.dataset, score_thres, device)
                for pred in obj_dts
            ]

        obj_dts = [{k: v.to(cpu_device) for k, v in t.items()} for t in obj_dts]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): dt for target, dt in zip(targets, obj_dts)}

        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator, metric_logger

