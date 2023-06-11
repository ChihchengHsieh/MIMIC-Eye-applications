import os, pickle, torch
import torch.nn as nn
from typing import Dict, Tuple, Union
from models.dynamic_loss import DynamicWeightedLoss
from utils.train import get_dynamic_loss, get_optimiser

from .build import create_model_from_setup
from .train import TrainingInfo
from torch.optim.optimizer import Optimizer
from .setup import ModelSetup

class CLPretrainedLoadParams:
    def __init__(self, model_name, model_part, fix_weight) -> None:
        self.model_name = model_name
        self.model_part = model_part
        self.fix_wieght = fix_weight

def load_cl_pretrained(model, model_name, load_part, fix_weight):
    # set load_part = "feature_extractors.xrays" to load the pretrained image backbone.
    device = next(model.parameters()).device

    cp: Dict = torch.load(
        os.path.join("trained_models", model_name), map_location=device
    )

    load_part_dict = {k:v  for k,v  in cp['model_state_dict'].items() if k.startswith(load_part)}
    model.load_state_dict(load_part_dict, strict=False)

    if fix_weight:
        for n, param in model.named_parameters():
            if n.startswith(load_part):
                param.requires_grad = False
        load_part_params = list(filter(lambda p: p.requires_grad, [ param for n, param in model.named_parameters() if n.startswith(load_part)]))
        assert len(load_part_params)==0
    
    return model

def get_trained_model(
    model_select, device,
) -> Tuple[nn.Module, TrainingInfo, Union[Optimizer, None]]:

    with open(os.path.join("training_records", f"{model_select.value}.pkl"), "rb") as f:
        train_info: TrainingInfo = pickle.load(f)

    model = create_model_from_setup(train_info.model_setup)
    model.to(device)

    cp: Dict = torch.load(
        os.path.join("trained_models", model_select.value), map_location=device
    )

    model.load_state_dict(cp["model_state_dict"])

    model.to(device)

    dynamic_loss_weight = None
    if "dynamic_weight_state_dict" in cp:
        dynamic_loss_weight = get_dynamic_loss(loss_keys=model.get_all_losses_keys(), device=device)
        dynamic_loss_weight.to(device)
        dynamic_loss_weight.load_state_dict(cp["dynamic_weight_state_dict"])

    params = model.get_all_params(dynamic_loss_weight=dynamic_loss_weight)

    optim = None
    if "optimizer_state_dict" in cp:
        optim: torch.optim.optimizer.Optimizer = get_optimiser(
            params, train_info.model_setup
        )
        optim.load_state_dict(cp["optimizer_state_dict"])

    return model, train_info, optim, dynamic_loss_weight
    # return model, train_info, None, None


def get_current_epoch(trained_model) -> int:
    return int(
        (
            [substr for substr in trained_model.value.split("_") if "epoch" in substr][
                0
            ]
        ).replace("epoch", "")
    )


def get_model_name(trained_model, naming_map=None) -> str:
    return (
        naming_map[trained_model] if naming_map else str(trained_model).split(".")[-1]
    )


def get_model_label(trained_modelL, naming_map) -> str:
    return (
        get_model_name(trained_modelL, naming_map)
        + f" (epoch: {get_current_epoch(trained_modelL)})"
    )


def get_dataset_label(dataset, select_model):
    return dataset + f" (epoch: {get_current_epoch(select_model)})"
