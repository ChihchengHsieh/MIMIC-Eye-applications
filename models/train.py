from datetime import datetime
from .setup import ModelSetup
import numpy as np


class TrainingTimer(object):
    def __init__(self) -> None:
        self.init_t = datetime.now()
        self.start_t = None
        self.end_t = None
        self.last_epoch = None
        self.epoch_start_t = None

    def start_training(self,):
        self.start_t = datetime.now()

    def start_epoch(self,):
        self.epoch_start_t = datetime.now()

    def end_epoch(self, epoch):
        self.last_epoch = epoch

        finish_time = datetime.now()
        epoch_took = finish_time - self.epoch_start_t

        sec_already_took = (finish_time - self.start_t).seconds
        speed = sec_already_took / self.last_epoch

        return epoch_took, sec_already_took, speed

    def end_training(self,):
        self.end_t = datetime.now()

    def has_took_sec_from_init(self,):
        return (datetime.now() - self.init_t).seconds

    def has_took_sec(self,):
        return (datetime.now() - self.start_t).seconds


class TrainingInfo:
    def __init__(self, model_setup: ModelSetup):
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []

        # self.train_ap_ars = []
        # self.val_ap_ars = []
        # self.test_ap_ars = None

        self.last_val_evaluator = None
        self.last_train_evaluator = None
        self.test_evaluator = None

        self.best_val_performance = -np.inf
        self.best_performance_model_path = None

        self.final_model_path = None
        self.previous_ar_model = None
        self.previous_ap_model = None
        self.model_setup = model_setup
        self.timer = TrainingTimer()
        self.epoch = 0
        self.performance = {"train": [], "val": [], "test": []}
        self.all_tasks = None
        super(TrainingInfo).__init__()

    def __str__(self):
        title = "=" * 40 + f"For Training [{self.model_setup.name}]" + "=" * 40
        section_divider = len(title) * "="

        return (
            title + "\n" + str(self.model_setup) + "\n" + section_divider + "\n\n"
            f"Best performance model has been saved to: [{self.best_performance_model_path}]"
            + "\n"
            f"The final model has been saved to: [{self.final_model_path}]"
            + "\n\n"
            + section_divider
        )

