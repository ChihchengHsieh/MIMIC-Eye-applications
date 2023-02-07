from datetime import datetime
from .setup import ModelSetup
from enum import Enum


class TrainingInfo:
    def __init__(self, model_setup: ModelSetup):
        self.train_losses = []
        self.val_losses= []
        self.test_losses = []

        self.train_ap_ars = []
        self.val_ap_ars = []
        self.test_ap_ars = None

        self.last_val_evaluator  = None
        self.last_train_evaluator = None
        self.test_evaluator = None

        self.best_val_ar = -1
        self.best_val_ap = -1
        self.best_ar_val_model_path = None
        self.best_ap_val_model_path = None

        self.final_model_path = None
        self.previous_ar_model = None
        self.previous_ap_model = None
        self.model_setup = model_setup
        self.start_t = datetime.now()
        self.end_t = None
        self.epoch = 0
        super(TrainingInfo).__init__()

    def __str__(self):
        title = "=" * 40 + f"For Training [{self.model_setup.name}]" + "=" * 40
        section_divider = len(title) * "="

        return (
            title + "\n" + str(self.model_setup) + "\n" + section_divider + "\n\n"
            f"Best AP validation model has been saved to: [{self.best_ap_val_model_path}]"
            + "\n"
            f"Best AR validation model has been saved to: [{self.best_ar_val_model_path}]"
            + "\n"
            f"The final model has been saved to: [{self.final_model_path}]"
            + "\n\n"
            + section_divider
        )

