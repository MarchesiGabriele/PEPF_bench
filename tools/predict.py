import pandas as pd
import numpy as np
import tempfile
from tools.models_settings import MoiraiModelSettings
from tools.cli.train import start_finetune

class MyPredictor:
    def __init__(self, settings: dict, dataset_name: str, colnames: list[str]): 
        self.settings = settings
        self.predictor = None
        self.dataset_name = dataset_name
        self.colnames = colnames
        if 'TARG__target' in self.colnames: self.colnames.remove('TARG__target')


    def prepare_model(self, train_data: pd.DataFrame, checkpoint_path: str) -> None:
        match self.settings["model_name"]:
            case "moirai":
                moirai_settings = MoiraiModelSettings(train_data, self.settings, self.colnames)
                self.predictor = moirai_settings.create_predictor(self.dataset_name, checkpoint_path)
            case _:
                raise ValueError(f"Model {self.settings['model_name']} not supported")

    def predict(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> pd.DataFrame:
        predictions, quantiles_predictions, samples_predictions = None, None, None
        match self.settings["model_name"]:
            case "moirai":
                moirai_settings = MoiraiModelSettings(train_data, self.settings, self.colnames)
                predictions, quantiles_predictions, samples_predictions = moirai_settings.predict(self.predictor, train_data, test_data)
            case _:
                raise ValueError(f"Model {self.settings['model_name']} not supported")
        return predictions, quantiles_predictions, samples_predictions

    def finetune(self):
        start_finetune() 

