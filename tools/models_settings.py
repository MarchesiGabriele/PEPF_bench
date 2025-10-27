import pandas as pd
import numpy as np

class MoiraiModelSettings:
    def __init__(self, train_data: pd.DataFrame, settings: dict, covariate_col_names: list[str]):
        self.train_data = train_data
        self.settings = settings
        self.covariate_col_names = covariate_col_names

    def create_predictor(self, dataset_name: str, checkpoint_path: str):
        from tools.utils.forecast import Forecast
        from uni2ts.model.moirai import MoiraiModule, MoiraiForecast

        if self.settings["finetune"]:
            model = Forecast.load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                prediction_length=self.settings["prediction_length"],
                context_length=self.settings["context_length"],
                patch_size=self.settings["patch_size"],
                num_samples=self.settings["num_samples"],
                target_dim=1,
                feat_dynamic_real_dim= 0 if not self.settings["include_covariates"] else len(self.covariate_col_names),
                past_feat_dynamic_real_dim=0 if not self.settings["include_covariates"] else len(self.covariate_col_names),  
            )            
        else:
            model = Forecast(
                module= MoiraiModule.from_pretrained(f"Salesforce/{self.settings['model']}".replace("_", "-")), 
                prediction_length=self.settings["prediction_length"],
                context_length=self.settings["context_length"],
                patch_size=self.settings["patch_size"], 
                num_samples=self.settings["num_samples"],
                target_dim=1,
                feat_dynamic_real_dim= 0 if not self.settings["include_covariates"] else len(self.covariate_col_names),
                past_feat_dynamic_real_dim=0 if not self.settings["include_covariates"] else len(self.covariate_col_names),  
            )
        return model.create_predictor(batch_size=self.settings["batch_size"])
    
    def predict(self, predictor, train_data: pd.DataFrame, test_data: pd.DataFrame):
        from gluonts.dataset.common import ListDataset
        train_d = train_data.set_index("ds")
        train_d = train_d.sort_index()

        test_d = test_data.set_index("ds")
        test_d = test_d.sort_index()

        data_dict = {
            "start": train_d.index[0],
            "target": train_d["TARG__target"].values
        }
        
        # Add covariates if they are enabledshow 
        if self.settings["include_covariates"]:
            future_covariates = []
            past_covariates = []
            for col in self.covariate_col_names:
                if col in train_d.columns:
                    future_covariates.append(np.concatenate([train_d[col].values, test_d[col].values[:self.settings["prediction_length"]]]))
                    past_covariates.append(train_d[col].values)
            if future_covariates:
                data_dict["feat_dynamic_real"] = np.stack(future_covariates, axis=0)
            if past_covariates:
                data_dict["past_feat_dynamic_real"] = np.stack(past_covariates, axis=0)
        
        custom_data = ListDataset([data_dict], freq='h') 
        forecasts = predictor.predict(custom_data)
        forecast_vals = next(iter(forecasts))
        predictions = forecast_vals.mean
        quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        quantiles_predictions = np.array(forecast_vals.quantile(0.1)).reshape(-1, 1)
        for q in quantiles[1:]:  # Skip the first one as we already added it
            quantile_values = forecast_vals.quantile(q).reshape(-1, 1)
            quantiles_predictions = np.hstack([quantiles_predictions, quantile_values])

        return predictions, quantiles_predictions, forecast_vals.samples  # point, quantiles, samples
    
