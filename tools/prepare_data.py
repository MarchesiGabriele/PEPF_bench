import pandas as pd
import os

from collections.abc import Generator
from typing import Any
import datasets
from datasets import Features, Sequence, Value

class PrepareData:
    def __init__(self, data_path: str, col_names: list, settings: dict, dataset_name: str):
        self.data_path = data_path
        self.col_names = col_names
        self.settings = settings
        self.dataset_name = dataset_name

    # load data for single dataset and compose full dataframe
    def load_data(self, include_covariates: bool = True, is_de: bool = False) -> pd.DataFrame:
        print(f"Loading data from {self.data_path}")
        df = pd.read_pickle(self.data_path)

        df_full = pd.DataFrame()
        for date in df.keys():
            df_day = df[date]
            df_day = df_day[self.col_names]
            df_full = pd.concat([df_full, df_day])
        df_full["unique_id"] = "0" # add a unique identifier for each time series 
        df_full["ds"] = df_full.index 
        df_full['ds'] = pd.to_datetime(df_full['ds'])
        df_full = df_full.reset_index(drop=True)

        if not include_covariates:
            df_full = df_full[['ds', 'TARG__target', 'unique_id']] # keep only the target 

        return df_full
    
    # prepare dataframe for single model
    def prepare_data(self, df_full: pd.DataFrame):
        model_name = self.settings["model_name"]
        recalib_window_size = self.settings["recalib_window_size"]
        print(f"Preparing data for {model_name} model...")
        # If start_train is not specified, use the first available date in the dataset
        if "start_train" in self.settings and self.settings["start_train"] is not None:
            first_train_day_idx = df_full[df_full['ds'] == pd.Timestamp(self.settings["start_train"])].index
        else:
            first_train_day_idx = pd.Index([0])  # Start from the first row of the dataset
        first_test_day_idx = df_full[df_full['ds'] == (pd.Timestamp(self.settings["start_test"]) - pd.Timedelta(days=recalib_window_size))].index
        last_test_day_idx = df_full[df_full['ds'] == pd.Timestamp(self.settings["end_test"])].index
        train_data = df_full.iloc[first_train_day_idx.item():first_test_day_idx.item()] # all the sample before the test year (first test day excluded)
        test_data = df_full.iloc[first_test_day_idx.item():last_test_day_idx.item()] # all the sample after the test year (first test day included, last test day excluded)
        test_data = test_data.iloc[:self.settings["test_days"]*24]

        print(f"First Train Date: {train_data.iloc[0]['ds']}")
        print(f"First Test Date: {test_data.iloc[0]['ds']}")
        print(f"Last Test Date: {test_data.iloc[-1]['ds']}")
        return train_data, test_data
            
    def prepare_finetune_data(self, df: pd.DataFrame, val_days: int):
        def train_multivar_example_gen_func() -> Generator[dict[str, Any], None, None]:
            yield {
                "target": train_df.to_numpy().T,  # array of shape (var, time)
                "start": train_df.index[0],
                "freq": pd.infer_freq(train_df.index),
                "item_id": "item_0",
            }

        def val_multivar_example_gen_func() -> Generator[dict[str, Any], None, None]:
            yield {
                "target": val_df.to_numpy().T,  # array of shape (var, time)
                "start": val_df.index[0],
                "freq": pd.infer_freq(val_df.index),
                "item_id": "item_0",
            }

        # Set 'ds' as index and remove 'ds' and 'unique_id' columns
        df = df.set_index('ds')
        if 'unique_id' in df.columns:
            df = df.drop(columns=['unique_id'])

        train_df = df.iloc[:-val_days, :]  
        val_df = df.iloc[:, :]  

        features = Features(
            dict(
                target=Sequence(
                    Sequence(Value("float32")), length=len(df.columns)
                ),  # multivariate time series are saved as (var, time)
                start=Value("timestamp[s]"),
                freq=Value("string"),
                item_id=Value("string"),
            )
        ) 

        output_train_path = "./tools/cli/results/train_dataset"
        output_val_path = "./tools/cli/results/val_dataset"

        train_dataset = datasets.Dataset.from_generator(
            train_multivar_example_gen_func, features=features
        )
        val_dataset = datasets.Dataset.from_generator(
            val_multivar_example_gen_func, features=features
        )

        train_dataset.save_to_disk(output_train_path)    
        val_dataset.save_to_disk(output_val_path)

        import yaml
        
        offset_adjustment = len(train_df) - val_days
        
        yaml_path = "./tools/cli/conf/finetune/val_data/val_dataset.yaml"
        
        try:
            # Read the current yaml file
            with open(yaml_path, 'r') as file:
                config = yaml.safe_load(file)
            
            # Update the offset value
            config['_args_']['offset'] = offset_adjustment
            
            # Write the updated config back to the file
            with open(yaml_path, 'w') as file:
                yaml.dump(config, file, default_flow_style=False)
            
            print(f"Updated offset in {yaml_path} to {offset_adjustment}")
        except Exception as e:
            print(f"Error updating offset in val_dataset.yaml: {e}")



class PrepareDataCSV:
    def __init__(self, csv_path: str, settings: dict):
        self.csv_path = csv_path
        self.settings = settings

    def load_data(self, include_covariates: bool = True) -> pd.DataFrame:
        print(f"Loading data from {self.csv_path}")
        df = pd.read_csv(self.csv_path, index_col=0, parse_dates=True)

        # Use all real columns from CSV, only rename Price to TARG__target
        df_mapped = df.copy()
        df_mapped = df_mapped.rename(columns={"Price": "TARG__target"})

        df_mapped["unique_id"] = "0"
        df_mapped["ds"] = df.index
        df_mapped = df_mapped.reset_index(drop=True)

        if not include_covariates:
            df_mapped = df_mapped[['ds', 'TARG__target', 'unique_id']]

        return df_mapped

    def prepare_data(self, df_full: pd.DataFrame):
        print("Preparing data for zero-shot prediction...")

        # Filter data based on date ranges
        if "start_train" in self.settings and self.settings["start_train"] is not None:
            first_train_day_idx = df_full[df_full['ds'] == pd.Timestamp(self.settings["start_train"])].index
            if len(first_train_day_idx) == 0:
                # If exact match not found, use first available date
                first_train_day_idx = pd.Index([0])
                print(f"Warning: start_train date {self.settings['start_train']} not found, using first available date")
            else:
                first_train_day_idx = first_train_day_idx[0:1]  # Take first match
        else:
            first_train_day_idx = pd.Index([0])

        # Find first test day - use first timestamp >= start_test
        first_test_mask = df_full['ds'] >= pd.Timestamp(self.settings["start_test"])
        if first_test_mask.any():
            first_test_day_idx = df_full[first_test_mask].index[0:1]  # Take first match
        else:
            # If exact match not found, find the closest available date after start_test
            available_dates_after = df_full[df_full['ds'] > pd.Timestamp(self.settings["start_test"])]['ds']
            if len(available_dates_after) > 0:
                closest_date = available_dates_after.iloc[0]
                first_test_day_idx = df_full[df_full['ds'] == closest_date].index[0:1]
                print(f"Warning: start_test date {self.settings['start_test']} not found, using closest available date {closest_date}")
            else:
                raise ValueError(f"No data found after start_test date {self.settings['start_test']}")

        # Find last test day - use last timestamp <= end_test
        last_test_mask = df_full['ds'] <= pd.Timestamp(self.settings["end_test"])
        if last_test_mask.any():
            last_test_day_idx = df_full[last_test_mask].index[-1:]  # Take last match
        else:
            # If exact match not found, find the closest available date before end_test
            available_dates_before = df_full[df_full['ds'] < pd.Timestamp(self.settings["end_test"])]['ds']
            if len(available_dates_before) > 0:
                closest_date = available_dates_before.iloc[-1]
                last_test_day_idx = df_full[df_full['ds'] == closest_date].index[-1:]
                print(f"Warning: end_test date {self.settings['end_test']} not found, using closest available date {closest_date}")
            else:
                raise ValueError(f"No data found before end_test date {self.settings['end_test']}")

        train_data = df_full.iloc[first_train_day_idx[0]:first_test_day_idx[0]]
        test_data = df_full.iloc[first_test_day_idx[0]:last_test_day_idx[0]+1]
        test_data = test_data.iloc[:self.settings["test_days"]*24]

        print(f"First Train Date: {train_data.iloc[0]['ds']}")
        print(f"First Test Date: {test_data.iloc[0]['ds']}")
        print(f"Last Test Date: {test_data.iloc[-1]['ds']}")

        return train_data, test_data



