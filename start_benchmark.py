import os
os.environ["CUSTOM_DATA_PATH"] = "./tools/cli/results"
os.environ["ENV_TYPE"] = "moirai"

import pandas as pd
from tools.prepare_data import PrepareData
from tools.predict import MyPredictor 
import shutil
from tools.utilities import save_results, save_samples_predictions_csv, compute_metrics_for_dataset
from tools.config_utils import update_model_name_in_config, update_patch_size_in_config, update_batch_size_in_config, update_num_epochs_in_config, update_patience_in_config, update_context_length_in_config, update_seed_in_config, update_ft_schedule_in_config
import numpy as np

settings = { 
    "include_covariates": True,
    "finetune": False,

    "seeds": [20, 33, 178, 250, 540],
    
    "model_name": os.getenv('ENV_TYPE'),
    "shift_days": 7, # SIZE OF WINDOW BEETWEEN RECALIBRATION
    "prediction_length": 24,
    "start_train": '2019-01-01 00:00:00', # '2024-01-01 00:00:00', # If you want to use the first day of the dataset as start_train, set it to None
    "start_test": '2023-10-01 00:00:00',
    "end_test": '2024-09-30 00:00:00',
    "recalib_window_size": 0, # Days to add to test set
    "test_days": 365 + 0, # 2024 is a leap year (ONLY FOR TESTING, FOR BENCHMARK USE 366) # ATTENTION: ADD RECALIB_WINDOW_SIZE TO 366 DAYS
    "days_used_for_validation": 32*5, # must be multiple of patchsize
    
    "prediction_type": "samples", # "point" or "samples" or "quantiles"
    "num_samples": 1000, # num samples predicted for each hour of the forecast horizon 
   
    # Choose finetuning schedule: "out_proj" or "out_proj_and_last_encoder_layer" or "full_finetune"
    "ft_schedule": "full_finetune", # describe the type of finetuning
    "finetune_type": "fft_small", # describe the type of finetuning
    
    "model": "moirai_1.1_R_small",
    "context_length": 2048,
    "patch_size": 32, # choose from {"auto", 8, 16, 32, 64, 128}
    "batch_size": 32,
    "num_epochs": 100,
    "patience": 10,

    "datasets_paths": {
        'BE': "preprocessing/data/datasets/DFK_BE_market__2018-12-25__2025-4-17/standard_scaler.p",
        'DE': "preprocessing/data/datasets/DFK_DE_market__2018-12-31__2025-4-17/standard_scaler.p",
        'ES': "preprocessing/data/datasets/DFK_ES_market__2018-12-25__2025-4-17/standard_scaler.p",
        'SE_3': "preprocessing/data/datasets/DFK_SE_3_market__2018-12-25__2025-4-17/standard_scaler.p",
    },
}

#######################################################################################################################################################################

for seed in (settings["seeds"] if settings["finetune"] else [20]):

    # update yaml files
    update_seed_in_config(seed)
    update_patch_size_in_config(settings["patch_size"])
    update_batch_size_in_config(settings["batch_size"])
    update_model_name_in_config(settings["model"])
    update_num_epochs_in_config(settings["num_epochs"])
    update_patience_in_config(settings["patience"])
    update_context_length_in_config(settings["context_length"])
    update_ft_schedule_in_config(settings["ft_schedule"], settings["model"])
    

    if settings["finetune"] == False and os.getenv("USE_LORA") == "True":
        raise ValueError("USE_LORA must be False when finetune is False")

    # Update settings with current seed
    settings["seed"] = seed
    
    # Create run directory for this seed (call save_results early to get the path)
    run_dir = None

    # Initialize current run results
    current_results = {}
    for dname in settings["datasets_paths"].keys():
        if dname not in current_results:
            current_results[dname] = {
                'mae': None,
                'rmse': None,
                'crps': None,
                'picp': None,
            }

    # For each dataset 
    for dataset_name, dataset_path in settings["datasets_paths"].items():   
        print(f"\n\nProcessing {dataset_name} dataset...")

        col_names = ["TARG__target",
                "FUTU__load_f",
                "FUTU__wind_f",
                "FUTU__solar_f",
                "CONS__wd_sin",
                "CONS__wd_cos",
                "CONS__ts_age"]

        # Check if dataset is SE_3 and remove solar column if needed
        if dataset_name == 'SE_3' and 'FUTU__solar_f' in col_names:
            print(f"Removing FUTU__solar_f column for {dataset_name} dataset")
            col_names.remove('FUTU__solar_f')
        
        data_tool = PrepareData(dataset_path, col_names, settings, dataset_name)
        loaded_df = data_tool.load_data(include_covariates=settings["include_covariates"])

        print(f"Training {settings['model_name']} model for {dataset_name} dataset...")

        train_data, test_data = data_tool.prepare_data(loaded_df)
        
        feature_columns = [col for col in train_data.columns if col not in ['ds', 'TARG__target', 'unique_id']]
        
        predictor = MyPredictor(settings, dataset_name, col_names)
        
        # Initialize arrays to store predictions and quantiles
        point_predictions_array = [] 
        samples_predictions_array = np.empty((0, settings["num_samples"]))
        quantiles_predictions_array = np.empty((0, 9))

        # Fine tuning  
        for day in range(settings["test_days"]):
            if day % settings["shift_days"] == 0: 
                # Delete the checkpoint file if it exists
                checkpoint_dir = f"./tools/cli/outputs/finetune/{settings['model']}/train_dataset/{'default_run' if os.getenv('USE_LORA') == 'True' else 'finetuning_scheduler_run'}/checkpoints"
                print(f"Checkpoint directory: {checkpoint_dir}")
                if os.path.exists(checkpoint_dir):
                    print(f"Removing checkpoint directory: {checkpoint_dir}")
                    shutil.rmtree(checkpoint_dir)

                if settings["finetune"]: 
                    print(f"Preparing finetune data...")
                    data_tool.prepare_finetune_data(train_data, settings["days_used_for_validation"])
                    print(f"Finetuning...")
                    predictor.finetune()

                checkpoint_path = f"{checkpoint_dir}/checkpoint_best.ckpt" 
                print(f"Creating predictor...")
                predictor.prepare_model(train_data, checkpoint_path)

            print(f"Predicting day {day}...") 
            predictions, quantiles_predictions, samples_predictions = predictor.predict(train_data, test_data)

            if predictions is not None: assert predictions.shape == (settings["prediction_length"],)
            if quantiles_predictions is not None: assert quantiles_predictions.shape == (settings["prediction_length"], 9)
            if samples_predictions is not None: assert samples_predictions.T.shape == (settings["prediction_length"], settings["num_samples"])
            point_predictions_array += predictions.tolist()
            samples_predictions_array = np.vstack((samples_predictions_array, samples_predictions.T))
            quantiles_predictions_array = np.vstack((quantiles_predictions_array, quantiles_predictions))

            train_data = pd.concat([train_data, test_data.iloc[:24]]) # add day to training set
            test_data = test_data.iloc[24:] # go to next test day

        point_predictions_array = np.array(point_predictions_array)
        print("POINT PREDICTIONS ARRAY SHAPE: ", point_predictions_array.shape)
        print("SAMPLES PREDICTIONS ARRAY SHAPE: ", samples_predictions_array.shape)
        print("QUANTILES PREDICTIONS ARRAY SHAPE: ", quantiles_predictions_array.shape)

        if point_predictions_array.shape[0] > settings["test_days"]*24: # since I predict one week at a time, I might have more predictions than one year, cut the excess
            point_predictions_array = point_predictions_array[:settings["test_days"]*24]
            samples_predictions_array = samples_predictions_array[:settings["test_days"]*24, :]
            quantiles_predictions_array = quantiles_predictions_array[:settings["test_days"]*24, :]

        # save data - create run_dir if not exists yet (before saving samples)
        if run_dir is None:
            run_dir = save_results(settings, current_results, run_dir)
        
        # Save samples predictions as CSV for this dataset and seed
        save_samples_predictions_csv(settings, dataset_name, samples_predictions_array, run_dir)
        
        # Compute metrics using ComputeMetricsFromResults approach
        print(f"Computing metrics for {dataset_name} using ComputeMetricsFromResults...")
        _, truth_values = data_tool.prepare_data(loaded_df)
        metrics = compute_metrics_for_dataset(settings, dataset_name, run_dir, truth_values['TARG__target'].values)
        
        current_results[dataset_name]['mae'] = metrics['mae']
        current_results[dataset_name]['rmse'] = metrics['rmse']
        current_results[dataset_name]['crps'] = metrics['crps']
        current_results[dataset_name]['picp'] = metrics['picp']
        
    save_results(settings, current_results, run_dir)
