import os
os.environ["CUSTOM_DATA_PATH"] = "./tools/cli/results"
os.environ["ENV_TYPE"] = "moirai"
os.environ["USE_LORA"] = "False"

import pandas as pd
import numpy as np
import shutil
import pickle
import sys
from tools.predict import MyPredictor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tools.metric_tools import compute_crps, compute_picp
from tools.utilities import save_samples_predictions_csv
from tools.prepare_data import PrepareDataCSV

def save_results_DE(settings: dict, current_results: dict, run_dir) -> str:
    """
    Modified version of save_results that adds '_DE' suffix to folder name
    """
    if run_dir is None:
        # Create base directory for results if it doesn't exist
        base_results_dir = "benchmark_results"
        os.makedirs(base_results_dir, exist_ok=True)

        # Create the folder name for this specific run (with _DE suffix)
        if settings['finetune']:
            folder_name = f"results_bench_{settings['model_name']}_finetune_{settings.get('finetune_type', 'default')}_seed_{settings['seed']}_DE"
        else:
            folder_name = f"results_bench_{settings['model_name']}_zeroshot_seed_{settings['seed']}_DE"

        # Handle folder name conflicts by adding progressive numbers
        final_folder_name = folder_name
        counter = 1
        while os.path.exists(os.path.join(base_results_dir, final_folder_name)):
            final_folder_name = f"{folder_name}_{counter}"
            counter += 1

        # Create the run-specific directory
        run_dir = os.path.join(base_results_dir, final_folder_name)
        os.makedirs(run_dir, exist_ok=True)

        return run_dir

    # Create conf_files directory inside the run directory
    conf_files_dir = os.path.join(run_dir, "conf_files")
    os.makedirs(conf_files_dir, exist_ok=True)

    # Define source paths for configuration files
    finetune_conf_dir = "tools/cli/conf/finetune"
    files_to_copy = {
        "default.yaml": os.path.join(finetune_conf_dir, "default.yaml"),
        "val_dataset.yaml": os.path.join(finetune_conf_dir, "val_data", "val_dataset.yaml"),
        f"{settings['model_name']}.yaml": os.path.join(finetune_conf_dir, "model", f"{settings['model']}.yaml")
    }

    # Copy configuration files
    for dest_name, source_path in files_to_copy.items():
        if os.path.exists(source_path):
            try:
                shutil.copy2(source_path, os.path.join(conf_files_dir, dest_name))
                print(f"Copied {source_path} to {conf_files_dir}")
            except Exception as e:
                print(f"Error copying {source_path}: {e}")
        else:
            print(f"Warning: Source file {source_path} not found")

    # Set the path for the results pickle file
    results_filename = os.path.join(run_dir, "benchmark_results.pkl")

    all_results = {}

    # Add the current model results to the all_results dictionary
    if settings["model_name"] not in all_results:
        all_results[settings["model_name"]] = {}

    for dataset_name in current_results:
        all_results[settings["model_name"]][dataset_name] = current_results[dataset_name]

    # Save the updated results dictionary to the new pickle file
    with open(results_filename, 'wb') as f:
        pickle.dump(all_results, f)

    # Save the printed results to a text file
    results_txt_path = os.path.join(run_dir, "results_summary.txt")
    with open(results_txt_path, 'w') as f:
        # Redirect print output to the file
        original_stdout = sys.stdout
        sys.stdout = f
        _print_results(all_results, current_results)
        sys.stdout = original_stdout

    # Save settings to a text file
    settings_txt_path = os.path.join(run_dir, "run_settings.txt")
    with open(settings_txt_path, 'w') as f:
        for key, value in settings.items():
            f.write(f"{key}: {value}\n")

    print(f"Results saved in directory: {run_dir}")
    print(f"Files created:")
    print(f"- {results_filename}")
    print(f"- {results_txt_path}")
    print(f"- {settings_txt_path}")
    print(f"- Configuration files in {conf_files_dir}/")

    # Still print results to console
    _print_results(all_results, current_results)

    return run_dir


def _print_results(all_results: dict, current_results: dict) -> None:
    print("\n----- RESULTS FOR CURRENT RUN -----")
    for dataset_name in current_results.keys():
        print(f"\n\nDATASET: {dataset_name}")
        print(f"\tMAE: {current_results[dataset_name]['mae']}")
        print(f"\tRMSE: {current_results[dataset_name]['rmse']}")
        print(f"\tCRPS: {current_results[dataset_name]['crps']}")
        print(f"\tPICP: {current_results[dataset_name]['picp']}")
        print()
    if len(all_results) > 1:  # If we have results from more than just this run
        print("\n----- ALL HISTORICAL RESULTS -----")
        for model in all_results:
            print(f"\nMODEL: {model}")
            for dataset in all_results[model]:
                print(f"  DATASET: {dataset}")
                print(f"    MAE: {all_results[model][dataset]['mae']}")
                print(f"    RMSE: {all_results[model][dataset]['rmse']}")
                print(f"    CRPS: {all_results[model][dataset]['crps']}")
                print(f"    PICP: {all_results[model][dataset]['picp']}")


settings = {
    "include_covariates": True,
    "finetune": False,  # Zero-shot only
    "seed": 20, 

    "model_name": os.getenv('ENV_TYPE'),
    "shift_days": 1,
    "prediction_length": 24,
    "start_train": '2015-01-01 00:00:00',
    "start_test": '2019-06-27 00:00:00',
    "end_test": '2021-01-01 00:00:00',
    "test_days": 554,

    "prediction_type": "samples",
    "num_samples": 1000,

    "model": "moirai_1.1_R_small",
    "context_length": 2048,
    "patch_size": 32,
    "batch_size": 32,
    "num_epochs": 100,
    "patience": 10,
}

#######################################################################################################################################################################

for seed in (settings["seeds"] if settings["finetune"] else [20]):
    # Update settings with current seed
    settings["seed"] = seed

    print("\n=== Zero-shot Energy Price Forecasting Benchmark for DE Dataset ===")
    print(f"Model: {settings['model']}")
    print(f"Seed: {settings['seed']}")
    print(f"Context Length: {settings['context_length']}")
    print(f"Prediction Length: {settings['prediction_length']}")
    print(f"Test Period: {settings['start_test']} to {settings['end_test']}")

    # Initialize current run results 
    current_results = {}
    run_dir = None

    for dname in ['DE']:  # Single dataset
        if dname not in current_results:
            current_results[dname] = {
                'mae': None,
                'rmse': None,
                'crps': None,
                'picp': None,
            }

    # Load and prepare data
    data_tool = PrepareDataCSV("preprocessing/data/DE.csv", settings)
    loaded_df = data_tool.load_data(include_covariates=settings["include_covariates"])

    print(f"Loaded dataframe shape: {loaded_df.shape}")
    print(f"Date range: {loaded_df['ds'].min()} to {loaded_df['ds'].max()}")

    train_data, test_data = data_tool.prepare_data(loaded_df)

    # Column names for predictor - using all real CSV columns only
    col_names = ["TARG__target",
                "Load_DA_Forecast",
                "Renewables_DA_Forecast",
                "EUA",
                "API2_Coal",
                "TTF_Gas",
                "Brent_oil"]

    predictor = MyPredictor(settings, "DE", col_names)

    # Initialize arrays to store predictions
    point_predictions_array = []
    samples_predictions_array = np.empty((0, settings["num_samples"]))
    quantiles_predictions_array = np.empty((0, 9))

    print(f"\nStarting zero-shot prediction for {settings['test_days']} days...")

    # Zero-shot prediction loop
    for day in range(settings["test_days"]):
        if day % settings["shift_days"] == 0:
            print(f"Preparing model for day {day}...")
            predictor.prepare_model(train_data, None)  # No checkpoint for zero-shot

        print(f"Predicting day {day}/{settings['test_days']}...")

        predictions, quantiles_predictions, samples_predictions = predictor.predict(train_data, test_data)

        # Validate predictions shape
        if predictions is not None:
            assert predictions.shape == (settings["prediction_length"],)
        if quantiles_predictions is not None:
            assert quantiles_predictions.shape == (settings["prediction_length"], 9)
        if samples_predictions is not None:
            assert samples_predictions.T.shape == (settings["prediction_length"], settings["num_samples"])

        # Store predictions
        point_predictions_array += predictions.tolist()
        samples_predictions_array = np.vstack((samples_predictions_array, samples_predictions.T))
        quantiles_predictions_array = np.vstack((quantiles_predictions_array, quantiles_predictions))

        # Update training data with new day
        train_data = pd.concat([train_data, test_data.iloc[:24]])
        test_data = test_data.iloc[24:]

    # Convert to numpy array
    point_predictions_array = np.array(point_predictions_array)

    print(f"\nPrediction completed!")
    print(f"Point predictions shape: {point_predictions_array.shape}")
    print(f"Samples predictions shape: {samples_predictions_array.shape}")
    print(f"Quantiles predictions shape: {quantiles_predictions_array.shape}")

    # Trim predictions if needed
    if point_predictions_array.shape[0] > settings["test_days"]*24:
        point_predictions_array = point_predictions_array[:settings["test_days"]*24]
        samples_predictions_array = samples_predictions_array[:settings["test_days"]*24, :]
        quantiles_predictions_array = quantiles_predictions_array[:settings["test_days"]*24, :]

    # Load actual values for metric computation using the same logic as prepare_data
    test_start_mask = loaded_df['ds'] >= pd.Timestamp(settings["start_test"])
    if test_start_mask.any():
        test_start_idx = loaded_df[test_start_mask].index[0:1]
    else:
        # If exact match not found, find the closest available date after start_test
        available_dates_after = loaded_df[loaded_df['ds'] > pd.Timestamp(settings["start_test"])]['ds']
        if len(available_dates_after) > 0:
            closest_date = available_dates_after.iloc[0]
            test_start_idx = loaded_df[loaded_df['ds'] == closest_date].index[0:1]
            print(f"Warning: start_test date {settings['start_test']} not found for metrics, using closest available date {closest_date}")
        else:
            raise ValueError(f"No data found after start_test date {settings['start_test']}")

    actual_values = loaded_df.iloc[test_start_idx[0]:test_start_idx[0] + settings["test_days"]*24]["TARG__target"].values

    # Save samples predictions as CSV (before computing metrics)
    if run_dir is None:
        run_dir = save_results_DE(settings, current_results, run_dir)

    # Save samples predictions as CSV for this dataset and seed
    save_samples_predictions_csv(settings, "DE", samples_predictions_array, run_dir)

    # Compute metrics using the same approach as ComputeMetricsFromResults
    if len(actual_values) == len(point_predictions_array):
        # Set up quantiles and bounds like in ComputeMetricsFromResults
        quantiles = [i/10 for i in range(1, 10)]  # 9 quantiles from 10% to 90%
        PICP_alpha = 0.8  # 80% confidence interval
        inv_alpha = (1-PICP_alpha)/2
        ub = int((1-inv_alpha)*10) - 1  # Upper bound index (8 for 90%)
        lb = int((inv_alpha)*10) - 1    # Lower bound index (0 for 10%)

        zsquantiles = np.quantile(samples_predictions_array, quantiles, axis=1).T

        # MAE using median (50th percentile, index 4 for 9 quantiles)
        mae = mean_absolute_error(actual_values, zsquantiles[:, 4])

        # RMSE using median (50th percentile, index 4 for 9 quantiles)
        rmse = np.sqrt(mean_squared_error(actual_values, zsquantiles[:, 4]))

        # CRPS using the exact same approach as ComputeMetricsFromResults
        crps = np.mean(compute_crps(
            actual_values.reshape(settings['test_days'], 24),
            zsquantiles.reshape(settings['test_days'], 24, 9),
            quantiles
        ))

        # PICP using the exact same approach as ComputeMetricsFromResults
        picp = np.mean(compute_picp(
            actual_values.reshape(settings['test_days'], 24),
            zsquantiles.reshape(settings['test_days'], 24, 9),
            lb, ub
        ))

        # Store results in current_results structure
        current_results["DE"]['mae'] = mae
        current_results["DE"]['rmse'] = rmse
        current_results["DE"]['crps'] = crps
        current_results["DE"]['picp'] = picp

        print(f"\n=== RESULTS FOR DE DATASET (Zero-shot) ===")
        print(f"Actual values shape: {actual_values.shape}")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"CRPS: {crps:.4f}")
        print(f"PICP ({PICP_alpha*100}%): {picp:.4f}")

        # Save final results
        save_results_DE(settings, current_results, run_dir)
        print(f"\nBenchmark completed successfully for seed {settings['seed']}!")
    else:
        print(f"Error: Mismatch in array lengths - actual: {len(actual_values)}, predicted: {len(point_predictions_array)}")
