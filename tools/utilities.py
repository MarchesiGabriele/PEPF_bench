import os
import pickle
import numpy as np
import pandas as pd
import shutil


def save_results(settings: dict, current_results: dict, run_dir) -> str:
    if run_dir is None:
        # Create base directory for results if it doesn't exist
        base_results_dir = "benchmark_results"
        os.makedirs(base_results_dir, exist_ok=True)

        # Create the folder name for this specific run
        finetune_type = settings.get('finetune_type', 'default')  # Get finetune type or use 'default' if not specified
        if settings['finetune']:
            folder_name = f"results_bench_{settings['model_name']}_finetune_{finetune_type}_seed_{settings['seed']}"
        else:
            folder_name = f"results_bench_{settings['model_name']}_zero_shot"
        
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
        import sys
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
        


def save_samples_as_dataframe(settings: dict, dataset_name: str, samples_predictions_array: np.ndarray) -> None:
    """
    Convert samples_predictions_array to a DataFrame and save it as a pickle file.
    
    Args:
        settings: Dictionary containing model settings
        dataset_name: Name of the dataset
        samples_predictions_array: NumPy array containing sample predictions
    """
    # Create directory for samples if it doesn't exist
    samples_dir = "samples_results"
    os.makedirs(samples_dir, exist_ok=True)
    
    # Convert samples to DataFrame
    # Create column names for each sample
    columns = [f'sample_{i}' for i in range(samples_predictions_array.shape[1])]
    samples_df = pd.DataFrame(samples_predictions_array, columns=columns)
    
    # Set the path for the samples file
    filename = f"{samples_dir}/{settings['model_name']}_{dataset_name}_{'with_covariates' if settings['include_covariates'] else 'without_covariates'}_{'finetune' if settings['finetune'] else 'zero_shot'}.pkl"
    
    # Save the DataFrame to a pickle file
    samples_df.to_pickle(filename)
    
    print(f"Samples DataFrame saved to {filename}")


def save_samples_predictions_csv(settings: dict, dataset_name: str, samples_predictions_array: np.ndarray, run_dir: str) -> None:
    dataset_dir = os.path.join(run_dir, f"samples_{dataset_name}")
    os.makedirs(dataset_dir, exist_ok=True)
    
    current_seed = settings.get('seed', 'unknown_seed')
    
    columns = [f'sample_{i}' for i in range(samples_predictions_array.shape[1])]
    
    time_index = range(samples_predictions_array.shape[0])
    
    samples_df = pd.DataFrame(samples_predictions_array, columns=columns, index=time_index)
    samples_df.index.name = 'hour'
    
    csv_filename = f"{dataset_name}_samples_seed_{current_seed}.csv"
    csv_path = os.path.join(dataset_dir, csv_filename)
    samples_df.to_csv(csv_path, index=True)
    
    print(f"Samples CSV saved: {csv_path}")
    print(f"Shape: {samples_predictions_array.shape} -> CSV with {len(samples_df)} rows and {len(columns)} sample columns")


def compute_metrics_for_dataset(settings: dict, dataset_name: str, run_dir: str, truth_values: np.ndarray) -> dict:
    from tools.metric_tools import ComputeMetricsFromResults
    
    temp_settings = {
        'bench_folder_name': 'temp_single_seed', 
        'PICP_alpha': 0.9,
        'datasets': [dataset_name],
        'test_days': settings['test_days'],
        'datasets_paths': settings['datasets_paths']
    }
    
    cpm = ComputeMetricsFromResults(temp_settings, dataset_name)
    
    current_seed = settings.get('seed', 'unknown_seed')
    
    dataset_dir = os.path.join(run_dir, f"samples_{dataset_name}")
    csv_filename = f"{dataset_name}_samples_seed_{current_seed}.csv"
    csv_path = os.path.join(dataset_dir, csv_filename)
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Samples file not found: {csv_path}")
    
    samples_df = pd.read_csv(csv_path, index_col=0)  
    samples_data = samples_df.values  
    
    cpm.seed_datasets = {str(current_seed): samples_data}
    
    cpm.truth_values = truth_values
    
    cpm.seeds = [str(current_seed)]
    cpm.quantiles = [i/100 for i in range(1, 100)]
    inv_alpha = (1-temp_settings['PICP_alpha'])/2
    cpm.ub = int((1-inv_alpha)*100)
    cpm.lb = int((inv_alpha)*100)
    
    cpm.results = {"dataset_region": dataset_name, "each_seed_results": {}}
    
    myquantiles = np.quantile(samples_data, cpm.quantiles, axis=1).T
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from tools.metric_tools import compute_crps, compute_picp
    
    mae = mean_absolute_error(cpm.truth_values, myquantiles[:, 49])  # 49 is the median (50th percentile)
    rmse = np.sqrt(mean_squared_error(cpm.truth_values, myquantiles[:, 49]))
    crps = np.mean(compute_crps(cpm.truth_values.reshape(settings['test_days'], 24), 
                               myquantiles.reshape(settings['test_days'], 24, 99), cpm.quantiles))
    picp = np.mean(compute_picp(cpm.truth_values.reshape(settings['test_days'], 24), 
                               myquantiles.reshape(settings['test_days'], 24, 99), cpm.lb, cpm.ub))
    
    return {'mae': mae, 'rmse': rmse, 'crps': crps, 'picp': picp}
