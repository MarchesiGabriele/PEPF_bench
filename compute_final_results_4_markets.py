import os
from tools.metric_tools import *


for idx, model in enumerate(['jsu_dnn','zeroshot_small', 'zeroshot_base', 'zeroshot_large', 'fft_small', 'fft_base']):
    settings = {
        'bench_folder_name': model,
        "PICP_alpha": 0.9,
        'datasets': ['BE', 'DE', 'ES', 'SE_3'],    
        "model_name": "moirai",
        "shift_days": 7, 
        "prediction_length": 24,
        "start_train": '2019-01-01 00:00:00',
        "start_test": '2023-10-01 00:00:00',
        "end_test": '2024-10-01 00:00:00', 
        "recalib_window_size": 0, 
        "test_days": 366 + 0, 
        "days_used_for_validation": 32*5, 
        "num_samples": 1000, 
        "datasets_paths": {
            'BE': "preprocessing/data/datasets/DFK_BE_market__2018-12-25__2025-4-17/standard_scaler.p",
            'DE': "preprocessing/data/datasets/DFK_DE_market__2018-12-31__2025-4-17/standard_scaler.p",
            'ES': "preprocessing/data/datasets/DFK_ES_market__2018-12-25__2025-4-17/standard_scaler.p",
            'SE_3': "preprocessing/data/datasets/DFK_SE_3_market__2018-12-25__2025-4-17/standard_scaler.p",
        },
    }

    # Remove the results file if it already exists
    results_filename = f"./final_results/4_markets/{settings['bench_folder_name']}_results.txt"
    if os.path.exists(results_filename):
        os.remove(results_filename)
        print(f"File {results_filename} removed.")


    ### LOAD DATA
    for dataset in settings['datasets']:
        cpm = ComputeMetricsFromResults(settings, dataset)

        cpm.load_model_data()

        cpm.compute_single_results()
        if "zeroshot" not in settings['bench_folder_name'] and "jsu_dnn" not in settings['bench_folder_name']:
            cpm.compute_probabilistic_aggregation()
            cpm.compute_quantile_aggregation()
        cpm.compute_and_plot_picp()
        if "fft" in settings['bench_folder_name']:
            cpm.compute_and_plot_picp(agg='quantile')
        if idx == 0: # only need to run one time for each dataset
            cpm.DM_test_mae_crps(['jsu_dnn', 'zeroshot_small','zeroshot_base','zeroshot_large','fft_small_p','fft_small_v', 'fft_base_p','fft_base_v']) 

        cpm.save_results()