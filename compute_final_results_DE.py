import os
from tools.metric_tools import *


for idx, model in enumerate(['jsu_dnn', 'zeroshot_small_de', 'zeroshot_base_de', 'zeroshot_large_de']):
    settings = {
        'bench_folder_name': model,
        "PICP_alpha": 0.8,
        'datasets': ['DE'],    
        "model_name": "moirai",
        "shift_days": 1, 
        "prediction_length": 24,
        "start_train": '2015-01-01 00:00:00',
        "start_test": '2019-06-27 00:00:00',
        "end_test": '2021-01-01 00:00:00',
        "recalib_window_size": 0, 
        "test_days": 554 + 0, 
        "days_used_for_validation": 32*5, 
        "num_samples": 1000, 
        "datasets_paths": {
            'DE': "preprocessing/data/DE.csv",
        },
    }

    # Remove the results file if it already exists
    results_filename = f"./final_results/DE/{settings['bench_folder_name']}_results.txt"
    if os.path.exists(results_filename):
        os.remove(results_filename)
        print(f"File {results_filename} removed.")


    ### LOAD DATA
    for dataset in settings['datasets']:
        cpm = ComputeMetricsFromResults(settings, dataset, is_de=True)
        cpm.load_model_data()
        cpm.compute_single_results()
        cpm.compute_and_plot_picp()
        if idx == 0: # only need to run one time for each dataset
            cpm.DM_test_mae_crps(['jsu_dnn', 'zeroshot_small_de','zeroshot_base_de','zeroshot_large_de']) 

        cpm.save_results()