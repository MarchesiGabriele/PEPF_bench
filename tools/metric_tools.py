import numpy as np
from typing import List
from scipy import stats
import os
import pandas as pd
from .prepare_data import PrepareData
from sklearn.metrics import mean_absolute_error, mean_squared_error 
import matplotlib.pyplot as plt
import matplotlib as mpl    
import pickle
import json
from .prepare_data import PrepareDataCSV

def compute_crps(labels, pred_quantiles, quantiles):
    """
    labels: truth values [366 x 24]
    pred_quantiles: predictions (for each quantile) [366 x 24 x n_quantiles]
    quantiles: list of quantiles [0.01, ..., 0.99] or [0.1, ..., 0.9]
    """
    
    loss = []
    for i, q in enumerate(quantiles):
        error = np.subtract(labels, pred_quantiles[:, :, i])
        loss_q = np.maximum(q * error, (q - 1) * error)
        loss.append(np.expand_dims(loss_q,-1))
    loss = np.mean(np.concatenate(loss, axis=-1), axis=-1)
    print("compute_crps function completed")
    return loss


def compute_picp(labels, pred_quantiles, lq_idx, uq_idx):
    """
    labels: truth values [366 x 24] 
    pred_quantiles: predictions (for each quantile) [366 x 24 x 99]
    """
    picp = []
    for hour in range(24):
        PI_hits = np.logical_and(pred_quantiles[:, hour, lq_idx] <= labels[:, hour],
                                labels[:, hour] <= pred_quantiles[:, hour, uq_idx])
        picp.append(np.mean(PI_hits))
    print("compute_picp function completed")
    return np.mean(picp)*100


def DM(p_real, losses_model_1, losses_model_2, version='multivariate'):
    """
    Parameters
    ----------
    p_real : numpy.ndarray
        Array of shape :math:`(n_\\mathrm{days}, n_\\mathrm{prices/day})` representing the real market
        prices
    losses_model_1 : TYPE
        Array of shape :math:`(n_\\mathrm{days}, n_\\mathrm{prices/day})` representing the losses of the first model
    losses_model_2 : TYPE
        Array of shape :math:`(n_\\mathrm{days}, n_\\mathrm{prices/day})` representing the losses of the second model
    norm : int, optional
        Norm used to compute the loss differential series. At the moment, this value must either
        be 1 (for the norm-1) or 2 (for the norm-2).
    version : str, optional
        Version of the test as defined in
        `here <https://epftoolbox.readthedocs.io/en/latest/modules/cite.html>`_. It can have two values:
        ``'univariate`` or ``'multivariate``
    Returns
    -------
    float, numpy.ndarray
        The p-value after performing the test. It is a float in the case of the multivariate test
        and a numpy array with a p-value per hour for the univariate test

    """
    # Checking that all time series have the same shape
    if p_real.shape != losses_model_1.shape or p_real.shape != losses_model_2.shape:
        raise ValueError('The three time series must have the same shape')

    # Ensuring that time series have shape (n_days, n_prices_day)
    if len(p_real.shape) == 1 or (len(p_real.shape) == 2 and p_real.shape[1] == 1):
        raise ValueError('The time series must have shape (n_days, n_prices_day')

    # Computing the test statistic
    if version == 'univariate':
        # Computing the loss differential series for the univariate test
        d = losses_model_1 - losses_model_2

        # Computing the test statistic
        mean_d = np.mean(d, axis=0)
        var_d = np.var(d, ddof=0, axis=0)

    elif version == 'multivariate':
        # Computing the loss differential series for the multivariate test
        d = np.mean(losses_model_1, axis=1) - np.mean(losses_model_2, axis=1)
        # Computing the test statistic
        mean_d = np.mean(d)
        var_d = np.var(d, ddof=0)

    N = d.shape[0]
    DM_stat = mean_d / np.sqrt((1 / N) * var_d)

    p_value = 1 - stats.norm.cdf(DM_stat)

    print("DM function completed")
    return p_value


class ComputeMetricsFromResults:    
    def __init__(self, settings, dataset_name, is_de=False, compare_context=False):
        self.truth_values = None
        self.zeroshot_df = None
        self.seed_datasets = dict()
        self.is_de = is_de
        self.seeds = []
        self.quantiles = [] 
        self.ub = None 
        self.lb = None 
        self.dataset_name = dataset_name
        self.results = {"dataset_region": dataset_name, 
                "model_info": settings["bench_folder_name"],
                "each_seed_results":{},
               }
        self.settings = settings
        print("__init__ function completed")
        self.model_names_coded ={
        "zeroshot_small": "S-ZS" if not compare_context else "S-2048",
        "zeroshot_base": "B-ZS" if not compare_context else "B-2048",
        "zeroshot_large": "L-ZS" if not compare_context else "L-2048",
        "jsu_dnn": "JSU-DNN",
        "fft_small": "S-FFT",
        "fft_base": "B-FFT",
        "fft_small_p": "S-FFT-P",
        "fft_small_v": "S-FFT-V",
        "fft_base_p": "B-FFT-P",
        "fft_base_v": "B-FFT-V",
        "zeroshot_small_nocov": "S-NC",
        "zeroshot_base_nocov": "B-NC",
        "zeroshot_large_nocov": "L-NC",
        "zeroshot_small_256": "S-256",
        "zeroshot_base_256": "B-256",
        "zeroshot_large_256": "L-256",
        "zeroshot_small_512": "S-512",
        "zeroshot_base_512": "B-512",
        "zeroshot_large_512": "L-512",
        "zeroshot_small_1024": "S-1024",
        "zeroshot_base_1024": "B-1024",
        "zeroshot_large_1024": "L-1024",
        "zeroshot_small_4096": "S-4096",
        "zeroshot_base_4096": "B-4096",
        "zeroshot_large_4096": "L-4096",
        "zeroshot_small_de": "S-DE",
        "zeroshot_base_de": "B-DE",
        "zeroshot_large_de": "L-DE",
        }
 


    def load_model_data(self) -> None:
        each_dataset_model = {} 
        if "zeroshot" in self.settings['bench_folder_name']:
            path = f'./benchmark_results/{self.settings["bench_folder_name"]}/samples_{self.dataset_name}/'
            df_model = pd.read_csv(f'{path}{self.dataset_name}_samples_seed_20.csv').iloc[:,1:] # only one seed for zeroshot models
            df_model = df_model.values
            self.zeroshot_df = df_model
        elif "jsu_dnn" in self.settings['bench_folder_name']:
            path = f'./benchmark_results/DNN/' if not self.is_de else f'./benchmark_results/DE/'
            with open(f'{path}{self.dataset_name}_JSU-DNN.p', 'rb') as f:
                df_model = pickle.load(f)
            df_model = df_model['results'] if not self.is_de else df_model['results']['JSU-DNN']
            self.zeroshot_df = df_model.iloc[:,1:].values
            self.truth_values = df_model['target'].values if not self.is_de else df_model['DE_price'].values
            self.seeds = [1]
            self.quantiles = [i/10 for i in range(1, 10)] if self.is_de else [i/100 for i in range(1, 100)]
            inv_alpha = (1-self.settings['PICP_alpha'])/2
            if self.is_de:
                # For DE dataset, always use 9 quantiles (0.1 to 0.9)
                self.ub = int((1-inv_alpha)*10)-1
                self.lb = int((inv_alpha)*10)-1
            else:
                # For other datasets, use 99 quantiles (0.01 to 0.99)
                self.ub = int((1-inv_alpha)*100)-1
                self.lb = int((inv_alpha)*100)-1
            self.results["picp%"] = self.settings['PICP_alpha']*100
            print("load_model_data function completed")
            return
        else:
            path = f'./benchmark_results/{self.settings["bench_folder_name"]}/'
            for file in os.listdir(path):
                seed = file.split('_')[-1] 
                df_model = pd.read_csv(f'{path}{file}/samples_{self.dataset_name}/{self.dataset_name}_samples_seed_{seed}.csv')
                each_dataset_model[seed] = df_model.iloc[:, 1:]
            self.seed_datasets = each_dataset_model 

        # Load and prepare truth data 
        if self.is_de:
            data_tool = PrepareDataCSV(self.settings["datasets_paths"][self.dataset_name], self.settings)
            loaded_df = data_tool.load_data(include_covariates=True)
            _, test_data = data_tool.prepare_data(loaded_df)
        else:
            data_tool = PrepareData(self.settings["datasets_paths"][self.dataset_name], ['TARG__target'], self.settings, self.dataset_name)
            loaded_df = data_tool.load_data(include_covariates=True)
            _, test_data = data_tool.prepare_data(loaded_df)
        self.truth_values = test_data['TARG__target'].values
        self.seeds = [seed for seed in self.seed_datasets.keys()]
        self.quantiles = [i/100 for i in range(1, 100)] if not self.is_de else [i/10 for i in range(1, 10)]
        inv_alpha = (1-self.settings['PICP_alpha'])/2
        self.ub = int((1-inv_alpha)*100)-1 if not self.is_de else int((1-inv_alpha)*10)-1
        self.lb = int((inv_alpha)*100)-1 if not self.is_de else int((inv_alpha)*10)-1
        self.results["picp%"] = self.settings['PICP_alpha']*100
        print("load_model_data function completed")


    def get_prob_aggregation_quantiles(self):
        """
        Stack all the predictions for each seed and compute the quantiles.
        """
        all_seeds_predictions = None
        for seed in self.seeds:
            seed_predictions = self.seed_datasets[seed] 
            if all_seeds_predictions is None:
                all_seeds_predictions = seed_predictions
            else:
                all_seeds_predictions = np.hstack((all_seeds_predictions, seed_predictions))
        print("get_prob_aggregation_quantiles function completed")
        return np.quantile(all_seeds_predictions, self.quantiles, axis=1).T
    

    def get_quantile_aggregation_quantiles(self):
        """
        Compute the quantiles for each seed and then average them.
        """
        seed_quantiles = []
        for seed in self.seeds:
            seed_predictions = self.seed_datasets[seed].values
            seed_quantile = np.quantile(seed_predictions, self.quantiles, axis=1).T
            seed_quantiles.append(seed_quantile)
        seed_quantiles = np.array(seed_quantiles)
        print("get_quantile_aggregation_quantiles function completed")
        return np.mean(seed_quantiles, axis=0)


    def compute_single_results(self) -> None:
        """ 
        Compute mae, rmse, crps, picp.
        If the model is not zero shot (i.e. has more seed results), compute the results for each seed.
        """
        if "zeroshot" in self.settings['bench_folder_name']:
            zsquantiles = np.quantile(self.zeroshot_df, self.quantiles, axis=1).T
            num_quantiles = 9 if self.is_de else 99
            median_idx = num_quantiles // 2  # Use middle quantile as median
            self.results["mae"] = mean_absolute_error(self.truth_values, zsquantiles[:, median_idx])
            self.results["rmse"] = np.sqrt(mean_squared_error(self.truth_values, zsquantiles[:, median_idx]))
            self.results["crps"] = np.mean(compute_crps(self.truth_values.reshape(self.settings['test_days'], 24), zsquantiles.reshape(self.settings['test_days'], 24, num_quantiles), self.quantiles))
            self.results["picp"] = np.mean(compute_picp(self.truth_values.reshape(self.settings['test_days'], 24), zsquantiles.reshape(self.settings['test_days'], 24, num_quantiles), self.lb, self.ub))
        elif "jsu_dnn" in self.settings['bench_folder_name']:
            zsquantiles = self.zeroshot_df
            num_quantiles = 9 if self.is_de else 99
            median_idx = num_quantiles // 2  # Use middle quantile as median
            self.results["mae"] = mean_absolute_error(self.truth_values, zsquantiles[:, median_idx])
            self.results["rmse"] = np.sqrt(mean_squared_error(self.truth_values, zsquantiles[:, median_idx]))
            self.results["crps"] = np.mean(compute_crps(self.truth_values.reshape(self.settings['test_days'], 24), zsquantiles.reshape(self.settings['test_days'], 24, num_quantiles), self.quantiles))
            self.results["picp"] = np.mean(compute_picp(self.truth_values.reshape(self.settings['test_days'], 24), zsquantiles.reshape(self.settings['test_days'], 24, num_quantiles), self.lb, self.ub))
        else:
            print(self.seed_datasets.items())
            for k,v in self.seed_datasets.items():
                myquantiles = np.quantile(v, self.quantiles, axis=1).T
                num_quantiles = 9 if self.is_de else 99
                median_idx = num_quantiles // 2  # Use middle quantile as median
                self.results['each_seed_results'][f"seed_{k}"] = {
                        "mae" : mean_absolute_error(self.truth_values, myquantiles[:, median_idx]),
                        "rmse" : np.sqrt(mean_squared_error(self.truth_values, myquantiles[:, median_idx])),
                        "crps" : np.mean(compute_crps(self.truth_values.reshape(self.settings['test_days'], 24), myquantiles.reshape(self.settings['test_days'], 24, num_quantiles), self.quantiles)),
                        "picp" : np.mean(compute_picp(self.truth_values.reshape(self.settings['test_days'], 24), myquantiles.reshape(self.settings['test_days'], 24, num_quantiles), self.lb, self.ub)),
                    }
        print("compute_single_results function completed")


    def compute_probabilistic_aggregation(self):
        myquantiles = self.get_prob_aggregation_quantiles()
        num_quantiles = 9 if self.is_de else 99
        median_idx = num_quantiles // 2  # Use middle quantile as median
        self.results["mae_probabilistic_agg"] = mean_absolute_error(self.truth_values, myquantiles[:, median_idx])
        self.results["rmse_probabilistic_agg"] = np.sqrt(mean_squared_error(self.truth_values, myquantiles[:, median_idx]))
        self.results["crps_probabilistic_agg"] = np.mean(compute_crps(self.truth_values.reshape(self.settings['test_days'], 24), myquantiles.reshape(self.settings['test_days'], 24, num_quantiles), self.quantiles))
        self.results["picp_probabilistic_agg"] = np.mean(compute_picp(self.truth_values.reshape(self.settings['test_days'], 24), myquantiles.reshape(self.settings['test_days'], 24, num_quantiles), self.lb, self.ub))
        print("compute_probabilistic_aggretation function completed")


    def compute_quantile_aggregation(self):
        averaged_quantiles = self.get_quantile_aggregation_quantiles()
        num_quantiles = 9 if self.is_de else 99
        median_idx = num_quantiles // 2  # Use middle quantile as median
        self.results["mae_quantile_agg"]= mean_absolute_error(self.truth_values, averaged_quantiles[:, median_idx])
        self.results["rmse_quantile_agg"]= np.sqrt(mean_squared_error(self.truth_values, averaged_quantiles[:, median_idx]))
        self.results["crps_quantile_agg"]= np.mean(compute_crps(self.truth_values.reshape(self.settings['test_days'], 24), averaged_quantiles.reshape(self.settings['test_days'], 24, num_quantiles), self.quantiles))
        self.results["picp_quantile_agg"]= np.mean(compute_picp(self.truth_values.reshape(self.settings['test_days'], 24), averaged_quantiles.reshape(self.settings['test_days'], 24, num_quantiles), self.lb, self.ub))
        print("compute_quantile_aggretation function completed")


    def compute_and_plot_picp(self, agg='probabilistic'):
        PICP_tot = []
        alpha_list = [i/100 for i in range(2,98)] if not self.is_de else [i/10 for i in range(2,10,2)]
        print("alpha_list", alpha_list)
        for alpha in alpha_list:
            inv_alpha = (1-alpha)/2
            if self.is_de:
                # For DE dataset, always use 9 quantiles (0.1 to 0.9)
                inv_alpha = round(inv_alpha, 2)
                ub = int((1-inv_alpha)*10)-1
                lb = int((inv_alpha)*10)-1
            else:
                # For other datasets, use 99 quantiles (0.01 to 0.99)
                ub = int((1-inv_alpha)*100)-1
                lb = int((inv_alpha)*100)-1
            if "zeroshot" in self.settings['bench_folder_name']:
                zsquantiles = np.quantile(self.zeroshot_df, self.quantiles, axis=1).T
                num_quantiles = 9 if self.is_de else 99
                picp = np.mean(compute_picp(self.truth_values.reshape(self.settings['test_days'], 24), zsquantiles.reshape(self.settings['test_days'], 24, num_quantiles), lb, ub))
            elif "jsu_dnn" in self.settings['bench_folder_name']:
                zsquantiles = self.zeroshot_df
                num_quantiles = 9 if self.is_de else 99
                picp = np.mean(compute_picp(self.truth_values.reshape(self.settings['test_days'], 24), zsquantiles.reshape(self.settings['test_days'], 24, num_quantiles), lb, ub))
            else:
                if agg == 'probabilistic':
                    averaged_quantiles = self.get_prob_aggregation_quantiles()
                elif agg == 'quantile':
                    averaged_quantiles = self.get_quantile_aggregation_quantiles()
                num_quantiles = 9 if self.is_de else 99
                picp = np.mean(compute_picp(self.truth_values.reshape(self.settings['test_days'], 24), averaged_quantiles.reshape(self.settings['test_days'], 24, num_quantiles), lb, ub))
            PICP_tot.append(picp/100)

        print("PICP_tot", PICP_tot)
        
        if self.is_de:
            # Save PICP data to text file in table format
            self._save_picp_table(alpha_list, PICP_tot)
        else:
            # For non-DE datasets, create the plot as before
            plt.figure(figsize=(10, 10))
            plt.plot(alpha_list, PICP_tot, 'b-', label='PICP')
            plt.plot(alpha_list, alpha_list, 'r--', label='Ideal calibration')
            plt.xlabel('α', fontsize=35)
            plt.ylabel('Coverage', fontsize=35)
            plt.title(self.model_names_coded[self.settings['bench_folder_name']] + ("" if "fft" not in self.settings["bench_folder_name"] else f"-p" if agg == 'probabilistic' else f"-v"), fontsize=35)
            plt.grid(True)
            plt.legend(fontsize=30)
            plt.tick_params(axis='both', which='major', labelsize=30)
            plt.savefig(f'./final_results/4_markets/picp_calibration_{self.settings["bench_folder_name"]}{"" if "fft" not in self.settings["bench_folder_name"] else f"_p" if agg == 'probabilistic' else f"_v"}_{self.dataset_name}.png')
            plt.close()
        print("compute_and_plot_picp function completed")
    
    
    def _save_picp_table(self, alpha_list, picp_values):
        """
        Save PICP values in a table format for DE dataset.
        Creates/updates a table with columns: DE, small, base, large, jsudnn
        """
        # Determine model type from bench_folder_name
        bench_name = self.settings['bench_folder_name']
        if 'small' in bench_name:
            model_col = 'small'
        elif 'base' in bench_name:
            model_col = 'base'
        elif 'large' in bench_name:
            model_col = 'large'
        elif 'jsu' in bench_name.lower():
            model_col = 'jsudnn'
        else:
            model_col = 'unknown'
        
        # File path for the table
        table_file = f'./final_results/DE/picp_table_{self.dataset_name}.txt'
        
        # Load existing data if file exists
        data_dict = {}
        if os.path.exists(table_file):
            with open(table_file, 'r') as f:
                lines = f.readlines()
            
            # Parse existing data - skip header (line 0) and separator (line 1)
            for line in lines[2:]:
                if line.strip() and not line.startswith('-'):
                    parts = line.strip().split()
                    if len(parts) >= 5:  # alpha_label + 4 model columns
                        alpha_label = parts[0]
                        data_dict[alpha_label] = {
                            'small': parts[1] if parts[1] != '-' else None,
                            'base': parts[2] if parts[2] != '-' else None,
                            'large': parts[3] if parts[3] != '-' else None,
                            'jsudnn': parts[4] if parts[4] != '-' else None
                        }
        
        # Initialize empty data for missing alphas
        for alpha in alpha_list:
            alpha_label = f'PICP_α={alpha:.1f}'
            if alpha_label not in data_dict:
                data_dict[alpha_label] = {}
        
        # Add current model's data
        for i, alpha in enumerate(alpha_list):
            alpha_label = f'PICP_α={alpha:.1f}'
            if alpha_label not in data_dict:
                data_dict[alpha_label] = {}
            data_dict[alpha_label][model_col] = f'{picp_values[i]:.3f}'
        
        # Write table to file
        with open(table_file, 'w') as f:
            # Header
            f.write(f"{'':15s} {'small':8s} {'base':8s} {'large':8s} {'jsudnn':8s}\n")
            f.write("-" * 55 + "\n")
            
            # Data rows
            for alpha in alpha_list:
                alpha_label = f'PICP_α={alpha:.1f}'
                row_data = data_dict.get(alpha_label, {})
                # Handle None values by converting to '-'
                small_val = row_data.get('small', '-') or '-'
                base_val = row_data.get('base', '-') or '-'
                large_val = row_data.get('large', '-') or '-'
                jsudnn_val = row_data.get('jsudnn', '-') or '-'
                f.write(f"{alpha_label:15s} ")
                f.write(f"{small_val:8s} ")
                f.write(f"{base_val:8s} ")
                f.write(f"{large_val:8s} ")
                f.write(f"{jsudnn_val:8s}\n")
        
        print(f"PICP table saved to {table_file}")



    def _load_quantiles(self, model_name):
        """
        Load quantiles for each model (for DM TEST)
        """
        each_dataset_model = {} 
        if "zeroshot" in model_name:
            path = f'./benchmark_results/{model_name}/samples_{self.dataset_name}/'
            for file in os.listdir(path):
                df_model = pd.read_csv(f'{path}{self.dataset_name}_samples_seed_20.csv').iloc[:,1:]
                df_model = df_model.values
                df_model = np.quantile(df_model, self.quantiles, axis=1).T 
                print("_load_quantiles function completed")
                return df_model
        elif "jsu_dnn" in model_name:
            path = f'./benchmark_results/DNN/' if not self.is_de else f'./benchmark_results/DE/'
            with open(f'{path}{self.dataset_name}_JSU-DNN.p', 'rb') as f:
                df_model = pickle.load(f)
            df_model = df_model['results'].iloc[:,1:].values if not self.is_de else df_model['results']['JSU-DNN'].iloc[:,1:].values
            print("_load_quantiles function completed")
            return df_model
        else:
            path = f'./benchmark_results/{model_name[:-2]}_old/' if "small" in model_name else f'./benchmark_results/{model_name[:-2]}/'
            for file in os.listdir(path):
                seed = file.split('_')[-1] 
                df_model = pd.read_csv(f'{path}{file}/samples_{self.dataset_name}/{self.dataset_name}_samples_seed_{seed}.csv')
                each_dataset_model[seed] = df_model.iloc[:, 1:]

            if model_name[-2:] == '_p':  # probabilistic aggregation
                # Stack all the predictions for each seed and compute the quantiles
                all_seeds_predictions = None
                for seed in [seed for seed in each_dataset_model.keys() if seed != "truth"]:
                    seed_predictions = each_dataset_model[seed]
                    if all_seeds_predictions is None:
                        all_seeds_predictions = seed_predictions
                    else:
                        all_seeds_predictions = np.hstack((all_seeds_predictions, seed_predictions))
                df_model = np.quantile(all_seeds_predictions, self.quantiles, axis=1).T
                print("_load_quantiles function completed")
                return df_model
            elif model_name[-2:] == '_v':  # quantile aggregation
                # Compute the quantiles for each seed and then average them
                seed_quantiles = []
                for seed in [seed for seed in each_dataset_model.keys() if seed != "truth"]:
                    seed_predictions = each_dataset_model[seed].values
                    seed_quantile = np.quantile(seed_predictions, self.quantiles, axis=1).T
                    seed_quantiles.append(seed_quantile)
                seed_quantiles = np.array(seed_quantiles)
                df_model = np.mean(seed_quantiles, axis=0)
                print("_load_quantiles function completed")
                return df_model


    def DM_test_mae_crps(self, models_DM, suffix=None, nocov=False):
        models_data = {}
        for model in models_DM:
            models_data[model] = self._load_quantiles(model)


        # Initialize matrices for p-values
        n_models = len(models_DM)
        dm_mae_matrix = np.ones((n_models, n_models))
        dm_crps_matrix = np.ones((n_models, n_models))

        # Compute DM test for all pairs
        for i, model1 in enumerate(models_DM):
            for j, model2 in enumerate(models_DM):
                if i != j:  # Skip diagonal (same model)
                    dfmodel1 = models_data[model1]
                    dfmodel2 = models_data[model2]

                    print(model1, model2)
                    print(dfmodel1.shape, dfmodel2.shape)
                    print(self.truth_values.shape)
                    
                    median_idx = 4 if self.is_de else 49
                    mae1 = np.abs(self.truth_values - dfmodel1[:, median_idx])
                    mae2 = np.abs(self.truth_values - dfmodel2[:, median_idx])

                    num_quantiles = 9 if self.is_de else 99
                    crps1 = compute_crps(self.truth_values.reshape(self.settings['test_days'], 24), dfmodel1.reshape(self.settings['test_days'], 24, num_quantiles), self.quantiles)
                    crps2 = compute_crps(self.truth_values.reshape(self.settings['test_days'], 24), dfmodel2.reshape(self.settings['test_days'], 24, num_quantiles), self.quantiles)

                    # Convert to numpy arrays if they are lists
                    crps1 = np.array(crps1)
                    crps2 = np.array(crps2)

                    dm_mae_result = DM(self.truth_values.reshape(self.settings['test_days'], 24), mae1.reshape(self.settings['test_days'], 24), mae2.reshape(self.settings['test_days'], 24))
                    dm_crps_result = DM(self.truth_values.reshape(self.settings['test_days'], 24), crps1, crps2)

                    # Store average p-value across hours
                    dm_mae_matrix[i, j] = np.mean(dm_mae_result)
                    dm_crps_matrix[i, j] = np.mean(dm_crps_result)


        print(f"DM MAE Matrix for {self.dataset_name}:")
        print(dm_mae_matrix)
        print(f"DM CRPS Matrix for {self.dataset_name}:")
        print(dm_crps_matrix)

        # color map
        red = np.concatenate([np.linspace(0, 1, 50), np.linspace(1, 0.5, 50)[1:], [0]])
        green = np.concatenate([np.linspace(0.5, 1, 50), np.zeros(50)])
        blue = np.zeros(100)
        rgb_color_map = np.concatenate([red.reshape(-1, 1), green.reshape(-1, 1),
                                        blue.reshape(-1, 1)], axis=1)
        rgb_color_map = mpl.colors.ListedColormap(rgb_color_map)

        # Create mask for diagonal elements (same model comparisons)
        mask = np.eye(n_models, dtype=bool)
        
        # Plot DM test results matrices
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        # MAE DM Test Matrix
        dm_mae_masked = np.ma.masked_where(mask, dm_mae_matrix)
        im1 = ax1.imshow(dm_mae_masked, cmap=rgb_color_map, vmin=0, vmax=0.1)
        ax1.set_xticks(range(n_models))
        ax1.set_xticklabels([self.model_names_coded[model] for model in models_DM], rotation=45, fontsize=30)
        ax1.set_yticks(range(n_models))
        ax1.set_yticklabels([self.model_names_coded[model] for model in models_DM], fontsize=30)
        ax1.set_title(f"{self.dataset_name if not self.is_de else 'DE20'} - MAE", fontsize=35)

        plt.colorbar(im1, ax=ax1, shrink=0.8)

        # CRPS DM Test Matrix
        dm_crps_masked = np.ma.masked_where(mask, dm_crps_matrix)
        im2 = ax2.imshow(dm_crps_masked, cmap=rgb_color_map, vmin=0, vmax=0.1)
        ax2.set_xticks(range(n_models))
        ax2.set_xticklabels([self.model_names_coded[model] for model in models_DM], rotation=45, fontsize=30)
        ax2.set_yticks(range(n_models))
        ax2.set_yticklabels([self.model_names_coded[model] for model in models_DM], fontsize=30)
        ax2.set_title(f"{self.dataset_name if not self.is_de else 'DE20'} - CRPS", fontsize=35)

        plt.colorbar(im2, ax=ax2, shrink=0.8)
        
        plt.tight_layout()
        if suffix != None:
            plt.savefig(f'./final_results/dm_test_matrix_4markets_{self.dataset_name}_compare_context_{suffix}.png', dpi=300, bbox_inches='tight')
        else:
            if self.is_de:
                plt.savefig(f'./final_results/dm_test_matrix_DE_{self.dataset_name}.png', dpi=300, bbox_inches='tight')
            elif nocov:
                plt.savefig(f'./final_results/dm_test_matrix_4markets_{self.dataset_name}_nocov.png', dpi=300, bbox_inches='tight')
            else:
                plt.savefig(f'./final_results/dm_test_matrix_4markets_{self.dataset_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("DM_test_mae_crps function completed")


    def _convert_numpy_types(self, obj):
        """Recursively convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj

    def save_results(self, compare_context=False, nocov=False):
        folder = None
        if nocov:
            folder = "4_markets_nocov"
        elif compare_context:
            folder = "4_markets_compare_context"
        elif self.is_de:
            folder = "DE"
        else:
            folder = "4_markets"

        results_filename = f"./final_results/{folder}/{self.settings['bench_folder_name']}_results.txt"
        # Convert numpy types to native Python types
        serializable_results = self._convert_numpy_types(self.results)
        results_str = json.dumps(serializable_results, indent=4)
        with open(results_filename, 'a') as f:
            f.write(f"\nDataset results:\n{results_str}\n")
            f.write("-" * 80 + "\n")  # Add separator between entries
        print("save_results function completed")




