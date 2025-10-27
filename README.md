# Assessing time series foundation models for probabilistic electricity price forecasting: toward a unified benchmark

## Data collection and pre-processing

The data for the 4 markets (BE, DE, ES, and SE_3) was taken from https://github.com/Kasra-Aliyon/Deepforkit/tree/main/data, extended with data from the open ENTSO-E transparency platform.

**Setup:**

1.  Download the  BE, DE, ES, and SE_3 `.csv` files from the DeepForKit repository and place them inside the `./preprocessing/data/raw/deepforkit` folder.
2.  For each region, download the yearly ENTSO-E data for "Total load - Day-ahead/Actual", "Energy Prices", and "Generation Forecast for Wind & Solar" for 2023 and 2024. Rename each file `<year>.csv` (e.g., `2023.csv`) and place it in the corresponding subfolder (`load`, `price`, or `wind_solar`) for that market inside `./preprocessing/data/raw/entso_e`.
3.  Run the `preprocessing/script_generate_Deepforkit_entsoe_dataset.py` script.

---

To prepare the data for the DE20 experiments, download `DE.csv` from https://github.com/gmarcjasz/distributionalnn/blob/main/Datasets/DE.csv and place it in the `preprocessing/data` folder.

---

## Benchmark execution

* `start_benchmark.py`: Runs the benchmarks for the 4 markets (BE, DE, ES, SE).
* `start_benchmark_DE.py`: Runs the benchmark for DE20.

Use the settings `include_covariates` and `finetune` to decide whether to include covariates or start the finetuning process.

Once the results are computed and located in the `benchmark_results` folder, you can use the `compute_final_*.py` scripts to compute the PICP plots, DM test, MAE, and CRPS computations. The results should appear in a folder named `final_results`.

# Cite
