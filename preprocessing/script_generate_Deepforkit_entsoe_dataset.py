from data.deepforkit_datasets_utils import DeepforkitDatasetBuilder
from data.entsoe_datasets_utils import EntsoeDatasetBuilder

data_configs={
    'BE': {
        'dataset_series': {'price': 'target',
                           'load_f': 'futu',
                           'solar_f': 'futu',
                           'wind_f': 'futu'},
        'start_date': '2018-12-25',
        'indexes_to_fix': [
                {'column': 'TARG__target',
                 'date': '2019-06-8',
                 'hour': ':',
                 'lags': 168},
                {'column': 'FUTU__load_f',
                 'date': '2023-06-22',
                 'hour': ':',
                 'lags': 168}
        ],
    },
    'DE': {
        'dataset_series': {'price': 'target',
                           'load_f': 'futu',
                           'solar_f': 'futu',
                           'wind_f': 'futu'},
        'start_date': '2018-12-31',
        'indexes_to_fix': [
                {'column': 'FUTU__load_f',
                 'date': '2022-02-22',
                 'hour': ':',
                 'lags': 168},
                {'column': 'FUTU__load_f',
                 'date': '2022-03-24',
                 'hour': ':',
                 'lags': 168},
        ],
    },
    'ES': {
        'dataset_series': {'price': 'target',
                           'load_f': 'futu',
                           'solar_f': 'futu',
                           'wind_f': 'futu'},
        'start_date': '2018-12-25',
        'indexes_to_fix': [
            {'column': 'FUTU__load_f',
             'date': '2022-10-30',
             'hour': 2,
             'lags': 168},
        ],
    },
    'SE_3': {
        'dataset_series': {'price': 'target',
                           'load_f': 'futu',
                           #'solar_f': 'futu', #excluded since 0 before 2022
                           'wind_f': 'futu'},
        'start_date': '2018-12-25',
        'indexes_to_fix': [
        ],
    },
}


#----------------------------------------------------------------------
# Script to generate dataset from entso-e files
#----------------------------------------------------------------------
markets = ['BE','DE','ES','SE_3']#['BE', 'DE', 'ES', 'SE_3']
end_date = '2023-6-30'
print_series = False
pred_horiz = 24
past_days_context = 7
mode='standard_scaler'
past_scaler_end='2023-10-1'

start_date_entsoe = '2023-07-01'
end_date_entsoe = '2024-9-30'

for market in markets:
    #----------------------------------------------------------------------------------
    data = DeepforkitDatasetBuilder(market=market,
                                    series_to_include=data_configs[market]['dataset_series'],
                                    starting_date=data_configs[market]['start_date'], end_date=end_date)

    #----------------------------------------------------------------------------------
    # Plot dataset to investigate erroneous samples
    if print_series:
        data.plot_dataset(column='FUTU__solar_f')
        data.plot_dataset(column='TARG__target')
        data.plot_dataset(column='FUTU__load_f')
        data.plot_dataset(column='FUTU__wind_f')

    # Fix erroneous samples
    data.fix_erroneous_samples(indexes=data_configs[market]['indexes_to_fix'], print_series=print_series)


    # Load entso-e data
    entsoe_data = EntsoeDatasetBuilder(market=market,
                                       series_to_include=data_configs[market]['dataset_series'],
                                       starting_date=start_date_entsoe, end_date=end_date_entsoe)

    #outliers = entsoe_data.check_outliers()
    data.append_samples(entsoe_data.dataset, end_date_entsoe)

    if print_series:
        data.plot_dataset(column='FUTU__solar_f')
        data.plot_dataset(column='TARG__target')
        data.plot_dataset(column='FUTU__load_f')
        data.plot_dataset(column='FUTU__wind_f')
    #---------------------------------------------------------------------------------
    # store name of columns to scale before insert the other columns
    columns_to_scale = data.dataset.columns.tolist()
    #----------------------------------------------------------------------------------
    # include additional features
    data.add_dayofweek_cycvariable()
    data.add_month_cycvariable()
    data.add_dayofyear_cycvariable()
    data.add_holidays()
    data.add_ts_age()

    #----------------------------------------------------------------------------------
    # Scale dataset
    context_length = pred_horiz * past_days_context
    data.scale_dataset(prediction_length=pred_horiz,
                       context_length=context_length,
                       col_to_transf=columns_to_scale,
                       mode=mode,
                       past_scaler_end=past_scaler_end)

    #----------------------------------------------------------------------------------
    # save dataset to pickle
    data.save_dataset()
    print('Done data!')
