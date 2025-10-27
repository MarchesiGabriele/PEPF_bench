import os
import numpy as np
import pandas
import pandas as pd
import sys
from datetime import datetime
from typing import Dict, List
import matplotlib.pyplot as plt
from tools.data_utils import features_keys, get_dataset_save_path
import holidays
import pickle
from data.holidays_feat import smooth_holiday


deepforkit_series = {"Price_DA": "price",
                     "Load_AC": 'load_a',
                     "Load_DA": 'load_f',
                     "Sol_DA": 'solar_f',
                     "Won_DA": 'wind_f',
                     "Woff_DA": "windoff_f"
}

class DeepforkitDatasetBuilder:
    def __init__(self, market: str,
                 series_to_include: Dict,
                 starting_date: str, end_date: str, num_daily_samples: int=24,
                 check_missing_data: bool=True, fix_nan: bool = True, interpolation_limit: int=1):

        self.market = market
        self.starting_date = starting_date
        self.end_date = end_date
        self.num_daily_samples = num_daily_samples
        self.check_missing_data = check_missing_data
        self.fix_nan = fix_nan
        self.interpolation_limit = interpolation_limit

        self.dataset = self.__build_data__(series_to_include=series_to_include)

    def __get_data_path__(self):
        return os.path.join(os.getcwd(), 'data', 'raw', 'deepforkit')

    def __get_csv_path__(self, market, starting_year, end_year):
        return str(market) + '_market__' + str(starting_year) + '__' + str(end_year) + '.csv'

    def __get_market_var_name__(self, var_type: str, var: str):
        if var_type == 'target':
            vn = features_keys[var_type] + 'target'
        else:
            vn=features_keys[var_type] + var
        return vn

    def __build_data__(self, series_to_include: Dict):
        filetype = '.csv'
        path = os.path.join(self.__get_data_path__(), self.market + filetype)
        df = pd.read_csv(path)
        df['datetime'] = pd.to_datetime(df['Unnamed: 0'])
        df.set_index('datetime', inplace=True)
        # Drop the original date_column and date_extracted
        df.drop(columns=['Unnamed: 0'], inplace=True)
        df.rename(columns=deepforkit_series, inplace=True)
        df_market = df[list(series_to_include)]
        df_market = df_market.astype(
            {col: 'float32' for col in df_market.select_dtypes(include=['float64']).columns})
        # set the column name in the new dataset format
        for var_name in list(df_market.columns):
            df_market.rename(columns={var_name: self.__get_market_var_name__(series_to_include[var_name], var_name)},
                      inplace=True)
        # Set starting date
        return df_market[self.starting_date:]

    def add_dayofweek_cycvariable(self, decimals=6):
        df_wd = self.dataset.index.dayofweek
        self.dataset[features_keys['const'] + 'wd_sin'] = np.around(np.sin(df_wd * (2. * np.pi / 7)), decimals=decimals)
        self.dataset[features_keys['const'] +'wd_cos'] = np.around(np.cos(df_wd * (2. * np.pi / 7)), decimals=decimals)

    def add_month_cycvariable(self, decimals=6):
        df_mnth = self.dataset.index.month
        self.dataset[features_keys['const'] + 'mnth_sin'] = np.around(np.sin(df_mnth * (2. * np.pi / 12)), decimals=decimals)
        self.dataset[features_keys['const'] + 'mnth_cos'] = np.around(np.cos(df_mnth * (2. * np.pi / 12)), decimals=decimals)

    def add_dayofyear_cycvariable(self, decimals=6):
        df_yd = self.dataset.index.day_of_year
        self.dataset[features_keys['const'] + 'yd_sin'] = np.around(np.sin(df_yd * (2. * np.pi / 365)), decimals=decimals)
        self.dataset[features_keys['const'] + 'yd_cos'] = np.around(np.cos(df_yd * (2. * np.pi / 365)), decimals=decimals)

    def add_ts_age(self):
        self.dataset[features_keys['const'] + 'ts_age'] = np.log(1 + np.arange(stop=len(self.dataset)) // self.num_daily_samples)

    def add_holidays(self):
        def is_holiday(date):
            return date in it_holidays
        it_holidays = holidays.country_holidays('IT')
        self.dataset['Date'] = self.dataset.index.date
        self.dataset[features_keys['const'] + 'is_holiday'] = self.dataset['Date'].apply(is_holiday)
        hol_d = smooth_holiday(self.dataset[features_keys['const'] + 'is_holiday'].astype("float64").to_numpy().reshape(-1, 24)[:,0])
        self.dataset[features_keys['const'] + 'is_holiday_smooth'] = np.tile(hol_d[:, np.newaxis], (1, 24)).flatten()

    def scale_dataset(self, prediction_length, context_length, col_to_transf,
                      mode='context_scaler', past_scaler_end=None):
        time_series = {}
        self.mode = mode
        day_lag = int(context_length / prediction_length)
        for d in range(int(len(self.dataset)/prediction_length)-day_lag):
            start = d*prediction_length
            end = start + context_length + prediction_length
            data_d = self.dataset.iloc[start:end]
            for col in col_to_transf:
                mu_name = features_keys['const'] + col + '_trasf_p0'
                std_name = features_keys['const'] + col + '_trasf_p1'
                col_sc_n = col + '_scaled'
                x = data_d[col].to_numpy()
                x_h = x.reshape(-1, prediction_length)[:-1].flatten()
                eps=1e-5 # apply correction as in RevIN for numerical purpose (in case of small std in the features)
                if mode == 'context_scaler':
                    mu = np.mean(x_h)
                    scale = np.std(x_h) + eps
                elif mode == 'standard_scaler':
                    scaler_context = self.dataset.loc[:past_scaler_end]
                    col_val = scaler_context[col]
                    mu = col_val.mean()
                    scale = col_val.std() + eps
                elif mode == 'unscaled':
                    mu = 0
                    scale = 1
                else:
                    sys.exit('uknown mode!')

                data_d.insert(loc=len(data_d.columns), column=mu_name, value=np.repeat(mu, context_length + prediction_length))
                data_d.insert(loc=len(data_d.columns), column=std_name,
                              value=np.repeat(scale, context_length + prediction_length))
                data_d.insert(loc=len(data_d.columns), column=col_sc_n,
                              value=(data_d.loc[:,col]-data_d.loc[:,mu_name])/data_d.loc[:,std_name])

            time_series[data_d.index[-1].date()]=data_d

        self.dataset = time_series


    def plot_dataset(self, column=None):
        if column == None:
            plt.plot(self.dataset)
            plt.grid()
            plt.show()
        else:
            self.dataset[column].plot()
            plt.ylabel(column)
            plt.grid()
            plt.show()

    def save_dataset(self):
        dir = 'DFK_' + str(self.market) + '_market__' + str(self.starting_date) + '__' + str(self.end_date)
        file_dir = os.path.join(get_dataset_save_path(), dir)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        file_path = os.path.join(file_dir, self.mode + '.p')
        with open(file_path, 'wb') as f:
            pickle.dump(self.dataset, f)

    def fix_erroneous_samples(self, indexes, print_series=False):
        for index in indexes:
            replace_date = index['date']
            lag=index['lags']
            column = index['column']

            if print_series:
                self.dataset[column].plot()
                plt.ylabel(column)
                plt.grid()
                plt.show()

            if index['hour']==':':
                replace_hours = list(range(0, 24))
            else:
                replace_hours = [index['hour']]
            for replace_hour in replace_hours:
                self.dataset.loc[(self.dataset.index.date == pd.to_datetime(replace_date).date())
                                 & (self.dataset.index.hour == replace_hour), column] = \
                    self.dataset.loc[(self.dataset.index.date == pd.to_datetime(replace_date).date() - pd.Timedelta(hours=lag)) & (
                                self.dataset.index.hour == replace_hour), column].values

            if print_series:
                self.dataset[column].plot()
                plt.ylabel(column)
                plt.grid()
                plt.show()

    def append_samples(self, data, end_date):
        self.dataset = pd.concat([self.dataset, data])
        self.end_date = end_date