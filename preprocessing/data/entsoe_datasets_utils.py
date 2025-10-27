import os
import numpy as np
import pandas
import pandas as pd
import sys
from datetime import datetime, timedelta
from typing import Dict, List
import matplotlib.pyplot as plt
from tools.data_utils import features_keys, get_dataset_save_path
import holidays
import pickle
from data.holidays_feat import smooth_holiday

# Class used to structure the strings related to each entso-e csv
class _EntsoeDataConfs:
    def __init__(self, folder, column_key, date_column):
        self.folder = folder
        self.column_key = column_key
        self.date_column = date_column

# energy price, load, load-forecast (available on entso-e/nordpool before day ahead market closure)
entsoe_series = {'price':  _EntsoeDataConfs(folder='price',
                                           column_key='Day-ahead Price',
                                           date_column='MTU (CET/CEST)'),

                 'load_a': _EntsoeDataConfs(folder='load',
                                           column_key='Actual Total Load',
                                           date_column='Time (CET/CEST)'),

                 'load_f': _EntsoeDataConfs(folder='load',
                                            column_key='Day-ahead Total Load Forecast',
                                            date_column='Time (CET/CEST)'),

                 'solar_f': _EntsoeDataConfs(folder='wind_solar',
                                            column_key='Generation - Solar [MW] Day Ahead',
                                            date_column='MTU (CET/CEST)'),

                 'wind_f': _EntsoeDataConfs(folder='wind_solar',
                                             column_key='Generation - Wind Onshore [MW] Day Ahead',
                                             date_column='MTU (CET/CEST)')
}

class EntsoeDatasetBuilder:
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

        self.dataset = self.__fix_dataset__(self.__build_entso_e_data__(series_to_include=series_to_include))

    def __get_entsoe_data_path__(self):
        return os.path.join(os.getcwd(), 'data', 'raw', 'entso_e')

    def __get_entso_e_csv_path__(self, market, starting_year, end_year):
        return str(market) + '_market__' + str(starting_year) + '__' + str(end_year) + '.csv'

    def __get_market_var_name__(self, var_type: str, market: str, var: str):
        if var_type == 'target':
            vn = features_keys[var_type] + 'target'
        else:
            vn=features_keys[var_type] + var
        return vn

    def __build_entso_e_data__(self, series_to_include: Dict):
        filetype = '.csv'
        d_par = lambda x: datetime.strptime(x[:16], '%d.%m.%Y %H:%M')
        # -------------------------------------------------------------------------------
        #  Load and aggregate csv data in pandas
        # -------------------------------------------------------------------------------
        df_vars = []
        # load the entso-e csv file for each variable to be included in the dataset
        for var_name in series_to_include.keys():
            df_var_lis = []
            var = entsoe_series[var_name]
            # load the file for each year involved in the dataset
            for year in [*range(int(self.starting_date[:4]), int(self.end_date[:4])+1, 1)]:
                path = os.path.join(self.__get_entsoe_data_path__(), self.market, var.folder, str(year) + filetype)

                #df = pd.read_csv(path, parse_dates=[var.date_column], date_parser=d_par)
                #df.set_index(df.columns[0], inplace=True)
                df = pd.read_csv(path)
                df['date_extracted'] = df[var.date_column].str[:16]
                df['date_converted'] = pd.to_datetime(df['date_extracted'], format='%d.%m.%Y %H:%M')
                # Set date_converted as the index
                df.set_index('date_converted', inplace=True)
                # Drop the original date_column and date_extracted
                df.drop(columns=[var.date_column, 'date_extracted'], inplace=True)

                # get the column from the file
                column_id = [col for col in df.columns if var.column_key in col]
                if len(column_id) > 1:
                    print('multiple columns with consistent name...')
                    print('check data file ' + self.market + str(year))
                    sys.exit()
                else:
                    column_id = column_id[0]
                df_part = df[column_id].copy()
                # set the column name in the new dataset format
                df_part.rename(self.__get_market_var_name__(series_to_include[var_name], self.market, var_name), inplace=True)
                # Resample in case of quarterly data
                date_end = datetime.strptime(self.end_date, '%Y-%m-%d')
                date_end = date_end + timedelta(days=1)
                df_part=df_part.loc[df.index < date_end].astype('float32')
                df_part=df_part.resample('h').mean()
                df_var_lis.append(df_part)
            df_y = pd.concat(df_var_lis, axis=0)
            df_vars.append(df_y)

        df_market = pd.concat(df_vars, axis=1)

        # Set starting date
        return df_market[self.starting_date:]

    def __fix_dataset__(self, df_market):
        # Remove duplicated hours (daylight savings) for simplicity
        df_market = df_market[~df_market.index.duplicated(keep='first')]
        if len(df_market[df_market.index.duplicated(keep=True)]) > 0:
            exit()

        # Check and fix nan
        # Filled by day-1 values
        if self.fix_nan:
            co=0
            for idx, day_samples in df_market.groupby(df_market.index.date):
                # Check eventual missing hours
                if len(day_samples) < self.num_daily_samples:
                    print(str(idx) + '  --> missing hour')
                    exit()
                for col_id, ds_column in day_samples.items():
                    if (ds_column.isna().sum()) > 1:
                        null_items_dt = ds_column[ds_column.isnull()].index.tolist()
                        for ni_dt in null_items_dt:
                            df_market[col_id][ni_dt] = df_market[col_id][ni_dt - pd.DateOffset(days=1)]
                            print('fixed_nan (using d-1 value) for ' + col_id + ' in dt: ' + str(ni_dt))
                            co=co+1
            print(str(co))

        # Interpolate missing samples using given limit
        df_market.interpolate(limit=self.interpolation_limit, inplace=True)

        # Count eventual missing samples
        if self.check_missing_data:
            df_market.info()
            rows_with_nan = []
            for index, row in df_market.iterrows():
                is_nan_series = row.isnull()
                if is_nan_series.any():
                    rows_with_nan.append(index)
            print(rows_with_nan)

        # Convert to float32
        df_market = df_market.astype(
            {col: 'float32' for col in df_market.select_dtypes(include=['float64']).columns})
        return df_market

    # check outliers using IQR
    def check_outliers(self):
        outliers={}
        for col in self.dataset.columns:
            Q1 = self.dataset[col].quantile(0.25)
            Q3 = self.dataset[col].quantile(0.75)
            IQR = Q3 - Q1

            # identify outliers
            threshold = 1.5
            outliers[col] = self.dataset[(self.dataset[col] < Q1 - threshold * IQR) | (self.dataset[col] > Q3 + threshold * IQR)][col]
            print('--------------------------------------------------------')
            print('outliers in series: ' + col)
            print('--------------------------------------------------------')
            print(outliers[col])

        return outliers

    def remove_outliers(self, outliers: pandas.DataFrame, column: str, rows: [], method='interpol'):
        outliers_to_remove = np.array(outliers[column].index.tolist())
        outl = list(outliers_to_remove[rows])
        for row in outl:
            self.dataset.at[row, column] = np.nan
        if method == 'interpol':
            self.dataset.interpolate(limit=1, inplace=True)
        else:
            sys.exit('ERROR: unsupported outlier removal method')

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

                if mode == 'context_scaler':
                    if col[:4] == 'TARG':
                        mu = np.mean(x_h)
                    else:
                        mu = np.mean(x_h)
                    scale = np.std(x_h)

                elif mode == 'standard_scaler':
                    scaler_context = self.dataset.loc[:past_scaler_end]
                    col_val = scaler_context[col]
                    if col[:4] == 'TARG':
                        mu = col_val.mean()#0
                    else:
                        mu = col_val.mean()
                    scale = col_val.std()
                else:
                    sys.exit('uknown mode!')

                data_d.insert(loc=len(data_d.columns), column=mu_name,
                              value=np.repeat(mu, context_length + prediction_length))
                data_d.insert(loc=len(data_d.columns), column=std_name,
                              value=np.repeat(scale, context_length + prediction_length))
                data_d.insert(loc=len(data_d.columns), column=col_sc_n,
                              value=(data_d.loc[:,col]-data_d.loc[:,mu_name])/data_d.loc[:,std_name])
            time_series[data_d.index[-1].date()]=data_d

        self.dataset = time_series



    def prescale_feat(self, date_last, feat_to_prescale):
        for feat in feat_to_prescale:
            feat_val = self.dataset.loc[:date_last][feat]
            mu = feat_val.mean()
            std = feat_val.std()
            self.dataset[feat]=(self.dataset[feat]-mu)/std

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

    def __get_entsoe_pickle_path__(self, market, starting_year, end_year):
        return str(market) + '_market__' + str(starting_year) + '__' + str(end_year) + '-' + self.mode + '.p'

    def save_dataset(self):
        # save dataset as pickle
        file_path = os.path.join(get_dataset_save_path(),
                                 self.__get_entsoe_pickle_path__(self.market, self.starting_date, self.end_date))

        with open(file_path, 'wb') as f:
            pickle.dump(self.dataset, f)