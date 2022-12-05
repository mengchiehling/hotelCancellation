import argparse
import os
from functools import partial
from typing import Optional, List
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

from src.io.path_definition import get_file, get_project_dir, _load_yaml
from src.io.load_data import data_preparation
from src.io.load_parameters import optimized_parameters
from train.logic.training_process import training_process, training_process_opt
from train.logic.optimization_process import optimization_process
from train.logic.model_selection import model_training_pipeline
from train.src import config


hotel_info = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_hotel_info.csv')))

cancel_target = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_target.csv')))
date_feature = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_date_feature.csv')))
hotel_meta = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_hotel_info.csv')), index_col=0)

covid_data = pd.read_excel(get_file(os.path.join('data', 'owid-covid-data.xlsx')))

twn_covid_data = covid_data[covid_data['iso_code'] == 'TWN']
twn_covid_data['date'] = twn_covid_data['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').strftime("%Y/%m/%d"))
twn_covid_data.set_index('date', inplace=True)


# le = LabelEncoder()
# date_feature['weekdate(星期，數值型)'] = le.fit_transform(date_feature['weekdate(星期，數值型)'])


def labelencoding(df, column: str):

    le = LabelEncoder()
    df.loc[:, column] = le.fit_transform(df[column])

    return df


#def data_preparation(hotel_id: int, date_feature: pd.DataFrame, cancel_target: pd.DataFrame
                     # , smooth:bool=False, diff: Optional[List[int]]=None
                     #):

    #column = f"hotel_{hotel_id}_canceled"

    #cancel_target['date'] = cancel_target['date'].apply(lambda x: datetime.strptime(x, '%Y/%m/%d').strftime("%Y/%m/%d"))
    #cancel_target.set_index('date', inplace=True)

    #hotel_cancel = cancel_target[column].replace("-", np.nan).dropna().astype(int)

    #date_feature['date'] = date_feature['date'].apply(lambda x: datetime.strptime(x, '%Y/%m/%d').strftime("%Y/%m/%d"))
    #date_feature.set_index('date', inplace=True)
    #date_feature = date_feature.loc[hotel_cancel.index]

    #date_feature['canceled'] = hotel_cancel   # 原始值
    #date_feature['canceled_label'] = hotel_cancel  # hotel_cancel_smooth.mean()

    #twn_covid_data = covid_data[covid_data['iso_code'] == 'TWN']
    #twn_covid_data['date'] = twn_covid_data['date'].apply(
        #lambda x: datetime.strptime(x, '%Y-%m-%d').strftime("%Y/%m/%d"))
    #twn_covid_data.set_index('date', inplace=True)

    #covid_features_num = ['new_cases', 'new_deaths']  # ['new_cases', 'new_deaths']
    #covid_features_cat = []

    #date_feature = date_feature.join(twn_covid_data[covid_features_num+covid_features_cat].fillna(0))

    #num_feature_columns = ['canceled'] + covid_features_num
    #num_feature_columns = ['canceled','days2vecation','vecation_days','Precp','PrecpHour','SunShine','Temperature'] + covid_features_num

    #if smooth:
        # Smoothed features for input

        #for window in [3, 14]:
            #c  = f'canceled_{window}_roll'
            #roll = hotel_cancel.rolling(window=window, min_periods=1)
            #date_feature[c] = roll.mean()
            #num_feature_columns.extend([c])
        #date_feature['MACD'] = date_feature[f'canceled_3_roll'] - date_feature[f'canceled_14_roll']
        #num_feature_columns.append('MACD')

    #if diff is not None:
        #for time_diff in diff:
            #c = f'diff_{time_diff}'
            #date_feature[c] = (date_feature['canceled'] - date_feature['canceled'].shift(time_diff)).fillna(0)
            #num_feature_columns.append(c)

    #return num_feature_columns, covid_features_cat, date_feature


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    #parser.add_argument('--input_range', type=int, help='length of input time series')
    #parser.add_argument('--prediction_time', type=int, help='length of output time series')
    parser.add_argument('--hotel_id', type=int, help='id of hotel, should exists in cancel_dataset_target.csv')
    parser.add_argument('--configuration', type=str, help='A')
    #parser.add_argument('--diff', type=int,  nargs='+', help='差分', default=[])
    #parser.add_argument('--smooth', action='store_true')

    args = parser.parse_args()

    #input_range = args.input_range
    #prediction_time = args.prediction_time
    hotel_id = args.hotel_id
    model_type = 'LSTM2LSTM'
    config.configuration = args.configuration
    #diff = args.diff
    #smooth=args.smooth

    booking_feature = pd.read_csv(get_file(os.path.join('data', f'{hotel_id}.csv')))

    model_metadata = _load_yaml(get_file(os.path.join('config', 'training_config.yml')))

    basic_parameters = model_metadata['basic_parameters']

    n_splits = basic_parameters['n_splits']
    test_size = basic_parameters['test_size']
    max_train_size = basic_parameters['max_train_size']
    input_range = basic_parameters['input_range']
    prediction_time = basic_parameters['prediction_time']
    #n_splits = 7
    #test_size = 28

    #categorical_features = []#['vecation', 'weekdate','season','midd','sallery', 'is_rest_day','s_vecation', 'w_vecation','workingday','is_event','cov_policy']

    #numerical_features, covid_features_cat, date_feature = data_preparation(hotel_id, date_feature, cancel_target
                                                        # , smooth=smooth, diff=diff
                                                        #)
    date_feature = data_preparation(hotel_id, booking_feature, cancel_target, twn_covid_data)

    date_splitting_point = date_feature.index.get_loc('2022/08/07')
    date_feature = date_feature.iloc[:date_splitting_point+1]
    numerical_features = model_metadata['features_configuration'][args.configuration]['numerical']
    categorical_features = model_metadata['features_configuration'][args.configuration]['categorical']

    #categorical_features = categorical_features + covid_features_cat

    for encoded_column in categorical_features :
        date_feature = labelencoding(date_feature, encoded_column)



    # categorical_features = ['vecation(是否為國定連假)', 'weekdate(星期，數值型)']  # encoded_columns + nonencoded_columns



    #date_feature = date_feature[numerical_features+categorical_features]

    pbounds = model_metadata['lstm2lstm_pbounds']
    for key, value in pbounds.items():
        pbounds[key] = eval(value)

    #pbounds = {'batch_size': (16, 64), #(50,200)
               #'learning_rate': (0.0001, 0.01),
               #'encoder_lstm_units': (32, 512),
               #'dropout': (0.1, 0.4),
               #'recurrent_dropout': (0.1, 0.4),
               #'decoder_dense_units': (8, 32)}
               #'l2': (0, 0.1)}


    training_process_opt_fn = partial(training_process_opt, prediction_time=prediction_time, date_feature=date_feature,
                                      numerical_features=numerical_features, categorical_features=categorical_features,
                                      n_splits=n_splits, input_range=input_range,
                                      test_size=test_size, loss='mse', model_type=model_type,
                                      max_train_size=max_train_size)

    optimization_process(training_process_opt_fn, pbounds, model_type=model_type, hotel_id=hotel_id, configuration=args.configuration)




