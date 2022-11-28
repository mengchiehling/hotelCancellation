import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error

from src.io.load_parameters import optimized_parameters
from src.io.path_definition import get_file, _load_yaml
from src.io.load_data import data_preparation
from train.logic.training_process import training_process
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

cancel_target = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_target.csv')))

covid_data = pd.read_excel(get_file(os.path.join('data', 'owid-covid-data.xlsx')),
                           #engine='openpyxl'
                           )

twn_covid_data = covid_data[covid_data['iso_code']=='TWN']
twn_covid_data['date'] = twn_covid_data['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').strftime("%Y/%m/%d"))
twn_covid_data.set_index('date', inplace=True)



if __name__ == "__main__" :

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_range', type=int, help='length of input time series')
    parser.add_argument('--prediction_time', type=int, help='length of output time series')
    parser.add_argument('--hotel_id', type=int, help='id of hotel, should exists in cancel_dataset_target.csv')

    args = parser.parse_args()

    input_range = args.input_range
    prediction_time = args.prediction_time
    hotel_id = args.hotel_id
    model_type = 'LSTM2LSTM'

    booking_feature = pd.read_csv(get_file(os.path.join('data', f'{hotel_id}.csv')))

    #已經找出最優化的參數組合，它會放入下面的training process，最後產出y_true,y_pred
    params, _ = optimized_parameters(f"{hotel_id}_{model_type}" + "_logs_[\d]{8}-[\d]{2}.json")

    basic_parameters = _load_yaml(get_file(os.path.join('config', 'training_config.yml')))['basic_parameters']

    n_splits = basic_parameters['n_splits']
    test_size = basic_parameters['test_size']
    max_train_size = basic_parameters['max_train_size']

    params['batch_size'] = int(params['batch_size'])
    params['decoder_dense_units'] = int(params['batch_size'])
    params['encoder_lstm_units'] = int(params['encoder_lstm_units'])

    # 做training或evaluation都要讀取數據
    numerical_features, date_feature = data_preparation(hotel_id, booking_feature, cancel_target, twn_covid_data)

    date_feature = date_feature[numerical_features]

    y_true, y_pred = training_process(input_range=input_range, prediction_time=prediction_time,
                                      date_feature=date_feature, numerical_features=numerical_features,
                                      n_splits=n_splits,max_train_size=max_train_size, test_size=test_size,
                                      model_type=model_type, loss='mse', **params)


    adapted_mape = mean_absolute_percentage_error(y_true.flatten()+1, y_pred.flatten()+1)

    print(adapted_mape)