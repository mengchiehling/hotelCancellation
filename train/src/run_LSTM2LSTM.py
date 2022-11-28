import argparse
import os
import joblib
from functools import partial
from datetime import datetime

import pandas as pd

from src.io.path_definition import get_file, _load_yaml, get_project_dir
from src.io.load_data import data_preparation
from train.logic.training_process import training_process_opt
from train.logic.optimization_process import optimization_process

cancel_target = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_target.csv')))

covid_data = pd.read_excel(get_file(os.path.join('data', 'owid-covid-data.xlsx')),
                           #engine='openpyxl'
                           )

twn_covid_data = covid_data[covid_data['iso_code']=='TWN']
twn_covid_data['date'] = twn_covid_data['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').strftime("%Y/%m/%d"))
twn_covid_data.set_index('date', inplace=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_range', type=int, help='length of input time series')
    parser.add_argument('--prediction_time', type=int, help='length of output time series')
    parser.add_argument('--hotel_id', type=int, help='id of hotel, should exists in cancel_dataset_target.csv')
    parser.add_argument('--configuration', type=str, help='"A", "B", "C", "D", ...pick only one')

    args = parser.parse_args()

    input_range = args.input_range
    prediction_time = args.prediction_time
    hotel_id = args.hotel_id
    model_type = 'LSTM2LSTM'

    booking_feature = pd.read_csv(get_file(os.path.join('data', f'{hotel_id}.csv')))

    model_metadata = _load_yaml(get_file(os.path.join('config', 'training_config.yml')))

    basic_parameters = model_metadata['basic_parameters']

    n_splits = basic_parameters['n_splits']
    test_size = basic_parameters['test_size']
    max_train_size = basic_parameters['max_train_size']

    date_feature = data_preparation(hotel_id, booking_feature, cancel_target, twn_covid_data)
    date_splitting_point = date_feature.index.get_loc('2022/08/07')
    date_feature = date_feature.iloc[:date_splitting_point+1]
    numerical_features = model_metadata['features_configuration'][args.configuration]
    # numerical_features = model_metadata['features_configuration'][args.configuration]['numerical']
    # categorical_features = model_metadata['features_configuration'][args.configuration]['categorical']

    pbounds = model_metadata['lstm2lstm_pbounds']
    for key, value in pbounds.items():
        pbounds[key] = eval(value)

    training_process_opt_fn = partial(training_process_opt, prediction_time=prediction_time, date_feature=date_feature,
                                      numerical_features=numerical_features, n_splits=n_splits,
                                      input_range=input_range, test_size=test_size, loss='mse', model_type=model_type,
                                      max_train_size=max_train_size)

    optimization_process(training_process_opt_fn, pbounds, model_type=model_type, hotel_id=hotel_id, configuration=args.configuration)