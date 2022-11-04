import argparse
import os
from functools import partial

import pandas as pd

from src.io.path_definition import get_file, _load_yaml
from src.io.load_data import data_preparation
from train.logic.training_process import training_process_opt
from train.logic.optimization_process import optimization_process


hotel_info = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_hotel_info.csv')))

cancel_target = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_target.csv')))
date_feature = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_date_feature.csv')))
hotel_meta = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_hotel_info.csv')), index_col=0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_range', type=int, help='length of input time series')
    parser.add_argument('--prediction_time', type=int, help='length of output time series')
    parser.add_argument('--hotel_id', type=int, help='id of hotel, should exists in cancel_dataset_target.csv')

    args = parser.parse_args()

    input_range = args.input_range
    prediction_time = args.prediction_time
    hotel_id = args.hotel_id
    model_type = 'CNN2LSTM'

    basic_parameters = _load_yaml(get_file(os.path.join('config', 'training_config.yml')))['basic_parameters']

    n_splits = basic_parameters['n_splits']
    test_size = basic_parameters['test_size']
    max_train_size = basic_parameters['max_train_size']

    ncategorical_features = []

    numerical_features, date_feature = data_preparation(hotel_id, date_feature, cancel_target)

    date_feature = date_feature[numerical_features]

    pbounds = _load_yaml(get_file(os.path.join('config', 'training_config.yml')))['cnn2lstm_pbounds']
    for key, value in pbounds.items():
        pbounds[key] = eval(value)

    training_process_opt_fn = partial(training_process_opt, prediction_time=prediction_time, date_feature=date_feature,
                                      numerical_features=numerical_features, n_splits=n_splits,
                                      input_range=input_range, test_size=test_size, loss='mse',
                                      model_type=model_type, max_train_size=max_train_size)

    optimized_parameters = optimization_process(training_process_opt_fn, pbounds, model_type=model_type, hotel_id=hotel_id)




