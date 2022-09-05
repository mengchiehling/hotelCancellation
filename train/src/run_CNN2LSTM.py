import argparse
import os
from functools import partial
from typing import Optional, List

import numpy as np
import pandas as pd

from src.io.path_definition import get_file
from src.io.load_parameters import optimized_parameters
from train.logic.training_process import training_process, training_process_opt
from train.logic.optimization_process import optimization_process
from train.logic.data_preparation import data_preparation


hotel_info = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_hotel_info.csv')))

cancel_target = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_target.csv')), index_col=0)
date_feature = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_date_feature.csv')), index_col=0)
hotel_meta = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_hotel_info.csv')), index_col=0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_range', type=int, help='length of input time series')
    parser.add_argument('--prediction_time', type=int, help='length of output time series')
    parser.add_argument('--hotel_id', type=int, help='id of hotel, should exists in cancel_dataset_target.csv')
    parser.add_argument('--diff', type=int,  nargs='+', help='差分', default=[])
    parser.add_argument('--smooth', action='store_true')

    args = parser.parse_args()

    input_range = args.input_range
    prediction_time = args.prediction_time
    hotel_id = args.hotel_id
    model_type = 'CNN2LSTM'
    diff = args.diff
    smooth=args.smooth

    n_splits = 15
    test_size = 28

    numerical_features, date_feature = data_preparation(hotel_id, date_feature, cancel_target,
                                                        smooth=smooth, diff=diff)

    date_feature = date_feature[numerical_features]

    pbounds = {'batch_size': (4, 16),
               'learning_rate': (0.0001, 0.01),
               'encoder_filters': (32, 512),
               'dropout': (0.1, 0.4),
               'recurrent_dropout': (0.1, 0.4),
               'decoder_dense_units': (8, 32),
               'l2': (0, 0.1),
               'momentum': (0.9, 0.999)}

    training_process_opt_fn = partial(training_process_opt, prediction_time=prediction_time, date_feature=date_feature,
                                      numerical_features=numerical_features, n_splits=n_splits,
                                      input_range=input_range, test_size=test_size, loss='mse',
                                      model_type=model_type, max_train_size=365)

    optimized_parameters = optimization_process(training_process_opt_fn, pbounds, model_type=model_type)

    params, _ = optimized_parameters("CNN2LSTM_logs_[\d]{8}-[\d]{2}.json")





