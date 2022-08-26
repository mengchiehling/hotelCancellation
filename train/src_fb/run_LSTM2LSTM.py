import argparse
import os
from functools import partial

import numpy as np
import pandas as pd

from src.io.path_definition import get_file
from train.logic_fb.training_process import training_process, training_process_opt
from train.logic_fb.optimization_process import optimization_process


hotel_info = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_hotel_info.csv')))

cancel_target = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_target.csv')), index_col=0)
date_feature = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_date_feature.csv')), index_col=0)
hotel_meta = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_hotel_info.csv')), index_col=0)


def data_preparation(hotel_id: int, date_feature: pd.DataFrame, cancel_target: pd.DataFrame):

    column = f"hotel_{hotel_id}_canceled"

    hotel_cancel = cancel_target[column].replace("-", np.nan).dropna().astype(int)

    date_feature = date_feature.loc[hotel_cancel.index]
    date_feature['canceled'] = hotel_cancel   # 原始值

    return date_feature


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_range', type=int, help='length of input time series')
    parser.add_argument('--prediction_time', type=int, help='length of output time series')
    parser.add_argument('--hotel_id', type=int, help='id of hotel, should exists in cancel_dataset_target.csv')

    args = parser.parse_args()

    input_range = args.input_range
    prediction_time = args.prediction_time
    hotel_id = args.hotel_id
    model_type = 'LSTM2LSTM'

    n_splits = 15
    test_size = 28

    date_feature = data_preparation(hotel_id, date_feature, cancel_target)

    pbounds = {'batch_size': (4, 16),
               'learning_rate': (0.0001, 0.01),
               'encoder_lstm_units': (32, 512),
               'decoder_lstm_units': (32, 512),
               'dropout': (0.1, 0.4),
               'recurrent_dropout': (0.1, 0.4),
               'decoder_dense_units': (8, 32)}

    training_process_opt_fn = partial(training_process_opt, prediction_time=prediction_time, date_feature=date_feature,
                                      n_splits=n_splits, input_range=input_range,
                                      test_size=test_size, loss='mse', model_type=model_type,
                                      max_train_size=365)

    optimized_parameters = optimization_process(training_process_opt_fn, pbounds, model_type=model_type)



