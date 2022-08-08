import argparse
import os
from functools import partial

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

from src.io.path_definition import get_file
from train.logic.training_process import training_process, training_process_opt
from train.logic.optimization_process import optimization_process


hotel_info = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_hotel_info.csv')))

cancel_target = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_target.csv')), index_col=0)
date_feature = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_date_feature.csv')), index_col=0)
hotel_meta = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_hotel_info.csv')), index_col=0)

le = LabelEncoder()
date_feature['weekdate(星期，數值型)'] = le.fit_transform(date_feature['weekdate(星期，數值型)'])

def data_preparation(hotel_id: int, date_feature: pd.DataFrame, cancel_target: pd.DataFrame, smooth:bool=False):

    column = f"hotel_{hotel_id}_canceled"

    hotel_cancel = cancel_target[column].replace("-", np.nan).dropna().astype(int)
    max_room_number = hotel_meta.loc[hotel_id]['房間數']

    # scale to hotel cancellation to 0 - 1
    hotel_cancel = hotel_cancel/max_room_number

    date_feature = date_feature.loc[hotel_cancel.index]
    date_feature['canceled'] = hotel_cancel   # 原始值

    if smooth:
        # hotel_cancel_smooth = hotel_cancel.rolling(window=3, center=True)
        date_feature['canceled_label'] = hotel_cancel # hotel_cancel_smooth.mean()

        num_feature_columns = ['canceled']

        for window in [3, 14]:
            roll = hotel_cancel.rolling(window=window, min_periods=1)
            date_feature[f'canceled_{window}_roll'] = roll.mean()
            num_feature_columns.extend([f'canceled_{window}_roll'])
        date_feature['MACD'] = date_feature[f'canceled_3_roll'] - date_feature[f'canceled_14_roll']
        num_feature_columns.append('MACD')
    else:
        num_feature_columns = ['canceled']
        date_feature['canceled_label'] = hotel_cancel

    return num_feature_columns, date_feature


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_range', type=int, help='length of input time series')
    parser.add_argument('--prediction_time', type=int, help='length of output time series')
    parser.add_argument('--hotel_id', type=int, help='id of hotel, should exists in cancel_dataset_target.csv')
    parser.add_argument('--model_name', type=str, default='', help='model details')
    parser.add_argument('--model_type', type=str, help='model architecture')
    parser.add_argument('--covid_status', type=int, help='')

    args = parser.parse_args()

    input_range = args.input_range
    prediction_time = args.prediction_time
    hotel_id = args.hotel_id
    covid_status = args.covid_status
    model_name = args.model_name
    model_type = args.model_type

    if covid_status == 1:
        timestamps = (None, '2020/12/31')
    if covid_status == 2:
        timestamps = ('2021/1/1', None)


    n_splits = 7
    test_size = 28

    # categorical_features = ['midd(大學期中考週)', 'sallery(發薪日區間，每月5-10號)', 'is_rest_day(是否為假日)',
    #                         'vecation(是否為國定連假)', 's_vecation(暑假)', 'w_vecation(寒假)', 'weekdate(星期，數值型)']

    categorical_features = ['vecation(是否為國定連假)', 'weekdate(星期，數值型)']

    numerical_features, date_feature = data_preparation(hotel_id, date_feature, cancel_target, smooth=True)

    if not timestamps[0]:
        date_feature = date_feature.loc[: timestamps[1]]
    if not timestamps[1]:
        date_feature = date_feature.loc[timestamps[0]:]

    pbounds = {'batch_size': (4, 16),
               'learning_rate': (0.0001, 0.01)}

    training_process_opt_fn = partial(training_process_opt, prediction_time=prediction_time, date_feature=date_feature,
                                      numerical_features=numerical_features, categorical_features=categorical_features,
                                      n_splits=n_splits, encoder_lstm_units=[64], input_range=input_range,
                                      decoder_dense_units=[8], test_size=test_size, loss='mse', model_name=model_name,
                                      model_type=model_type, max_train_size=720)

    optimized_parameters = optimization_process(training_process_opt_fn, pbounds, model_type=model_type,
                                                model_name=model_name)
    #
    #         if date_feature_id > ...:
    #             (run..)
    #         else:
    #             continue


