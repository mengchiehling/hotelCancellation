import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error

from src.io.path_definition import get_file, load_yaml_file
from train.logic.data_preparation import tf_input_pipeline

hotel_info = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_hotel_info.csv')))

cancel_target = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_target.csv')))
date_feature = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_date_feature.csv')))
hotel_meta = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_hotel_info.csv')), index_col=0)


def data_preparation(hotel_id: int, date_feature: pd.DataFrame, cancel_target: pd.DataFrame):

    column = f"hotel_{hotel_id}_canceled"

    cancel_target['date'] = cancel_target['date'].apply(lambda x: datetime.strptime(x, '%Y/%m/%d').strftime("%Y/%m/%d"))
    cancel_target.set_index('date', inplace=True)
    hotel_cancel = cancel_target[column].replace("-", np.nan).dropna().astype(int)

    date_feature['date'] = date_feature['date'].apply(lambda x: datetime.strptime(x, '%Y/%m/%d').strftime("%Y/%m/%d"))
    date_feature.set_index('date', inplace=True)
    date_feature = date_feature.loc[hotel_cancel.index]

    date_feature['canceled'] = hotel_cancel   # 原始值

    num_feature_columns = ['canceled']

    return num_feature_columns, date_feature


if __name__ == "__main__" :

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_range', type=int, help='length of input time series')
    parser.add_argument('--prediction_time', type=int, help='length of output time series')
    parser.add_argument('--hotel_id', type=int, help='id of hotel, should exists in cancel_dataset_target.csv')

    args = parser.parse_args()

    input_range = args.input_range
    prediction_time = args.prediction_time
    hotel_id = args.hotel_id
    lead_time = 0

    basic_parameters = load_yaml_file(get_file(os.path.join('config', 'training_config.yml')))['basic_parameters']

    n_splits = basic_parameters['n_splits']
    test_size = basic_parameters['test_size']
    max_train_size = basic_parameters['max_train_size']

    # 做training或evaluation都要讀取數據

    numerical_features, date_feature = data_preparation(hotel_id, date_feature, cancel_target)
    date_feature = date_feature[numerical_features]

    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, max_train_size=max_train_size)

    y_pred = []
    y_true = []

    for n_fold, (train_index, test_index) in enumerate(tscv.split(date_feature)):
        test_index = np.arange(test_index[0] - input_range - lead_time - prediction_time, test_index[-1] + 1)

        df_train = date_feature.iloc[train_index]

        time_begin = df_train.index[0]
        time_end = df_train.index[-1]

        df_test = date_feature.iloc[test_index]

        X_test, y_test = tf_input_pipeline(df_test, input_range=input_range, prediction_time=prediction_time,
                                           numerical_features=numerical_features)

        X_test = np.squeeze(X_test['encoder_X_num'], axis=2)
        y_test = np.squeeze(y_test['true'], axis=2)[:, 0]
        y_pred.append(X_test.mean(axis=1))
        y_true.append(y_test)

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    diff = abs((y_pred - y_true) / y_true)

    diff[np.isinf(diff)] = np.nan

    # adapted_mape = mean_absolute_percentage_error(y_true.flatten() + 1, y_pred.flatten() + 1)
    adapted_mape = np.nanmean(diff)

    print(f"{hotel_id}: {np.round(adapted_mape, 2)}")