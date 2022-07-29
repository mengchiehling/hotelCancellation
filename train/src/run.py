import argparse
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

from src.io.path_definition import get_file
from train.logic.training_process import training_process


hotel_info = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_hotel_info.csv')))

cancel_target = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_target.csv')), index_col=0)
date_feature = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_date_feature.csv')), index_col=0)

le = LabelEncoder()
date_feature['weekdate(星期，數值型)'] = le.fit_transform(date_feature['weekdate(星期，數值型)'])

def data_preparation(hotel_id: int, date_feature: pd.DataFrame, cancel_target: pd.DataFrame, smooth:bool=False):

    column = f"hotel_{hotel_id}_canceled"

    hotel_cancel = cancel_target[column].replace("-", np.nan).dropna().astype(int)

    date_feature = date_feature.loc[hotel_cancel.index]
    date_feature['canceled'] = hotel_cancel

    if smooth:
        hotel_cancel_smooth = hotel_cancel.rolling(window=3, center=True)
        date_feature['canceled_label'] = hotel_cancel_smooth.mean()

        num_feature_columns = []

        for window in [7, 30, 60]:
            roll = hotel_cancel.rolling(window=window, min_periods=1)
            date_feature[f'canceled_{window}_roll'] = roll.mean()
            num_feature_columns.append(f'canceled_{window}_roll')

    else:
        num_feature_columns = ['canceled']
        date_feature['canceled_label'] = hotel_cancel

    return num_feature_columns, date_feature


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_range', type=int, help='length of input time series')
    parser.add_argument('--prediction_time', type=int, help='length of output time series')
    parser.add_argument('--hotel_id', type=int, help='id of hotel, should exists in cancel_dataset_target.csv')

    args = parser.parse_args()

    input_range = args.input_range
    prediction_time = args.prediction_time
    hotel_id = args.hotel_id

    n_splits = 7
    max_train_size = 360
    test_size = 28

    # categorical_features = ['midd(大學期中考週)', 'sallery(發薪日區間，每月5-10號)', 'is_rest_day(是否為假日)',
    #                         'vecation(是否為國定連假)', 's_vecation(暑假)', 'w_vecation(寒假)', 'weekdate(星期，數值型)']

    categorical_features = ['vecation(是否為國定連假)', 'weekdate(星期，數值型)']

    numerical_features, date_feature = data_preparation(hotel_id, date_feature, cancel_target, smooth=True)

    date_feature = date_feature.loc['2020/1/21': '2021/5/10']

    y_true, y_pred = training_process(input_range=input_range, prediction_time=prediction_time,
                                      date_feature=date_feature, numerical_features=numerical_features,
                                      categorical_features=categorical_features, n_splits=n_splits,
                                      max_train_size=max_train_size, encoder_lstm_units=[128, 512],
                                      decoder_dense_units=[10], test_size=test_size, loss='mse')

    import numpy as np

    for ix in range(n_splits):
        y_pred = np.round(y_pred)
        diff = abs((y_true[ix, :, :, 0] - y_pred[ix, :, :, 0]) / y_true[ix, :, :, 0])

        for iy in range(test_size):
            diff[iy][np.isinf(diff[iy])] = np.nan

        mape = np.nanmean(diff, axis=0)
        mape = np.round(mape * 100, 1)
        mse = mean_squared_error(y_true[ix, :, :, 0], y_pred[ix, :, :, 0], multioutput='raw_values')
        print(f"{ix}th fold: mape = {mape}")
        print(f"{ix}th fold: rmse = {np.sqrt(mse)}")
