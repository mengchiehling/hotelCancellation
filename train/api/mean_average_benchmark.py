import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
from src.common.load_data import load_training_data
from src.io.path_definition import get_file, load_yaml_file
from train.api.training_run import create_dataset, to_timeseries_dataframe


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_range', type=int, default=28)
    parser.add_argument('--prediction_time', type=int, default=7)
    parser.add_argument('--hotel_id', type=int, help='id of hotel, should exists in cancel_dataset_target.csv')
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--test_size', type=int, default=28)
    parser.add_argument('--max_train_size', type=int, default=180)

    args = parser.parse_args()

    input_range = args.input_range
    prediction_time = args.prediction_time
    hotel_id = args.hotel_id
    n_splits = args.n_splits
    test_size = args.test_size
    max_train_size = args.max_train_size
    lead_time = 0

    df, idx = load_training_data(args.hotel_id, remove_business_booking=True)

    train_dataset, test_dataset, _, _ = create_dataset(df, test_size=args.test_size)

    end_of_train_dataset = idx.index(train_dataset.iloc[-1]['check_in'])

    train_dataset = to_timeseries_dataframe(train_dataset, idx[:end_of_train_dataset+1])

    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, max_train_size=max_train_size)

    y_pred = []
    y_true = []

    for n_fold, (train_index, test_index) in enumerate(tscv.split(train_dataset)):
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