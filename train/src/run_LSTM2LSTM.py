import argparse
import os
from functools import partial
from typing import Optional, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

from src.io.path_definition import get_file, get_project_dir
from src.io.load_parameters import optimized_parameters
from train.logic.training_process import training_process, training_process_opt
from train.logic.optimization_process import optimization_process
from train.logic.model_selection import model_training_pipeline


hotel_info = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_hotel_info.csv')))

cancel_target = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_target.csv')), index_col=0)
date_feature = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_date_feature.csv')), index_col=0)
hotel_meta = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_hotel_info.csv')), index_col=0)

# le = LabelEncoder()
# date_feature['weekdate(星期，數值型)'] = le.fit_transform(date_feature['weekdate(星期，數值型)'])


def labelencoding(df, column: str):

    le = LabelEncoder()
    df.loc[:, column] = le.fit_transform(df[column])

    return df


def data_preparation(hotel_id: int, date_feature: pd.DataFrame, cancel_target: pd.DataFrame, smooth:bool=False,
                     diff: Optional[List[int]]=None):

    column = f"hotel_{hotel_id}_canceled"

    hotel_cancel = cancel_target[column].replace("-", np.nan).dropna().astype(int)

    date_feature = date_feature.loc[hotel_cancel.index]
    date_feature['canceled'] = hotel_cancel   # 原始值
    date_feature['canceled_label'] = hotel_cancel  # hotel_cancel_smooth.mean()

    num_feature_columns = ['canceled', 'canceled_label','days2vecation','vecation_days','Precp','PrecpHour','SunShine','Temperature']

    if smooth:
        # Smoothed features for input

        for window in [3, 14]:
            c  = f'canceled_{window}_roll'
            roll = hotel_cancel.rolling(window=window, min_periods=1)
            date_feature[c] = roll.mean()
            num_feature_columns.extend([c])
        date_feature['MACD'] = date_feature[f'canceled_3_roll'] - date_feature[f'canceled_14_roll']
        num_feature_columns.append('MACD')

    if diff is not None:
        for time_diff in diff:
            c = f'diff_{time_diff}'
            date_feature[c] = (date_feature['canceled'] - date_feature['canceled'].shift(time_diff)).fillna(0)
            num_feature_columns.append(c)

    return num_feature_columns, date_feature


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
    model_type = 'LSTM2LSTM'
    diff = args.diff
    smooth=args.smooth

    n_splits = 7
    test_size = 28

    categorical_features = ['vecation', 'weekdate','season','midd','sallery', 'is_rest_day','s_vecation', 'w_vecation','workingday','is_event','cov_policy']  # encoded_columns + nonencoded_columns

    for encoded_column in categorical_features:
        date_feature = labelencoding(date_feature, encoded_column)

    # categorical_features = ['midd(大學期中考週)', 'sallery(發薪日區間，每月5-10號)', 'is_rest_day(是否為假日)',
    #                         'vecation(是否為國定連假)', 's_vecation(暑假)', 'w_vecation(寒假)', 'weekdate(星期，數值型)']

    # categorical_features = ['vecation(是否為國定連假)', 'weekdate(星期，數值型)']  # encoded_columns + nonencoded_columns

    numerical_features, date_feature = data_preparation(hotel_id, date_feature, cancel_target,
                                                        smooth=smooth, diff=diff)

    date_feature = date_feature[numerical_features+categorical_features]

    pbounds = {'batch_size': (4, 16),
               'learning_rate': (0.0001, 0.01),
               'encoder_lstm_units': (32, 512),
               'dropout': (0.1, 0.4),
               'recurrent_dropout': (0.1, 0.4),
               'decoder_dense_units': (8, 32)}

    training_process_opt_fn = partial(training_process_opt, prediction_time=prediction_time, date_feature=date_feature,
                                      numerical_features=numerical_features, categorical_features=categorical_features,
                                      n_splits=n_splits, input_range=input_range,
                                      test_size=test_size, loss='mse', model_type=model_type,
                                      max_train_size=365)

    _ = optimization_process(training_process_opt_fn, pbounds, model_type=model_type)

    params, _ = optimized_parameters(f"{model_type}_logs_[\d]{8}-[\d]{2}.json")
    params['batch_size'] = int(params['batch_size'])
    params['decoder_dense_units'] = int(params['batch_size'])
    params['encoder_lstm_units'] = int(params['encoder_lstm_units'])

    model, scaler = model_training_pipeline(date_feature=date_feature, test_size=test_size, input_range=input_range,
                                            prediction_time=prediction_time, numerical_features=numerical_features,
                                            categorical_features=categorical_features,model_type=model_type, **params)


    dir = os.path.join(get_project_dir(), 'data', 'model', model_type)

    if not os.path.isdir(dir):
        os.makedirs(dir)
    model.save(os.path.join(dir, 'model'))

    with open(os.path.join(dir, 'scaler')) as f:
        joblib.dump(scaler, f)


    # change to different metrics

    y_true, y_pred = training_process(input_range=input_range, prediction_time=prediction_time,
                                      date_featur=date_feature, numerical_features=numerical_features, categorical_features=categorical_features,
                                      n_splits=n_splits,max_train_size=365, test_size=test_size, model_type=model_type, loss='mse', **params)

    adapted_mape = mean_absolute_percentage_error(y_true+1, y_pred+1)

    print(adapted_mape)