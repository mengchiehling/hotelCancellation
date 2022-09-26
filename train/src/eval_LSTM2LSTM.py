import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error

from src.io.load_parameters import optimized_parameters
from src.io.path_definition import get_file
from train.logic.training_process import training_process
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

hotel_info = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_hotel_info.csv')))

cancel_target = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_target.csv')), index_col=0)
date_feature = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_date_feature.csv')), index_col=0)
hotel_meta = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_hotel_info.csv')), index_col=0)


# 這個是要放在eval ? 還是放在run ?
def labelencoding(df, column: str):

    le = LabelEncoder()
    df.loc[:, column] = le.fit_transform(df[column])

    return df


def data_preparation(hotel_id: int, date_feature: pd.DataFrame, cancel_target: pd.DataFrame):

    column = f"hotel_{hotel_id}_canceled"

    hotel_cancel = cancel_target[column].replace("-", np.nan).dropna().astype(int)

    date_feature = date_feature.loc[hotel_cancel.index]
    date_feature['canceled'] = hotel_cancel   # 原始值
    date_feature['canceled_label'] = hotel_cancel  # hotel_cancel_smooth.mean()

    #num_feature_columns = ['canceled', 'canceled_label','days2vecation','vecation_days','Precp','PrecpHour','SunShine','Temperature']
    num_feature_columns = ['canceled', 'canceled_label']


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
    model_type = 'LSTM2LSTM'

    #已經找出最優化的參數組合，它會放入下面的training process，最後產出y_true,y_pred
    params, _ = optimized_parameters(f"{model_type}" + "_logs_[\d]{8}-[\d]{2}.json")

    params['batch_size'] = int(params['batch_size'])
    params['decoder_dense_units'] = int(params['batch_size'])
    params['encoder_lstm_units'] = int(params['encoder_lstm_units'])

    n_splits = 7
    test_size = 30

    categorical_features =[] #['vecation', 'weekdate','season','midd','sallery', 'is_rest_day','s_vecation', 'w_vecation','workingday','is_event','cov_policy']  # encoded_columns + nonencoded_columns

    #for encoded_column in categorical_features:
        #date_feature = labelencoding(date_feature, encoded_column)


    # 做training或evaluation都要讀取數據
    numerical_features, date_feature = data_preparation(hotel_id, date_feature, cancel_target)
    # 數值型+類別型，已經合成為date_feature ? 那下面training_process還需要三者都放進去嗎 ?
    date_feature = date_feature[numerical_features+categorical_features]

    y_true, y_pred = training_process(input_range=input_range, prediction_time=prediction_time,
                                      date_feature=date_feature, numerical_features=numerical_features, categorical_features=categorical_features,
                                      n_splits=n_splits,max_train_size=180, test_size=test_size, model_type=model_type, loss='mse', **params)


    adapted_mape = mean_absolute_percentage_error(y_true+1, y_pred+1)

    print(adapted_mape)