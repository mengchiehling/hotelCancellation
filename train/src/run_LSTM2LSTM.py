import argparse
import os
from functools import partial
from datetime import datetime
import numpy as np
import pandas as pd

from src.io.path_definition import get_file
from train.logic.training_process import training_process_opt
from train.logic.optimization_process import optimization_process


hotel_info = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_hotel_info.csv')))

cancel_target = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_target.csv')))
date_feature = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_date_feature.csv')))
hotel_meta = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_hotel_info.csv')), index_col=0)

covid_data = pd.read_excel(get_file(os.path.join('data', 'owid-covid-data.xlsx')),
                           #engine='openpyxl'
                           )


def data_preparation(hotel_id: int, date_feature: pd.DataFrame, cancel_target: pd.DataFrame):

    column = f"hotel_{hotel_id}_canceled"

    cancel_target['date'] = cancel_target['date'].apply(lambda x: datetime.strptime(x, '%Y/%m/%d').strftime("%Y/%m/%d"))
    cancel_target.set_index('date', inplace=True)
    hotel_cancel = cancel_target[column].replace("-", np.nan).dropna().astype(int)

    date_feature['date'] = date_feature['date'].apply(lambda x: datetime.strptime(x, '%Y/%m/%d').strftime("%Y/%m/%d"))
    date_feature.set_index('date', inplace=True)
    date_feature = date_feature.loc[hotel_cancel.index]

    date_feature['canceled'] = hotel_cancel   # 原始值

    twn_covid_data = covid_data[covid_data['iso_code']=='TWN']
    twn_covid_data['date'] = twn_covid_data['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').strftime("%Y/%m/%d"))
    twn_covid_data.set_index('date', inplace=True)

    covid_features_num = [] #['new_cases', 'new_deaths']


    date_feature = date_feature.join(twn_covid_data[covid_features_num].fillna(0))

    num_feature_columns = ['canceled'] + covid_features_num

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
    model_type = 'LSTM2LSTM'

    n_splits = 10
    test_size = 20


    numerical_features, date_feature = data_preparation(hotel_id, date_feature, cancel_target)
    # print(numerical_features)

    date_feature = date_feature[numerical_features]

    pbounds = {'batch_size': (4, 16) , #(50,200)
               'learning_rate': (0.0001, 0.01),
               'encoder_lstm_units': (32, 512),
               'dropout': (0.1, 0.4),
               'recurrent_dropout': (0.1, 0.4),
               'decoder_dense_units': (8, 32),
               'l2': (0, 0.1)}

    training_process_opt_fn = partial(training_process_opt, prediction_time=prediction_time, date_feature=date_feature,
                                      numerical_features=numerical_features, n_splits=n_splits,
                                      input_range=input_range, test_size=test_size, loss='mse', model_type=model_type,
                                      max_train_size=200)


    optimization_process(training_process_opt_fn, pbounds, model_type=model_type, hotel_id=hotel_id)
