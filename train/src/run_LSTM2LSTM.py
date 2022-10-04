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

    covid_features_num = ['new_cases', 'new_deaths']
    covid_features_cat = ['']

    date_feature = date_feature.join(twn_covid_data[covid_features_num + covid_features_cat].fillna(0))

    num_feature_columns = ['canceled'] + covid_features_num

    return num_feature_columns,  covid_features_cat, date_feature


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

    n_splits = 7
    test_size = 28

    categorical_features = []

    numerical_features, covid_features_cat, date_feature = data_preparation(hotel_id, date_feature, cancel_target)

    categorical_features = categorical_features + covid_features_cat

    date_feature = date_feature[numerical_features]

    pbounds = {'batch_size': (4, 16),
               'learning_rate': (0.0001, 0.01),
               'encoder_lstm_units': (32, 512),
               'dropout': (0.1, 0.4),
               'recurrent_dropout': (0.1, 0.4),
               'decoder_dense_units': (8, 32),
               'l2': (0, 0.1)}

    training_process_opt_fn = partial(training_process_opt, prediction_time=prediction_time, date_feature=date_feature,
                                      numerical_features=numerical_features, n_splits=n_splits,
                                      input_range=input_range, test_size=test_size, loss='mse', model_type=model_type,
                                      max_train_size=365)


    optimization_process(training_process_opt_fn, pbounds, model_type=model_type, hotel_id=hotel_id)
    # 寫法好像不太一樣 ? 左邊為何要寫 _ =
    #_ = optimization_process(training_process_opt_fn, pbounds, model_type=model_type)

    #params, _ = optimized_parameters(f"{hotel_id}_{model_type}"+"_logs_[\d]{8}-[\d]{2}.json")

    #params['batch_size'] = int(params['batch_size'])
    #params['decoder_dense_units'] = int(params['batch_size'])
    #params['encoder_lstm_units'] = int(params['encoder_lstm_units'])

    #從這行一直到 joblib.dump可以另外開一個script放，以儲存訓練好的模型，不用到時候還要在訓練一遍
    #只有儲存參數(告訴我訓練模型該如何調)，但並沒有儲存和模型有關的資訊
    #params, _ = optimized_parameters(f"{model_type}" + "_logs_[\d]{8}-[\d]{2}.json")

    #params['batch_size'] = int(params['batch_size'])
    #params['decoder_dense_units'] = int(params['batch_size'])
    #params['encoder_lstm_units'] = int(params['encoder_lstm_units'])

    #model, scaler = model_training_pipeline(date_feature=date_feature, test_size=test_size, input_range=input_range,
                                            #prediction_time=prediction_time, numerical_features=numerical_features,
                                            #categorical_features=categorical_features,model_type=model_type, **params)


    #dir = os.path.join(get_project_dir(), 'data', 'model', model_type)

    #if not os.path.isdir(dir):
        #os.makedirs(dir)
    #model.save(os.path.join(dir, 'model'))

    #with open(os.path.join(dir, 'scaler'), mode='wb') as f:
        #joblib.dump(scaler, f)


    # change to different metrics

    #y_true, y_pred = training_process(input_range=input_range, prediction_time=prediction_time,
                                      #date_feature=date_feature, numerical_features=numerical_features, categorical_features=categorical_features,
                                      #n_splits=n_splits,max_train_size=180, test_size=test_size, model_type=model_type, loss='mse', **params)

    #adapted_mape = mean_absolute_percentage_error(y_true+1, y_pred+1)

    #print(adapted_mape)