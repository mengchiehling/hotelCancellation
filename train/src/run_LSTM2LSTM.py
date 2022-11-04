import argparse
import os
from functools import partial
import pandas as pd

from src.io.path_definition import get_file, _load_yaml
from src.io.load_data import data_preparation
from train.logic.training_process import training_process_opt
from train.logic.optimization_process import optimization_process


hotel_info = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_hotel_info.csv')))

cancel_target = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_target.csv')))
date_feature = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_date_feature.csv')))
hotel_meta = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_hotel_info.csv')), index_col=0)


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

    basic_parameters = _load_yaml(get_file(os.path.join('config', 'training_config.yml')))['basic_parameters']

    n_splits = basic_parameters['n_splits']
    test_size = basic_parameters['test_size']
    max_train_size = basic_parameters['max_train_size']

    categorical_features = []

    numerical_features, date_feature = data_preparation(hotel_id, date_feature, cancel_target)

    date_feature = date_feature[numerical_features]

    pbounds = _load_yaml(get_file(os.path.join('config', 'training_config.yml')))['lstm2lstm_pbounds']
    for key, value in pbounds.items():
        pbounds[key] = eval(value)

    training_process_opt_fn = partial(training_process_opt, prediction_time=prediction_time, date_feature=date_feature,
                                      numerical_features=numerical_features, n_splits=n_splits,
                                      input_range=input_range, test_size=test_size, loss='mse', model_type=model_type,
                                      max_train_size=max_train_size)


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