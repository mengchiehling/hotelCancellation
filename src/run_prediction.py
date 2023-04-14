import argparse
import os
from datetime import datetime

import pandas as pd
import numpy as np
# from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from src.io.path_definition import get_file, load_yaml_file
from src.io.load_data import data_preparation
from src.io.load_parameters import load_optimized_parameters
from src.logic.common.functions import generate_weekly_inputs
from train.logic.model_selection import model_training_pipeline
from train.logic.data_preparation import tf_input_pipeline
from train.src import config

cancel_target = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_target.csv')))

covid_data = pd.read_excel(get_file(os.path.join('data', 'owid-covid-data.xlsx')),
                           #engine='openpyxl'
                           )

twn_covid_data = covid_data[covid_data['iso_code']=='TWN']
twn_covid_data['date'] = twn_covid_data['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').strftime("%Y/%m/%d"))
twn_covid_data.set_index('date', inplace=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--hotel_id', type=int, help='id of hotel, should exists in cancel_dataset_target.csv')
    parser.add_argument('--configuration', type=str, help='')

    args = parser.parse_args()

    hotel_id = args.hotel_id
    model_type = 'LSTM2LSTM'
    config.configuration = args.configuration

    model_metadata = load_yaml_file(get_file(os.path.join('config', 'training_config.yml')))

    basic_parameters = model_metadata['basic_parameters']

    n_splits = basic_parameters['n_splits']
    test_size = basic_parameters['test_size']
    max_train_size = basic_parameters['max_train_size']
    prediction_time = basic_parameters['prediction_time']
    input_range = basic_parameters['input_range']

    # dir_ = os.path.join(get_project_dir(), 'data', 'model', f'{hotel_id}_{model_type}_{args.configuration}')
    # model = load_model(dir_)

    booking_feature = pd.read_csv(get_file(os.path.join('data', f'{hotel_id}.csv')))
    date_feature = data_preparation(hotel_id, booking_feature, cancel_target, twn_covid_data)

    categorical_features = model_metadata['features_configuration'][args.configuration]['categorical']

    cat_labelencoding_dict = dict()
    for cat_labelencoding in categorical_features:
        cat_labelencoding_dict[cat_labelencoding] = LabelEncoder()
        date_feature.loc[:,cat_labelencoding] = cat_labelencoding_dict[cat_labelencoding].fit_transform(date_feature[cat_labelencoding])


    date_splitting_point = date_feature.index.get_loc('2022/08/07')

    # train-test split
    # extract training data
    date_feature_train = date_feature.iloc[:date_splitting_point+1]  # until 2022/08/07
    # extract testing data
    date_feature_test = date_feature.iloc[date_splitting_point - input_range + 1 :date_splitting_point + prediction_time + 1]

    # Model training
    params, optimized_metric = load_optimized_parameters(f"{hotel_id}_{model_type}_{args.configuration}" + "_logs_[\d]{8}-[\d]{2}.json")

    #numerical_features = model_metadata['features_configuration'][args.configuration]
    numerical_features = model_metadata['features_configuration'][args.configuration]['numerical']


    datasize = len(date_feature_train)
    y_true = []
    y_pred = []

    while (date_splitting_point + prediction_time) < len(date_feature):
        model, scaler = model_training_pipeline(date_feature=date_feature_train, test_size=test_size, input_range=input_range,
                                                prediction_time=prediction_time, numerical_features=numerical_features,
                                                loss='mse', model_type=model_type, **params)

        date_feature_test.loc[:, numerical_features] = scaler.transform(date_feature_test[numerical_features])

        X_test, y_test = tf_input_pipeline(date_feature_test, input_range=input_range, prediction_time=prediction_time,
                                           numerical_features=numerical_features)

        y_true_extend = np.repeat(y_test['true'].reshape(-1, 1), len(scaler.scale_), axis=1)
        y_true_reshape = scaler.inverse_transform(y_true_extend)[:, 0].reshape(y_test['true'].shape)
        y_true.append(y_true_reshape)

        X_test = generate_weekly_inputs(X_test, y_test) # for booking data as decoder input

        pred = model.predict(X_test)

        y_pred_extend = np.repeat(pred.reshape(-1, 1), len(scaler.scale_), axis=1)
        y_pred_reshape = np.round(scaler.inverse_transform(y_pred_extend)[:, 0].reshape(pred.shape))
        y_pred.append(y_pred_reshape)

        date_splitting_point = date_splitting_point + 1 # date_splitting_point += 1

        date_feature_train = date_feature.iloc[:date_splitting_point + 1]
        date_feature_test = date_feature.iloc[date_splitting_point - input_range + 1: date_splitting_point + prediction_time + 1]

    # MAPE: only taking the first day into consideration
    # number_of_test_data, number_of_prediction_days = np.array(y_true).shape
    y_true = np.array(y_true)[:, 0] #所有第一天的y true
    y_pred = np.array(y_pred)[:, 0]
    y_pred = np.round(y_pred, 0)

    from sklearn.metrics import mean_absolute_percentage_error

    mape = mean_absolute_percentage_error(y_true + 1, y_pred + 1)
    print(mape)