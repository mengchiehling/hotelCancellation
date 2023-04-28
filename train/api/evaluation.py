import argparse
import os

import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error

from src import config
from src.io.path_definition import get_file, load_yaml_file, get_datafetch
from src.common.load_data import load_training_data
from src.common.tools import prediction_postprocessing, timeseries_prediction_postprocessing
from src.logic.common.functions import generate_weekly_inputs
from train.api.training_run import labelencoding, create_dataset, to_timeseries_dataframe
from train.common.data_preparation import tf_input_pipeline


def set_configuration():

    config.hotel_id = args.hotel_id
    config.configuration = args.configuration
    config.algorithm = args.algorithm
    config.n_splits = args.n_splits
    config.test_size = args.test_size
    config.max_train_size = args.max_train_size
    config.input_range = args.input_range
    config.prediction_time = args.prediction_time
    config.weekly_inputs = args.weekly_inputs


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--hotel_id', type=int, help='id of hotel, should exists in cancel_dataset_target.csv')
    parser.add_argument('--configuration', type=str, help='A')
    parser.add_argument('--algorithm', type=str, help="LSTM2LSTM, CNN2LSTM")
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--test_size', type=int, default=28)
    parser.add_argument('--max_train_size', type=int, default=180)
    parser.add_argument('--input_range', type=int, default=28)
    parser.add_argument('--prediction_time', type=int, default=7)
    parser.add_argument('--weekly_inputs', action='store_true')

    args = parser.parse_args()

    set_configuration()

    df, idx = load_training_data(args.hotel_id, remove_business_booking=True)

    file_path = get_file(os.path.join('config', 'training_config.yml'))
    metadata = load_yaml_file(file_path)

    features_configuration = metadata['feature_configuration'][args.configuration]

    config.categorical_features = features_configuration['categorical']
    config.numerical_features = features_configuration['numerical']
    config.future_features = features_configuration.get('future', None)

    for encoded_column in config.categorical_features:
        df = labelencoding(df, encoded_column)

    train_dataset, test_dataset, _, _ = create_dataset(df, test_size=args.test_size)

    end_of_train_dataset = idx.index(train_dataset.iloc[-1]['check_in'])  # index

    train_dataset = to_timeseries_dataframe(train_dataset, idx[:end_of_train_dataset + 1])
    test_dataset = to_timeseries_dataframe(test_dataset, idx[end_of_train_dataset:])

    test_dataset = pd.concat([train_dataset.iloc[-config.input_range:], test_dataset])

    scaler = None  # put the scaler here

    test_dataset.loc[:, config.numerical_features] = scaler.transform(test_dataset[config.numerical_features])

    X_test, y_test = tf_input_pipeline(test_dataset)
    # Ground Truth
    true_postprocessed = prediction_postprocessing(y_test['true'], scaler=scaler)
    true_avg = timeseries_prediction_postprocessing(true_postprocessed)

    X_test = generate_weekly_inputs(X_test, y_test)

    dir_ = os.path.join(get_datafetch(), 'model')
    basic_filename = os.path.join(dir_, f"{config.algorithm}_{config.configuration}_{config.hotel_id}")
    model = None  # the model

    # Prediction
    pred = model.predict(X_test)
    pred_postprocessed = prediction_postprocessing(pred, scaler=scaler)
    pred_avg = timeseries_prediction_postprocessing(pred_postprocessed)

    mape = mean_absolute_percentage_error(true_avg.flatten() + 1, pred_avg.flatten() + 1)