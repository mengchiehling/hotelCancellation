import argparse
import os
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
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

    # scaler = None  # put the scaler here
    dir_ = os.path.join(get_datafetch(), 'model')
    basic_filename = f"{config.algorithm}_{config.configuration}_{config.hotel_id}_evaluation"
    scaler_filename = os.path.join(dir_, basic_filename + "_scaler.joblib")
    scaler = joblib.load(scaler_filename)

    test_dataset.loc[:, config.numerical_features] = scaler.transform(test_dataset[config.numerical_features])

    X_test, y_test = tf_input_pipeline(test_dataset)
    # Ground Truth
    true_postprocessed = prediction_postprocessing(y_test['true'], scaler=scaler)

    y_true = []
    for ix, y in enumerate(true_postprocessed):
        if ix != len(true_postprocessed)-1:
            y_true.append(y[0])
        else:
            y_true.extend(list(y))

    # true_avg = timeseries_prediction_postprocessing(true_postprocessed)

    X_test = generate_weekly_inputs(X_test, y_test)

    model = tf.keras.models.load_model(os.path.join(dir_, f"{basic_filename}"))

    # Prediction
    pred = model.predict(X_test)
    pred_postprocessed = prediction_postprocessing(pred, scaler=scaler)

    y_pred = []
    for ix, y in enumerate(pred_postprocessed):
        if ix != len(pred_postprocessed)-1:
            y_pred.append(y[0])
        else:
            y_pred.extend(list(y))

    # pred_avg = pred_postprocessed[:, 0]
    # pred_avg = timeseries_prediction_postprocessing(pred_postprocessed)

    # 只看一天
    mape2 = mean_absolute_percentage_error(np.array(y_true) + 1, np.array(y_pred) + 1)
    print("第一天的MAPE值: {:.4f}".format(mape2))

    y_abs_diff = np.abs(np.array(y_true) - np.array(y_pred))
    wmape2 = y_abs_diff.sum() / np.array(y_true).sum()
    print("第一天的WMAPE值: {:.4f}".format(wmape2))

    # 整體
    mape1 = mean_absolute_percentage_error(true_postprocessed.flatten() + 1, pred_postprocessed.flatten() + 1)
    print(mape1)
    print("整體的MAPE值: {:.4f}".format(mape1))

    y_abs_diff1 = np.abs(true_postprocessed - pred_postprocessed)
    wmape1 = y_abs_diff1.sum() / true_postprocessed.sum()
    print("整體的WMAPE值: {:.4f}".format(wmape1))

    # 畫第一天
    fig, ax = plt.subplots()
    ax.plot(y_true, color="red", label="The actual number of canceled orders")
    ax.plot(y_pred, color="blue", label="The predict number of canceled orders")
    ax.set_xlabel("Check in date")
    ax.set_ylabel("Canceled orders")
    plt.legend()
    plt.savefig(f"{basic_filename}.png")

    # 印出來第一天的
    filepath = os.path.join(get_datafetch(),
                            f'predictResult(no fill zero)_{config.algorithm}_{config.hotel_id}_{config.configuration}.csv')

    _, test_dataset, _, _ = create_dataset(df, test_size=args.test_size)
    test_dataset = to_timeseries_dataframe(test_dataset, idx[end_of_train_dataset + 1:])
    test_dataset['pred_canceled'] = y_pred

    test_dataset.rename(columns={"canceled": "label", "cabceled_pred": 'time_series_pred'}, inplace=True)
    test_dataset = test_dataset[["label", "time_series_pred"]]
    test_dataset.to_csv(filepath)
