import argparse
import os
import joblib
from functools import partial

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src import config
from src.io.path_definition import get_file, load_yaml_file, get_datafetch
from src.common.load_data import data_preparation
from src.io.load_parameters import load_optimized_parameters
from train.common.model_selection import cross_validation, model_training_pipeline
from train.common.optimization_process import optimization_process


def labelencoding(df: pd.DataFrame, column: str):

    le = LabelEncoder()
    df.loc[:, column] = le.fit_transform(df[column])

    return df


def export_final_model(dataset_, evaluation: bool = False):
    # 儲存模型

    params, _ = load_optimized_parameters()

    model, scaler = model_training_pipeline(dataset_, **params)

    dir_ = os.path.join(get_datafetch(), 'model')
    if not os.path.isdir(dir_):
        os.makedirs(dir_)

    basic_filename = os.path.join(dir_, f"{config.algorithm}_{config.configuration}_{config.hotel_id}")

    if evaluation:
        filename_ = basic_filename + "_evaluation.sav"
    else:
        filename_ = basic_filename + ".sav"

    joblib.dump(model, filename_)


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
    # features_configuration = load_yaml_file(get_file(os.path.join('config', 'training_config.yml')))['features_configuration'][args.configuration]
    #
    # for key, values in features_configuration.items():
    #     config.features_configuration[key] = values


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--hotel_id', type=int, help='id of hotel, should exists in cancel_dataset_target.csv')
    parser.add_argument('--configuration', type=str, help='A')
    parser.add_argument('--algorithm', type=str)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--test_size', type=int, default=28)
    parser.add_argument('--max_train_size', type=int, default=180)
    parser.add_argument('--input_range', type=int, default=28)
    parser.add_argument('--prediction_time', type=int, default=7)
    parser.add_argument('--weekly_inputs', action='store_true')

    args = parser.parse_args()

    set_configuration()

    df = data_preparation(args.hotel_id, remove_business_booking=True)

    file_path = get_file(os.path.join('config', 'training_config.yml'))
    metadata = load_yaml_file(file_path)

    features_configuration = metadata['feature_configuration'][args.configuration]

    config.categorical_features = features_configuration['categorical']
    config.numerical_features = features_configuration['numerical']

    # for encoded_column in categorical_features:
    #     date_feature = labelencoding(date_feature, encoded_column)

    pbounds = metadata[f'{args.algorithm.lower()}_pbounds']
    for key, value in pbounds.items():
        pbounds[key] = eval(value)

    cross_validation_fn = partial(cross_validation, date_feature=df, loss='mse')

    optimization_process(cross_validation_fn, pbounds)
