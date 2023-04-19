from functools import partial
from typing import Optional
import argparse, os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from src import config
from src.io.path_definition import get_file, load_yaml_file, get_datafetch
from src.common.load_data import load_training_data
from src.common.tools import timeseries_train_test_split
from src.io.load_parameters import load_optimized_parameters
from train.common.model_selection import cross_validation, model_training_pipeline
from train.common.optimization_process import optimization_process



def labelencoding(df, column: str):

    le = LabelEncoder()
    df.loc[:, column] = le.fit_transform(df[column])

    return df


def create_dataset(dataset_: pd.DataFrame, test_size):
    # 等同於網路上的train_test_split步驟
    y = dataset_['label']

    df_hotel = dataset_.copy()

    train_dataset_, eval_dataset, train_target_, eval_target = timeseries_train_test_split(df_hotel)

    return train_dataset_, eval_dataset, train_target_, eval_target


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


def to_timeseries_dataframe(df_: pd.DataFrame, idx_: pd.Series) -> pd.DataFrame:

    """

    :param df_: PMS data
    :param idx_: range of timestamps
    :return:
    """

    # Simple time series, without extra features.
    booking_feature = df_.groupby(by="check_in").agg(canceled=('label', 'sum'),
                                                     booking=('label', 'count'),
                                                     holiday=('holiday', np.unique))

    booking_feature = booking_feature.reindex(idx_, fill_value=0)

    return booking_feature


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

    file_path = get_file(os.path.join('config', 'training_config.yml'))
    metadata = load_yaml_file(file_path)

    features_configuration = metadata['feature_configuration'][args.configuration]

    config.categorical_features = features_configuration['categorical']
    config.numerical_features = features_configuration['numerical']

    df, idx = load_training_data(args.hotel_id, remove_business_booking=True)

    for encoded_column in config.categorical_features:
        df = labelencoding(df, encoded_column)

    train_dataset, test_dataset, _, _ = create_dataset(df, test_size=args.test_size)

    end_of_train_dataset = idx.index(train_dataset.iloc[-1]['check_in'])  # index

    test_dataset = to_timeseries_dataframe(test_dataset, idx[:end_of_train_dataset+1])
    # train_dataset = to_timeseries_dataframe(train_dataset, idx)






    y_true, y_pred = training_process(input_range=input_range, prediction_time=prediction_time,
                                      date_feature=date_feature, numerical_features=numerical_features,categorical_features=categorical_features,
                                      n_splits=n_splits,max_train_size=365, test_size=test_size, model_type=model_type, loss='mse', **params)


    adapted_mape = mean_absolute_percentage_error(y_true.flatten()+1, y_pred.flatten()+1)

    print(adapted_mape)