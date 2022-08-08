import os
import random
from typing import Optional, List

import numpy as np
import pandas as pd
import tensorflow as tf

from train.logic.model_selection import cross_validation


def training_process(input_range: int, prediction_time: int, date_feature: pd.DataFrame,
                     numerical_features, categorical_features, n_splits: int,
                     max_train_size: int, encoder_lstm_units, decoder_dense_units, test_size,
                     batch_size: int, learning_rate: float, model_type: str, model_name: str,
                     decoder_lstm_units: Optional[List]=None, loss: str='mse'):

    tf.random.set_seed(42)
    os.environ['PYTHONHASHSEED']='42'
    random.seed(42)
    np.random.seed(42)

    encoder_lstm_units = [int(lstm_units) for lstm_units in encoder_lstm_units]

    if decoder_lstm_units:
        decoder_lstm_units = [int(lstm_units) for lstm_units in decoder_lstm_units]

    decoder_dense_units = [int(units) for units in decoder_dense_units]

    date_feature_copy = date_feature.copy()

    y_true, y_pred = cross_validation(date_feature=date_feature_copy, input_range=input_range,
                                      prediction_time=prediction_time, n_splits=n_splits, test_size=test_size,
                                      max_train_size=max_train_size, numerical_features=numerical_features,
                                      categorical_features=categorical_features, encoder_lstm_units=encoder_lstm_units,
                                      decoder_dense_units=decoder_dense_units, decoder_lstm_units=decoder_lstm_units,
                                      loss=loss, batch_size=batch_size, learning_rate=learning_rate,
                                      model_type=model_type, model_name=model_name)

    return y_true, y_pred

def training_process_opt(input_range: int, prediction_time: int, date_feature: pd.DataFrame,
                         numerical_features, categorical_features, n_splits: int,
                         max_train_size: int, encoder_lstm_units, decoder_dense_units, test_size,
                         batch_size, learning_rate, model_type: str, model_name: str,
                         decoder_lstm_units: Optional[List]=None, loss: str='mse'):

    # For hyperparameter optimization

    max_train_size = int(max_train_size)
    batch_size = int(batch_size)

    y_true, y_pred = training_process(input_range=input_range, prediction_time=prediction_time,
                                      date_feature=date_feature, numerical_features=numerical_features,
                                      categorical_features=categorical_features, n_splits=n_splits,
                                      max_train_size=max_train_size, encoder_lstm_units=encoder_lstm_units,
                                      decoder_dense_units=decoder_dense_units, test_size=test_size,
                                      decoder_lstm_units=decoder_lstm_units, loss=loss, batch_size=batch_size,
                                      learning_rate=learning_rate, model_name=model_name, model_type=model_type)

    mape_list = []

    for ix in range(n_splits):
        diff = abs((y_true[ix, :, :, 0] - y_pred[ix, :, :, 0]) / y_true[ix, :, :, 0])

        for iy in range(test_size):
            diff[iy][np.isinf(diff[iy])] = np.nan

        mape_list.append(diff)

    mape = np.nanmean(np.concatenate(mape_list))

    return -mape
