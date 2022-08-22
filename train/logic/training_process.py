import os
import random
from typing import Optional, List

import numpy as np
import pandas as pd
import tensorflow as tf

from train.logic.model_selection import cross_validation


def training_process(input_range: int, prediction_time: int, date_feature: pd.DataFrame,
                     numerical_features, categorical_features, n_splits: int,
                     max_train_size: int, test_size, batch_size: int, learning_rate: float,
                     model_type: str, encoder_lstm_units_0: int, loss: str='mse', dropout: float=0,
                     recurrent_dropout: float=0, **kwargs):

    tf.random.set_seed(42)
    os.environ['PYTHONHASHSEED']='42'
    random.seed(42)
    np.random.seed(42)

    date_feature_copy = date_feature.copy()

    y_true, y_pred = cross_validation(date_feature=date_feature_copy, input_range=input_range,
                                      prediction_time=prediction_time, n_splits=n_splits, test_size=test_size,
                                      max_train_size=max_train_size, numerical_features=numerical_features,
                                      categorical_features=categorical_features,
                                      encoder_lstm_units_0=encoder_lstm_units_0,
                                      loss=loss, batch_size=batch_size, learning_rate=learning_rate,
                                      model_type=model_type, dropout=dropout,
                                      recurrent_dropout=recurrent_dropout, **kwargs)

    return y_true, y_pred

def training_process_opt(input_range: int, prediction_time: int, date_feature: pd.DataFrame,
                         numerical_features, categorical_features, n_splits: int,
                         max_train_size: int, test_size, batch_size, learning_rate, model_type: str,
                         loss: str='mse', encoder_filters_0=None, encoder_lstm_units_0=None,
                         encoder_lstm_units_1=None, decoder_lstm_units_0=None, decoder_lstm_units_1=None,
                         decoder_dense_units=None, recurrent_dropout=0.0, dropout=0.0):

    # For hyperparameter optimization

    max_train_size = int(max_train_size)
    batch_size = int(batch_size)
    if encoder_lstm_units_0:
        encoder_lstm_units_0 = int(encoder_lstm_units_0)
    if encoder_lstm_units_1:
        encoder_lstm_units_1=int(encoder_lstm_units_1)
    if encoder_filters_0:
        encoder_filters_0 = int(encoder_filters_0)
    if decoder_lstm_units_0:
        decoder_lstm_units_0=int(decoder_lstm_units_0)
    if decoder_lstm_units_1:
        decoder_lstm_units_1=int(decoder_lstm_units_1)
    if decoder_dense_units:
        decoder_dense_units=int(decoder_dense_units)

    y_true, y_pred = training_process(input_range=input_range, prediction_time=prediction_time,
                                      date_feature=date_feature, numerical_features=numerical_features,
                                      categorical_features=categorical_features, n_splits=n_splits,
                                      max_train_size=max_train_size,  test_size=test_size, loss=loss,
                                      batch_size=batch_size, learning_rate=learning_rate, model_type=model_type,
                                      dropout=dropout, recurrent_dropout=recurrent_dropout,
                                      encoder_lstm_units_0=encoder_lstm_units_0,
                                      encoder_lstm_units_1=encoder_lstm_units_1,
                                      encoder_filters_0=encoder_filters_0,
                                      decoder_lstm_units_0=decoder_lstm_units_0,
                                      decoder_lstm_units_1=decoder_lstm_units_1,
                                      decoder_dense_units=decoder_dense_units,)

    mape_list = []

    for ix in range(n_splits):
        diff = abs((y_true[ix, :, :, 0] - y_pred[ix, :, :, 0]) / y_true[ix, :, :, 0])

        for iy in range(test_size):
            diff[iy][np.isinf(diff[iy])] = np.nan

        mape_list.append(diff)

    mape = np.nanmean(np.concatenate(mape_list))

    return -mape
