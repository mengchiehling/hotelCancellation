from typing import Optional

import numpy as np
import pandas as pd

from train.logic.model_selection import cross_validation


def training_process(input_range: int, prediction_time: int, date_feature: pd.DataFrame,
                     numerical_features, n_splits: int, max_train_size: int, test_size, batch_size: int,
                     learning_rate: float, model_type: str, loss: str='mse', dropout: float=0,
                     recurrent_dropout: float=0, **kwargs):

    date_feature_copy = date_feature.copy()

    y_true, y_pred = cross_validation(date_feature=date_feature_copy, input_range=input_range,
                                      prediction_time=prediction_time, n_splits=n_splits, test_size=test_size,
                                      max_train_size=max_train_size, numerical_features=numerical_features,
                                      loss=loss, batch_size=batch_size, learning_rate=learning_rate,
                                      model_type=model_type, dropout=dropout,
                                      recurrent_dropout=recurrent_dropout, **kwargs)

    return y_true, y_pred


def training_process_opt(input_range: int, prediction_time: int, date_feature: pd.DataFrame,
                         numerical_features, n_splits: int,
                         max_train_size: int, test_size, batch_size, learning_rate, model_type: str,
                         loss: str='mse', encoder_filters: Optional[int]=None, encoder_lstm_units: Optional[int]=None,
                         decoder_lstm_units: Optional[int]=None, decoder_dense_units=None, recurrent_dropout=0.0,
                         dropout=0.0, l2: float=0.0, momentum: float=0.99):

    # For hyperparameter optimization

    max_train_size = int(np.round(max_train_size, 0))
    batch_size = int(np.round(batch_size, 0))
    if encoder_lstm_units:
        encoder_lstm_units = int(np.round(encoder_lstm_units, 0))
    if encoder_filters:
        encoder_filters = int(np.round(encoder_filters, 0))
    if decoder_lstm_units:
        decoder_lstm_units=int(np.round(decoder_lstm_units, 0))
    if decoder_dense_units:
        decoder_dense_units=int(np.round(decoder_dense_units, 0))

    y_true, y_pred = training_process(input_range=input_range, prediction_time=prediction_time,
                                      date_feature=date_feature, numerical_features=numerical_features,
                                      n_splits=n_splits, max_train_size=max_train_size,  test_size=test_size, loss=loss,
                                      batch_size=batch_size, learning_rate=learning_rate, model_type=model_type,
                                      dropout=dropout, recurrent_dropout=recurrent_dropout,
                                      encoder_lstm_units=encoder_lstm_units,
                                      encoder_filters=encoder_filters,
                                      decoder_lstm_units=decoder_lstm_units,
                                      decoder_dense_units=decoder_dense_units,
                                      l2=l2, momentum=momentum)

    absolute_percentage_error = []

    for ix in range(n_splits):
        # dimension: n_splits, test_size, time, dense_units
        # optimized to first day
        diff = abs((y_true[ix, :, 0, 0] - y_pred[ix, :, 0, 0]) / (y_true[ix, :, 0, 0]+1) )

        absolute_percentage_error.append(diff)

    absolute_percentage_error = np.array(absolute_percentage_error)

    absolute_percentage_error[np.isinf(absolute_percentage_error)] = np.nan

    mape = np.nanmean(np.concatenate(absolute_percentage_error))

    if np.isnan(mape):
        mape = 10

    #symmetric_difference = abs(y_true[:, :, :, 0] - y_pred[:, :, :, 0])/(abs(y_true[:, :, :, 0]) + abs(y_pred[:, :, :, 0]))/2

    #return -symmetric_difference.mean()

    return -mape
