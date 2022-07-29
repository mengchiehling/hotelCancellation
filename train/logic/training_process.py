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
                                      loss=loss)

    return y_true, y_pred
