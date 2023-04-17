import importlib
import os
from typing import List, Optional
from functools import partial

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from train.common.data_preparation import tf_input_pipeline

from src import config
from src.logic.common.functions import generate_weekly_inputs


def feature_normalization(df_train, df_val):

    features = config.numerical_features

    scaler = MinMaxScaler()

    df_train.loc[:, features] = scaler.fit_transform(df_train[features])

    df_val.loc[:, features] = scaler.transform(df_val[features])

    return df_train, df_val, scaler


def model_training(model, x_train, y_train, x_val, y_val, batch_size, learning_rate: float, loss: str):

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(loss=loss, optimizer=optimizer)  # replace model.compile(loss=loss, optimizer='Adam')

    earlystopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    callbacks = [earlystopping]

    if config.weekly_inputs:
        # use 28 days canceled to build the 7 days mean field prediction as the inputs for decoder
        x_train = generate_weekly_inputs(x_train, y_train)
        x_val = generate_weekly_inputs(x_val, y_train)

    history = model.fit(x_train, {'outputs': y_train['outputs']}, epochs=20, batch_size=batch_size, verbose=1,
                        validation_data=(x_val, {'outputs': y_val['outputs']}), shuffle=True, callbacks=callbacks)

    return model


def model_training_pipeline(df: pd.DataFrame, loss='mse', **kwargs):

    """

    :param df:
    :param loss:
    :param kwargs:
    :return:
    """

    algorithm = config.algorithm

    df_train, df_val = train_test_split(df, test_size=config.test_size, shuffle=False)

    df_val = pd.concat([df_train.iloc[-(config.input_range + config.lead_time + config.prediction_time):], df_val])

    df_train, df_val, scaler = feature_normalization(df_train, df_val)

    tf_input_fn = partial(tf_input_pipeline)

    X_train, y_train = tf_input_fn(df_train)

    X_val, y_val = tf_input_fn(df_val)

    _, n_inputs, n_features = X_train['encoder_X_num'].shape
    _, n_outputs, _ = y_train['outputs'].shape

    assert config.algorithm in ['LSTM2LSTM', 'CNN2LSTM', 'BiLSTM2LSTM'], f"{config.algorithm} is not within 'LSTM2LSTM', 'CNN2LSTM', 'BiLSTM2LSTM'"

    m = importlib.import_module(f"train.logic.model.{algorithm}_architecture")

    if len(config.categorical_features) == 0:
        encoder_cat_dict = None
        decoder_cat_dict = None
    else:
        encoder_cat_dict = {}
        decoder_cat_dict = {}
        for c in config.categorical_features:
            encoder_cat_dict[f'{c}_encoder'] = X_train[f'{c}_encoder']
            decoder_cat_dict[f'{c}_decoder'] = X_train[f'{c}_decoder']

    model = m.build_model(n_inputs=n_inputs, n_features=n_features, n_outputs=n_outputs,
                          encoder_cat_dict=encoder_cat_dict, decoder_cat_dict=decoder_cat_dict,
                          **kwargs)

    batch_size = int(kwargs.get('batch_size', 4))
    learning_rate = kwargs.get('learning_rate', 0.001)

    model = model_training(model, X_train, y_train, X_val, y_val, batch_size=batch_size,
                           learning_rate=learning_rate, loss=loss)

    return model, scaler


def cross_validation(date_feature: pd.DataFrame, loss: str = 'mse', **kwargs):

    numerical_features = config.numerical_features
    categorical_features = config.categorical_features

    tscv = TimeSeriesSplit(n_splits=config.n_splits, test_size=config.test_size,
                           max_train_size=config.max_train_size)

    y_pred = []
    y_true = []

    for n_fold, (train_index, test_index) in enumerate(tscv.split(date_feature)):
        test_index = np.arange(test_index[0] - config.input_range - config.lead_time - config.prediction_time,
                               test_index[-1] + 1)

        df_train = date_feature.iloc[train_index]

        # Apply rescaling:
        # https://stackoverflow.com/questions/43467597/should-i-normalize-my-features-before-throwing-them-into-rnn
        # It might help improve the performance of the model

        time_begin = df_train.index[0]
        time_end = df_train.index[-1]

        df_test = date_feature.iloc[test_index]

        print(f"fold {n_fold}: training: {time_begin} - {time_end}, testing: {df_test.index[0]} - {df_test.index[-1]}")

        model, scaler = model_training_pipeline(df=df_train, loss=loss, **kwargs)

        df_test.loc[:, numerical_features] = scaler.transform(df_test[config.numerical_features])

        X_test, y_test = tf_input_pipeline(df_test)

        y_true_extend = np.repeat(y_test['true'].reshape(-1, 1), len(scaler.scale_), axis=1)
        y_true_reshape = scaler.inverse_transform(y_true_extend)[:, 0].reshape(y_test['true'].shape)
        y_true.append(y_true_reshape)

        X_test = generate_weekly_inputs(X_test, y_test)

        pred = model.predict(X_test)

        y_pred_extend = np.repeat(pred.reshape(-1, 1), len(scaler.scale_), axis=1)
        y_pred_reshape = np.round(scaler.inverse_transform(y_pred_extend)[:, 0].reshape(pred.shape))
        y_pred.append(y_pred_reshape)

    return np.array(y_true), np.array(y_pred)
