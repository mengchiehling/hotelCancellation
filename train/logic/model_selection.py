import importlib
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from train.logic.model.LSTM2LSTM_architecture import build_model
from train.logic.data_preparation import tf_input_pipeline


def feature_normalization(df_train, df_val, features: List):

    scaler = MinMaxScaler()

    df_train.loc[:, features] = scaler.fit_transform(df_train[features])

    df_val.loc[:, features] = scaler.transform(df_val[features])

    return df_train, df_val, scaler


def model_training(model, X_train, y_train, X_val, y_val, batch_size, learning_rate: float, loss: str):

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(loss=loss, optimizer=optimizer)  # replace model.compile(loss=loss, optimizer='Adam')

    earlystopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    callbacks = [earlystopping]
    # 原本 epochs 是設置20
    model.fit(X_train, {'outputs': y_train['outputs']}, epochs=20, batch_size=batch_size, verbose=0,
              validation_data=(X_val, {'outputs': y_val['outputs']}), shuffle=True, callbacks=callbacks)

    return model


def model_training_pipeline(date_feature: pd.DataFrame, test_size: int, input_range: int, prediction_time: int,
                            numerical_features, learning_rate: float, batch_size: int, model_type: str, loss: str='mse',
                            lead_time: int=0, dropout: float=0, recurrent_dropout: float=0, **kwargs):

    '''
    model_training_pipeline()

    :param date_feature:
    :param test_size:
    :param input_range:
    :param prediction_time:
    :param numerical_features:
    :param learning_rate:
    :param batch_size:
    :param model_type:
    :param loss:
    :param lead_time:
    :param dropout:
    :param recurrent_dropout:
    :param kwargs:
    :return:
    '''


    df_train, df_val = train_test_split(date_feature, test_size=test_size, shuffle=False)

    df_val = pd.concat([df_train.iloc[-(input_range + lead_time + prediction_time):], df_val])

    df_train, df_val, scaler = feature_normalization(df_train, df_val, numerical_features)

    X_train, y_train = tf_input_pipeline(df_train, input_range=input_range, prediction_time=prediction_time,
                                         numerical_features=numerical_features)

    X_val, y_val = tf_input_pipeline(df_val, input_range=input_range, prediction_time=prediction_time,
                                         numerical_features=numerical_features)

    _, n_inputs, n_features = X_train['encoder_X_num'].shape
    _, n_outputs, _ = y_train['outputs'].shape

    assert model_type in ['LSTM2LSTM', 'CNN2LSTM', 'BiLSTM2LSTM']

    m = importlib.import_module(f"train.logic.model.{model_type}_architecture")

    model = m.build_model(n_inputs=n_inputs, n_features=n_features, dropout=dropout,
                          recurrent_dropout=recurrent_dropout, n_outputs=n_outputs, **kwargs)

    # we can have customized optimizer as well

    model = model_training(model, X_train, y_train, X_val, y_val, batch_size=batch_size,
                           learning_rate=learning_rate, loss=loss)

    return model, scaler


def cross_validation(date_feature: pd.DataFrame, n_splits: int, test_size: int, input_range: int,
                     prediction_time: int, max_train_size: int, numerical_features: List,
                     batch_size: int, learning_rate: float, model_type: str, loss: str='mse',
                     lead_time: int=0, dropout: float=0, recurrent_dropout: float=0,
                     **kwargs):

    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, max_train_size=max_train_size)

    y_pred = []
    y_true = []

    for n_fold, (train_index, test_index) in enumerate(tscv.split(date_feature)):

        #若要訓練集與測試集的日期不重疊的話就註解掉下面這行
        test_index = np.arange(test_index[0] - input_range - lead_time - prediction_time, test_index[-1] + 1)

        df_train = date_feature.iloc[train_index]

        # Apply rescaling:
        # https://stackoverflow.com/questions/43467597/should-i-normalize-my-features-before-throwing-them-into-rnn
        # It might help improve the performance of the model

        time_begin = df_train.index[0]
        time_end = df_train.index[-1]

        df_test = date_feature.iloc[test_index]

        print(f"fold {n_fold}: training: {time_begin} - {time_end}, testing: {df_test.index[0]} - {df_test.index[-1]}")

        model, scaler = model_training_pipeline(date_feature=df_train, test_size=test_size, input_range=input_range,
                                                prediction_time=prediction_time, numerical_features=numerical_features,
                                                loss=loss, learning_rate=learning_rate, batch_size=batch_size,
                                                model_type=model_type, dropout=dropout,
                                                recurrent_dropout=recurrent_dropout, **kwargs)

        df_test.loc[:, numerical_features] = scaler.transform(df_test[numerical_features])

        X_test, y_test = tf_input_pipeline(df_test, input_range=input_range, prediction_time=prediction_time,
                                           numerical_features=numerical_features)
        y_true_extend = np.repeat(y_test['true'].reshape(-1, 1), len(scaler.scale_), axis=1)
        y_true_reshape = scaler.inverse_transform(y_true_extend)[:, 0].reshape(y_test['true'].shape)
        y_true.append(y_true_reshape)

        pred = model.predict(X_test)

        y_pred_extend = np.repeat(pred.reshape(-1, 1), len(scaler.scale_), axis=1)
        y_pred_reshape = scaler.inverse_transform(y_pred_extend)[:, 0].reshape(pred.shape)
        y_pred.append(y_pred_reshape)

    return np.array(y_true), np.array(y_pred)