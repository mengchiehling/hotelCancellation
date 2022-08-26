import importlib
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from train.logic_fb.data_preparation import to_supervised, parse_tf_input


def generate_holiday_df(date_feature: pd.DataFrame):

    bank_holiday_condition = (~date_feature['week'].isin(['SAT', 'SUN'])) & (date_feature['is_rest_day'] == 1)

    bank_holiday = pd.DataFrame({
        'holiday': 'holiday',
        'ds': pd.to_datetime(date_feature[bank_holiday_condition].index.tolist()),
        'lower_window': -1,
        'upper_window': 1
    })

    weekend_condition = date_feature['week'].isin(['SAT', 'SUN'])

    weekend = pd.DataFrame({
        'holiday': 'weekend',
        'ds': pd.to_datetime(date_feature[weekend_condition].index.tolist()),
        'lower_window': -1,
        'upper_window': 1
    })

    holidays = pd.concat((bank_holiday, weekend))

    return holidays


def prophet_decomposition(date_feature: pd.DataFrame):

    date_index = date_feature.index
    date_index = [index.replace("/", "-") for index in date_index]
    date_feature.index = date_index

    holidays = generate_holiday_df(date_feature)

    cancel_df = date_feature[['canceled']]
    cancel_df.rename(columns={'canceled': 'y'}, inplace=True)

    cancel_df.index.name = 'ds'
    cancel_df.reset_index(inplace=True)

    model = Prophet(holidays=holidays, weekly_seasonality=False, seasonality_mode='multiplicative')

    model.fit(cancel_df)

    forecast = model.predict(cancel_df[['ds']])

    return forecast['yhat'].values, model


def data_preparation(date_feature: pd.DataFrame, label_column: str):

    num_feature_columns = [label_column]

    for time_diff in [7, 14, 21, 28]:  # has to be replaced
        c = f'diff_{time_diff}'
        date_feature[c] = (date_feature[label_column] - date_feature[label_column].shift(time_diff)).fillna(0)
        num_feature_columns.append(c)

    return num_feature_columns, date_feature


def model_training(date_feature: pd.DataFrame, test_size: int, input_range: int, prediction_time: int,
                   learning_rate: float, batch_size: int, model_type: str, loss: str='mse',
                   lead_time: int=0, dropout: float=0, recurrent_dropout: float=0, **kwargs):

    label_column = 'canceled_label'

    # data preparation
    numerical_features, date_feature = data_preparation(date_feature, label_column=label_column)

    df_train, df_val = train_test_split(date_feature, test_size=test_size, shuffle=False)

    df_val = pd.concat([df_train.iloc[-(input_range + lead_time + prediction_time):], df_val])

    # data normalization

    feature_scaler = MinMaxScaler()  # default to 0 - 1
    target_scaler = MinMaxScaler()

    target_scaler.fit(df_train[[label_column]])

    df_train.loc[:, numerical_features] = feature_scaler.fit_transform(df_train[numerical_features])
    df_val.loc[:, numerical_features] = feature_scaler.transform(df_val[numerical_features])

    results_train = to_supervised(df_train, input_range=input_range, prediction_time=prediction_time,
                                  numerical_features=numerical_features)

    results_val = to_supervised(df_val, input_range=input_range, prediction_time=prediction_time,
                                numerical_features=numerical_features)

    _, n_inputs, n_features = results_train['encoder_X_num'].shape
    _, n_outputs, _ = results_train['y_label'].shape

    assert model_type in ['LSTM2LSTM', 'CNN2LSTM']

    m = importlib.import_module(f"train.logic_fb.model.{model_type}_architecture")

    model_tf = m.build_model(n_inputs=n_inputs, n_features=n_features, dropout=dropout, n_outputs=n_outputs,
                             recurrent_dropout=recurrent_dropout, **kwargs)

    # we can have customized optimizer as well

    optimizer = Adam(learning_rate=learning_rate)

    model_tf.compile(loss=loss, optimizer=optimizer)  # replace model.compile(loss=loss, optimizer='Adam')

    X_train, y_train = parse_tf_input(results_train)
    X_val, y_val = parse_tf_input(results_val)

    earlystopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    callbacks = [earlystopping]

    model_tf.fit(X_train, {'outputs': y_train['outputs']}, epochs=20, batch_size=batch_size, verbose=0,
              validation_data=(X_val, {'outputs': y_val['outputs']}), shuffle=True, callbacks=callbacks)

    return model_tf, feature_scaler, target_scaler # scaler is also a part of the model


def cross_validation(date_feature: pd.DataFrame, n_splits: int, test_size: int, input_range: int,
                     prediction_time: int, max_train_size: int, batch_size: int, learning_rate: float,
                     model_type: str, loss: str='mse', lead_time: int=0, dropout: float=0, recurrent_dropout: float=0,
                     **kwargs):

    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, max_train_size=max_train_size)

    y_pred = []
    y_true = []

    for n_fold, (train_index, test_index) in enumerate(tscv.split(date_feature)):

        df_train_full = date_feature.iloc[:train_index[-1]+1]

        prophet_fitted_values, model_prophet = prophet_decomposition(df_train_full)

        df_train_full['yhat'] = prophet_fitted_values
        df_train_full['canceled_label'] = df_train_full['canceled'] - df_train_full['yhat']

        model_tf, feature_scaler, target_scaler = model_training(date_feature=df_train_full.iloc[train_index],
                                                                 test_size=test_size, input_range=input_range,
                                                                 prediction_time=prediction_time, loss=loss,
                                                                 learning_rate=learning_rate, batch_size=batch_size,
                                                                 model_type=model_type, dropout=dropout,
                                                                 recurrent_dropout=recurrent_dropout, **kwargs)

        future = model_prophet.make_future_dataframe(periods=test_size)
        pred_prophet = model_prophet.predict(future)

        date_feature_prep = date_feature.iloc[:test_index[-1]+1]
        date_feature_prep['yhat'] = pred_prophet['yhat'].values
        date_feature_prep['canceled_label'] = date_feature_prep['canceled'] - date_feature_prep['yhat']

        numerical_features, date_feature_prep = data_preparation(date_feature_prep, label_column='canceled_label')

        time_begin = date_feature_prep.iloc[train_index].index[0]
        time_end = date_feature_prep.iloc[train_index].index[-1]

        test_index = np.arange(test_index[0] - input_range - lead_time - prediction_time, test_index[-1] + 1)

        df_test = date_feature_prep.iloc[test_index]

        print(f"fold {n_fold}: training: {time_begin} - {time_end}, testing: {df_test.index[0]} - {df_test.index[-1]}")

        df_test.loc[:, numerical_features] = feature_scaler.transform(df_test[numerical_features])

        results_test = to_supervised(df_test, input_range=input_range, prediction_time=prediction_time,
                                     numerical_features=numerical_features)

        X_test, y_test = parse_tf_input(results_test)

        y_true.append(y_test['true'])

        pred = model_tf.predict(X_test)
        pred = np.array([target_scaler.inverse_transform(x) for x in pred]) + y_test['yhat']

        y_pred.append(pred)

    return np.array(y_true), np.array(y_pred)