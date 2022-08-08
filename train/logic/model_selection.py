import importlib
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from train.logic.model.LSTM_architecture import build_model
from train.logic.data_preparation import to_supervised, parse_tf_input


def model_training(date_feature: pd.DataFrame, test_size: int, input_range: int, prediction_time: int,
                   numerical_features, categorical_features, encoder_lstm_units: List[int],
                   decoder_dense_units: List[int], category_input_dim: Dict, learning_rate: float,
                   batch_size: int, model_type: str, model_name: str, loss: str='mse',
                   decoder_lstm_units: Optional[List]=None, lead_time: int=0):

    df_train, df_val = train_test_split(date_feature, test_size=test_size, shuffle=False)

    df_val = pd.concat([df_train.iloc[-(input_range + lead_time + prediction_time):], df_val])

    results_train = to_supervised(df_train, input_range=input_range, prediction_time=prediction_time,
                                  numerical_features=numerical_features, categorical_features=categorical_features)

    for c, v in category_input_dim.items():
        results_train['decoder_X_cat'][c]['input_dim'] = v

    results_val = to_supervised(df_val, input_range=input_range, prediction_time=prediction_time,
                                numerical_features=numerical_features, categorical_features=categorical_features)

    _, n_inputs, n_features = results_train['encoder_X_num'].shape

    assert model_type in ['LSTM', 'CNNBiLSTM']
    assert model_name in ['', 'cAttention', 'LuongAttention']
    # model architecture according to model_name
    if model_name == "":
        m = importlib.import_module(f"train.logic.model.{model_type}_architecture")
    else:
        m = importlib.import_module(f"train.logic.model.{model_type}_{model_name}_architecture")

    model = m.build_model(n_inputs=n_inputs, n_features=n_features, decoder_cat_dict=results_train['decoder_X_cat'],
                          encoder_lstm_units=encoder_lstm_units, decoder_dense_units=decoder_dense_units,
                          decoder_lstm_units=decoder_lstm_units)

    # we can have customized optimizer as well

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(loss=loss, optimizer=optimizer)  # replace model.compile(loss=loss, optimizer='Adam')

    X_train, y_train = parse_tf_input(results_train)
    X_val, y_val = parse_tf_input(results_val)

    earlystopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    callbacks = [earlystopping]

    model.fit(X_train, {'outputs': y_train['outputs']}, epochs=20, batch_size=batch_size, verbose=1,
              validation_data=(X_val, {'outputs': y_val['outputs']}), shuffle=True, callbacks=callbacks)

    return model


def cross_validation(date_feature: pd.DataFrame, n_splits: int, test_size: int, input_range: int, prediction_time: int,
                     max_train_size: int, numerical_features: List, categorical_features: List,
                     encoder_lstm_units: List[int], decoder_dense_units: List[int], batch_size: int,
                     learning_rate: float, model_type: str, model_name: str, loss: str='mse',
                     decoder_lstm_units: Optional[List]=None, lead_time: int=0):

    category_input_dim = {c: len(np.unique(date_feature[c].values)) for c in categorical_features}

    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, max_train_size=max_train_size)

    y_pred = []
    y_true = []

    for n_fold, (train_index, test_index) in enumerate(tscv.split(date_feature)):
        test_index = np.arange(test_index[0] - input_range - lead_time - prediction_time, test_index[-1] + 1)

        df_train = date_feature.iloc[train_index]

        time_begin = df_train.index[0]
        time_end = df_train.index[-1]

        print(f"fold {n_fold}: {time_begin} - {time_end}")

        df_test = date_feature.iloc[test_index]

        results_test = to_supervised(df_test, input_range=input_range, prediction_time=prediction_time,
                                     numerical_features=numerical_features, categorical_features=categorical_features)
        X_test, y_test = parse_tf_input(results_test)

        y_true.append(y_test['true'])

        model = model_training(date_feature=df_train, test_size=test_size, input_range=input_range,
                               prediction_time=prediction_time, encoder_lstm_units=encoder_lstm_units,
                               decoder_dense_units=decoder_dense_units, decoder_lstm_units=decoder_lstm_units,
                               category_input_dim=category_input_dim, numerical_features=numerical_features,
                               categorical_features=categorical_features, loss=loss, learning_rate=learning_rate,
                               batch_size=batch_size, model_type=model_type, model_name=model_name)

        pred = model.predict(X_test)

        y_pred.append(pred)

    return np.array(y_true), np.array(y_pred)