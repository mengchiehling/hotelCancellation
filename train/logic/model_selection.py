import importlib
from typing import List, Dict, Optional
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.feature_selection import SelectKBest, f_regression
from train.logic.model.LSTM2LSTM_architecture import build_model
from train.logic.data_preparation import tf_input_pipeline

from src.io.path_definition import get_file, _load_yaml
from src.logic.common.functions import generate_weekly_inputs


basic_parameters = _load_yaml(get_file(os.path.join('config', 'training_config.yml')))['basic_parameters']


def select_features(X_train, y_train):
    # configure to select all features
    fs = SelectKBest(score_func=f_regression, k='all')
    try:
        fs.fit(X_train, y_train)
    except ValueError:
        print("")
    # transform test input data
    columns = X_train.columns
    pvalues = fs.pvalues_

    kept_columns = []
    threshold = 0.025

    for c, p in zip(columns, pvalues):
        if p < threshold:
            kept_columns.append(c)

    return kept_columns


def feature_normalization(df_train, df_val, features: List):

    scaler = MinMaxScaler()

    df_train.loc[:, features] = scaler.fit_transform(df_train[features])

    df_val.loc[:, features] = scaler.transform(df_val[features])

    return df_train, df_val, scaler


#def model_training(model, X_train, y_train, X_val, y_val, batch_size, learning_rate: float, loss: str):

    #optimizer = Adam(learning_rate=learning_rate)

    #model.compile(loss=loss, optimizer=optimizer)  # replace model.compile(loss=loss, optimizer='Adam')

    #earlystopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    #callbacks = [earlystopping]

    #X_tf_train = {'encoder_X_num': X_train['encoder_X_num']}
    #X_tf_train.update(X_train['encoder_X_cat'])
    #X_tf_train.update(X_train['decoder_X_cat'])

    #X_tf_val = {'encoder_X_num': X_val['encoder_X_num']}
    #X_tf_val.update(X_val['encoder_X_cat'])
    #X_tf_val.update(X_val['decoder_X_cat'])

    #model.fit(X_tf_train, {'outputs': y_train['outputs']}, epochs=20, batch_size=batch_size, verbose=0,
              #validation_data=(X_tf_val, {'outputs': y_val['outputs']}), shuffle=True, callbacks=callbacks)

    #return model


def model_training(model, X_train, y_train, X_val, y_val, batch_size, learning_rate: float, loss: str,
                   weekly_inputs: Optional[bool]=None):

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(loss=loss, optimizer=optimizer)  # replace model.compile(loss=loss, optimizer='Adam')

    earlystopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    callbacks = [earlystopping]

    if weekly_inputs:
        # use 28 days canceled to build the 7 days mean field prediction as the inputs for decoder
        X_train = generate_weekly_inputs(X_train, y_train)
        X_val = generate_weekly_inputs(X_val, y_train)

    model.fit(X_train, {'outputs': y_train['outputs']}, epochs=20, batch_size=batch_size, verbose=0,
              validation_data=(X_val, {'outputs': y_val['outputs']}), shuffle=True, callbacks=callbacks)

    return model



def model_training_pipeline(date_feature: pd.DataFrame, test_size: int, input_range: int, prediction_time: int,
                            numerical_features, categorical_features,learning_rate: float, batch_size: int, model_type: str, loss: str='mse',
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
                                         numerical_features=numerical_features, categorical_features= categorical_features)

    X_val, y_val = tf_input_pipeline(df_val, input_range=input_range, prediction_time=prediction_time,
                                     numerical_features=numerical_features, categorical_features= categorical_features)

    _, n_inputs, n_features = X_train['encoder_X_num'].shape
    _, n_outputs, _ = y_train['outputs'].shape

    assert model_type in ['LSTM2LSTM', 'CNN2LSTM', 'BiLSTM2LSTM']

    m = importlib.import_module(f"train.logic.model.{model_type}_architecture")

    #model = m.build_model(n_inputs=n_inputs, n_features=n_features, dropout=dropout,
                          #encoder_cat_dict=X_train['encoder_X_cat'], decoder_cat_dict=X_train['decoder_X_cat'],
                          #recurrent_dropout=recurrent_dropout, n_outputs=n_outputs, **kwargs)

    model = m.build_model(n_inputs=n_inputs, n_features=n_features, dropout=dropout,
                          encoder_cat_dict=X_train['encoder_X_cat'], decoder_cat_dict=X_train['decoder_X_cat'],
                          recurrent_dropout=recurrent_dropout, n_outputs=n_outputs,
                          weekly_inputs=basic_parameters['weekly_inputs'],**kwargs)

    #  can have customized optimizer as well
    # 新加入batch size np round和weekly_inputs
    batch_size = int(np.round(batch_size, 0))
    model = model_training(model, X_train, y_train, X_val, y_val, batch_size=batch_size,
                           learning_rate=learning_rate, loss=loss,
                           weekly_inputs=basic_parameters['weekly_inputs'])

    return model, scaler



def cross_validation(date_feature: pd.DataFrame, n_splits: int, test_size: int, input_range: int,
                     prediction_time: int, max_train_size: int, numerical_features: List,
                     batch_size: int, learning_rate: float, model_type: str, loss: str='mse',
                     lead_time: int=0, dropout: float=0, recurrent_dropout: float=0,
                     categorical_features: Optional[List[str]]=None, **kwargs):

    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, max_train_size=max_train_size)

    y_pred = []
    y_true = []

    for n_fold, (train_index, test_index) in enumerate(tscv.split(date_feature)):
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
                                                prediction_time=prediction_time, numerical_features=numerical_features,categorical_features= categorical_features,
                                                loss=loss, learning_rate=learning_rate, batch_size=batch_size,
                                                model_type=model_type, dropout=dropout,
                                                recurrent_dropout=recurrent_dropout, **kwargs)


        df_test.loc[:, numerical_features] = scaler.transform(df_test[numerical_features])

        X_test, y_test = tf_input_pipeline(df_test, input_range=input_range, prediction_time=prediction_time,
                                           numerical_features=numerical_features,categorical_features= categorical_features)

        y_true_extend = np.repeat(y_test['true'].reshape(-1, 1), len(scaler.scale_), axis=1)
        y_true_reshape = scaler.inverse_transform(y_true_extend)[:, 0].reshape(y_test['true'].shape)
        y_true.append(y_true_reshape)

        X_test = generate_weekly_inputs(X_test, y_test)

        pred = model.predict(X_test)

        y_pred_extend = np.repeat(pred.reshape(-1, 1), len(scaler.scale_), axis=1)
        y_pred_reshape = np.round(scaler.inverse_transform(y_pred_extend)[:, 0].reshape(pred.shape))
        y_pred.append(y_pred_reshape)

    return np.array(y_true), np.array(y_pred)


#def cross_validation(date_feature: pd.DataFrame, n_splits: int, test_size: int, input_range: int,
                     #prediction_time: int, max_train_size: int, numerical_features: List,
                     #batch_size: int, learning_rate: float, model_type: str, loss: str='mse',
                     #lead_time: int=0, dropout: float=0, recurrent_dropout: float=0,
                     #categorical_features: Optional[List[str]]=None,
                     #**kwargs):

    #tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, max_train_size=max_train_size)

    #y_pred = []
    #y_true = []

    #for n_fold, (train_index, test_index) in enumerate(tscv.split(date_feature)):

        #test_index = np.arange(test_index[0] - input_range - lead_time - prediction_time, test_index[-1] + 1)

        #df_train = date_feature.iloc[train_index]

        # Apply rescaling:
        # https://stackoverflow.com/questions/43467597/should-i-normalize-my-features-before-throwing-them-into-rnn
        # It might help improve the performance of the model

        #time_begin = df_train.index[0]
        #time_end = df_train.index[-1]

        #df_test = date_feature.iloc[test_index]

        #print(f"fold {n_fold}: training: {time_begin} - {time_end}, testing: {df_test.index[0]} - {df_test.index[-1]}")

        #model, scaler = model_training_pipeline(date_feature=df_train, test_size=test_size, input_range=input_range,
                                                #prediction_time=prediction_time, numerical_features=numerical_features,
                                                #categorical_features= categorical_features,
                                                #loss=loss, learning_rate=learning_rate, batch_size=batch_size,
                                                #model_type=model_type, dropout=dropout,
                                                #recurrent_dropout=recurrent_dropout, **kwargs)

        #df_test.loc[:, numerical_features] = scaler.transform(df_test[numerical_features])

        #X_test, y_test = tf_input_pipeline(df_test, input_range=input_range, prediction_time=prediction_time,
                                           #numerical_features=numerical_features, categorical_features= categorical_features)

        #y_true.append(y_test['true'])

        #X_tf_test = {'encoder_X_num': X_test['encoder_X_num']}
        #X_tf_test.update(X_test['encoder_X_cat'])
        #X_tf_test.update(X_test['decoder_X_cat'])

        #pred = model.predict(X_tf_test)

        #y_pred.append(pred)

    #return np.array(y_true), np.array(y_pred)
