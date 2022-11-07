from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, TimeDistributed, GlobalAveragePooling1D, RepeatVector

from src.logic.common.functions import parenthesis_striped


#def data_preparation(hotel_id: int, date_feature: pd.DataFrame, cancel_target: pd.DataFrame, smooth:bool=False,
                     #diff: Optional[List[int]]=None):

    #column = f"hotel_{hotel_id}_canceled"

    #hotel_cancel = cancel_target[column].replace("-", np.nan).dropna().astype(int)

    #date_feature = date_feature.loc[hotel_cancel.index]
    #date_feature['canceled'] = hotel_cancel   # 原始值

    #num_feature_columns = ['canceled']

    #return num_feature_columns, date_feature


def to_supervised(df: pd.DataFrame, input_range: int, prediction_time: int, numerical_features: List,
                  categorical_features: Optional[List[str]]=None, lead_time: int=0, prediction: bool=False):

    day_increment = 1

    date_feature_numerical = df[numerical_features]
    encoder_X_num = list()

    if categorical_features is not None:
        date_feature_categorical = df[categorical_features]
        # encoder categorical features
        decoder_X_cat = {}
        encoder_X_cat = {}

        for c in categorical_features:
            decoder_X_cat[f'{c}_decoder'] = []
            encoder_X_cat[f'{c}_encoder'] = []

    if not prediction:
        y = list()
        #y_label = list()

    in_start = 0

    for idx in range(len(df) - (lead_time + prediction_time)):

        in_end = in_start + input_range
        out_start = in_end + lead_time
        out_end = out_start + prediction_time
        if out_end < len(date_feature_numerical):

            encoder_X_num.append(date_feature_numerical.iloc[in_start: in_end].values)

            if categorical_features is not None:
                for c in categorical_features:
                    decoder_X_cat[f'{c}_decoder'].append(date_feature_categorical.iloc[out_start: out_end][c])
                    encoder_X_cat[f'{c}_encoder'].append(date_feature_categorical.iloc[in_start: in_end][c])

            if not prediction:
                y.append(df.iloc[out_start: out_end]['canceled'].tolist())
                #y_label.append(df.iloc[out_start: out_end]['canceled_label'].tolist())

            in_start += day_increment
        else:
            break

    # reshape output into [samples, timesteps, features]
    if not prediction:
        shape_0, shape_1 = np.array(y).shape
        y = np.array(y).reshape(shape_0, shape_1, 1)
        #y_label = np.array(y_label).reshape(shape_0, shape_1, 1)

    if categorical_features is not None:
        for layer_name in decoder_X_cat.keys():
            decoder_X_cat[layer_name] = np.array(decoder_X_cat[layer_name])
            decoder_X_cat[layer_name] = decoder_X_cat[layer_name].reshape(-1, prediction_time, 1)
        for layer_name in encoder_X_cat.keys():
            encoder_X_cat[layer_name] = np.array(encoder_X_cat[layer_name])
            encoder_X_cat[layer_name] = encoder_X_cat[layer_name].reshape(-1, input_range, 1)

    results = {'encoder_X_num': np.array(encoder_X_num)}
    if categorical_features is not None:
        results.update({'encoder_X_cat': encoder_X_cat,
                        'decoder_X_cat': decoder_X_cat})

    if not prediction:
        results['y'] = y  # 原始值
        #results['y_label'] = y_label #

    return results


def parse_tf_input(results: Dict, prediction: bool = False):

    X = {'encoder_X_num': results['encoder_X_num'],
         'encoder_X_cat': results.get('encoder_X_cat', None),
         'decoder_X_cat': results.get('decoder_X_cat', None)}

    if prediction:
        return X
    else:
        y = {'outputs': tf.convert_to_tensor(results['y'], dtype=tf.float32),
             'true': results['y']}
        return X, y


def generate_categorical_embeddings(section: str, cat_dict: Optional[Dict]=None):

    inputs_layers = []
    categorical_inputs = []

    if cat_dict:
        for key, value in cat_dict.items():
            q = value
            _, shape_1, shape_2 = q.shape

            cat_input = Input(shape=(shape_1, shape_2), name=f"{key}")

            inputs_layers.append(cat_input)

            input_dim = len(np.unique(value))
            y_embedding = Embedding(input_dim, 1)(cat_input)
            y_embedding = TimeDistributed(GlobalAveragePooling1D(), name=f"{parenthesis_striped(key)}_{section}_embed")(
                y_embedding)

            categorical_inputs.append(y_embedding)

    return inputs_layers, categorical_inputs


def tf_input_pipeline(df: pd.DataFrame, input_range: int, prediction_time: int, numerical_features,
                      categorical_features: Optional[List[str]]=None):

    results = to_supervised(df, input_range=input_range, prediction_time=prediction_time,
                            numerical_features=numerical_features, categorical_features=categorical_features)

    X, y = parse_tf_input(results)

    return X, y