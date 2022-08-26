from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, TimeDistributed, GlobalAveragePooling1D, RepeatVector

from src.logic.common.functions import parenthesis_striped


def to_supervised(df: pd.DataFrame, input_range: int, prediction_time: int, numerical_features: List,
                  categorical_features: List, lead_time: int=0, prediction: bool=False):

    day_increment = 1

    date_feature_numerical = df[numerical_features].drop(labels=['canceled_label'], axis=1)
    date_feature_categorical = df[categorical_features]

    encoder_X_num = list()
    # encoder categorical features
    decoder_X_cat = {}
    encoder_X_cat = {}

    for c in categorical_features:
        decoder_X_cat[c] = {'value': []}
        encoder_X_cat[c] = {'value': []}

    if not prediction:
        y = list()
        y_label = list()

    in_start = 0

    for idx in range(len(df) - (lead_time + prediction_time)):

        in_end = in_start + input_range
        out_start = in_end + lead_time
        out_end = out_start + prediction_time
        if out_end < len(date_feature_numerical):

            encoder_X_num.append(date_feature_numerical.iloc[in_start: in_end].values)

            for c in categorical_features:
                decoder_X_cat[c]['value'].append(date_feature_categorical.iloc[out_start: out_end][c])
                encoder_X_cat[c]['value'].append(date_feature_categorical.iloc[in_start: in_end][c])

            if not prediction:
                y.append(df.iloc[out_start: out_end]['canceled'].tolist())
                y_label.append(df.iloc[out_start: out_end]['canceled_label'].tolist())

            in_start += day_increment
        else:
            break

    # reshape output into [samples, timesteps, features]
    if not prediction:
        shape_0, shape_1 = np.array(y).shape
        y = np.array(y).reshape(shape_0, shape_1, 1)
        y_label = np.array(y_label).reshape(shape_0, shape_1, 1)

    for c in categorical_features:
        decoder_X_cat[c]['value'] = np.array(decoder_X_cat[c]['value'])
        decoder_X_cat[c]['value'] = decoder_X_cat[c]['value'].reshape(-1, prediction_time, 1)
        encoder_X_cat[c]['value'] = np.array(encoder_X_cat[c]['value'])
        encoder_X_cat[c]['value'] = encoder_X_cat[c]['value'].reshape(-1, input_range, 1)

    results = {'encoder_X_num': np.array(encoder_X_num),
               'encoder_X_cat': encoder_X_cat,
               'decoder_X_cat': decoder_X_cat}

    if not prediction:
        results['y'] = y  # 原始值
        results['y_label'] = y_label #

    return results


def parse_tf_input(results: Dict, prediction: bool = False):

    X = {'encoder_X_num': results['encoder_X_num']}

    # for key, value in results['encoder_X_cat'].items():
    #     X[f"{key}_encoder"] = value['value']

    for key, value in results['decoder_X_cat'].items():
        X[f"{key}_decoder"] = value['value']

    if prediction:
        return X
    else:
        y = {'outputs': tf.convert_to_tensor(results['y_label'], dtype=tf.float32),
             'true': results['y']}
        return X, y


def generate_categorical_embeddings(x, decoder_cat_dict: Optional[Dict]=None):

    inputs_layers = []
    categorical_inputs = []

    if decoder_cat_dict:
        for key, item in decoder_cat_dict.items():
            q = item['value']
            _, shape_1, shape_2 = q.shape

            cat_input = Input(shape=(shape_1, shape_2), name=f"{key}_decoder")

            inputs_layers.append(cat_input)

            input_dim = item['input_dim']
            y_embedding = Embedding(input_dim, 1)(cat_input)
            y_embedding = TimeDistributed(GlobalAveragePooling1D(), name=f"{parenthesis_striped(key)}_decoder_embed")(
                y_embedding)

            categorical_inputs.append(y_embedding)

    return inputs_layers, categorical_inputs