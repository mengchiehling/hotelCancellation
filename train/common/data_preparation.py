from copy import copy
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, TimeDistributed, GlobalAveragePooling1D, RepeatVector

from src import config
from src.logic.common.functions import parenthesis_striped


def to_supervised(df: pd.DataFrame, prediction: bool = False):

    numerical_features = config.numerical_features
    categorical_features = config.categorical_features

    day_increment = 1

    if len(categorical_features) > 0:
        date_feature_categorical = df[categorical_features]
        # encoder categorical features
        decoder_X_cat = {f'{c}_decoder': [] for c in categorical_features}
        encoder_X_cat = {f'{c}_encoder': [] for c in categorical_features}

    if not prediction:
        y = list()

    in_start = 0

    date_feature_numerical = df[numerical_features]
    encoder_X_num = list()

    for idx in range(len(df) - (config.lead_time + config.prediction_time)):

        in_end = in_start + config.input_range
        out_start = in_end + config.lead_time
        out_end = out_start + config.prediction_time
        if out_end < len(date_feature_numerical):

            encoder_X_num.append(date_feature_numerical.iloc[in_start: in_end].values)

            if len(categorical_features) > 0:
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

    if len(categorical_features) > 0:
        for layer_name in decoder_X_cat.keys():
            decoder_X_cat[layer_name] = np.array(decoder_X_cat[layer_name])
            decoder_X_cat[layer_name] = decoder_X_cat[layer_name].reshape(-1, config.prediction_time, 1)
        for layer_name in encoder_X_cat.keys():
            encoder_X_cat[layer_name] = np.array(encoder_X_cat[layer_name])
            encoder_X_cat[layer_name] = encoder_X_cat[layer_name].reshape(-1, config.input_range, 1)

    results = {'encoder_X_num': np.array(encoder_X_num)}
    if len(categorical_features) > 0:
        results.update({'encoder_X_cat': encoder_X_cat,
                        'decoder_X_cat': decoder_X_cat})

    if not prediction:
        results['y'] = y  # 原始值
        #results['y_label'] = y_label #

    return results


def parse_tf_input(results: Dict, prediction: bool = False):

    # X = {'encoder_X_num': results['encoder_X_num'],
    #      'encoder_X_cat': results.get('encoder_X_cat', None),
    #      'decoder_X_cat': results.get('decoder_X_cat', None)}

    X = copy(results)

    if prediction:
        return X
    else:
        y = {'outputs': tf.convert_to_tensor(results['y'], dtype=tf.float32),
             'true': results['y']}
        return X, y


def generate_categorical_embeddings(section: str, cat_dict: Optional[Dict] = None):

    categorical_input_layers = []
    embedding_layers = []

    if cat_dict:
        for key, value in cat_dict.items():
            q = value
            _, shape_1, shape_2 = q.shape

            cat_input = Input(shape=(shape_1, shape_2), name=f"{key}")

            categorical_input_layers.append(cat_input)

            input_dim = len(np.unique(value)) + 10  # +1 for unseen category: https://stackoverflow.com/questions/61527381/unseen-category-in-validation-set-gives-error-when-using-keras
            y_embedding = Embedding(input_dim, 1)(cat_input)
            y_embedding = TimeDistributed(GlobalAveragePooling1D(), name=f"{parenthesis_striped(key)}_{section}_embed")(
                y_embedding)

            embedding_layers.append(y_embedding)

    return categorical_input_layers, embedding_layers


def tf_input_pipeline(df: pd.DataFrame):

    results = to_supervised(df)

    X, y = parse_tf_input(results)

    return X, y
