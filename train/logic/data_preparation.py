from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, TimeDistributed, GlobalAveragePooling1D, RepeatVector

from src.logic.common.functions import parenthesis_striped


def data_preparation(hotel_id: int, date_feature: pd.DataFrame, cancel_target: pd.DataFrame, smooth:bool=False,
                     diff: Optional[List[int]]=None):

    column = f"hotel_{hotel_id}_canceled"

    hotel_cancel = cancel_target[column].replace("-", np.nan).dropna().astype(int)

    date_feature = date_feature.loc[hotel_cancel.index]
    date_feature['canceled'] = hotel_cancel   # 原始值

    num_feature_columns = ['canceled']

    return num_feature_columns, date_feature


def to_supervised(df: pd.DataFrame, input_range: int, prediction_time: int, numerical_features: List,
                  lead_time: int=0, prediction: bool=False):

    day_increment = 1

    date_feature_numerical = df[numerical_features]
    encoder_X_num = list()
    decoder_X_num = list()

    if not prediction:
        y = list()

    in_start = 0

    for idx in range(len(df) - (lead_time + prediction_time)):

        in_end = in_start + input_range
        out_start = in_end + lead_time
        out_end = out_start + prediction_time
        if out_end < len(date_feature_numerical):

            encoder_X_num.append(date_feature_numerical.iloc[in_start: in_end].values)
            decoder_X_num.append(date_feature_numerical['booking'].iloc[out_start: out_end].values)

            if not prediction:
                y.append(df.iloc[out_start: out_end]['canceled'].tolist())

            in_start += day_increment
        else:
            break

    # reshape output into [samples, timesteps, features]
    if not prediction:
        shape_0, shape_1 = np.array(y).shape
        y = np.array(y).reshape(shape_0, shape_1, 1)

    results = {'encoder_X_num': np.array(encoder_X_num),
               'decoder_X_num': np.array(decoder_X_num)}

    if not prediction:
        results['y'] = y  # 原始值

    return results


def parse_tf_input(results: Dict, prediction: bool = False):

    X = {'encoder_X_num': results['encoder_X_num'],
         'decoder_X_num': results['decoder_X_num']}

    if prediction:
        return X
    else:
        y = {'outputs': tf.convert_to_tensor(results['y'], dtype=tf.float32),
             'true': results['y']}
        return X, y


def tf_input_pipeline(df: pd.DataFrame, input_range: int, prediction_time: int, numerical_features):

    results = to_supervised(df, input_range=input_range, prediction_time=prediction_time,
                                  numerical_features=numerical_features)

    X, y = parse_tf_input(results)

    return X, y