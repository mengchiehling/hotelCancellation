from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, TimeDistributed, GlobalAveragePooling1D, RepeatVector

from src.logic.common.functions import parenthesis_striped


def to_supervised(df: pd.DataFrame, input_range: int, prediction_time: int, numerical_features: List,
                  lead_time: int=0, prediction: bool=False):

    day_increment = 1

    date_feature_numerical = df[numerical_features]

    encoder_X_num = list()

    if not prediction:
        y = list()
        y_label = list()
        y_hat = list()

    in_start = 0

    for idx in range(len(df) - (lead_time + prediction_time)):

        in_end = in_start + input_range
        out_start = in_end + lead_time
        out_end = out_start + prediction_time
        if out_end < len(date_feature_numerical):

            encoder_X_num.append(date_feature_numerical.iloc[in_start: in_end].values)

            if not prediction:
                y.append(df.iloc[out_start: out_end]['canceled'].tolist())
                y_label.append(df.iloc[out_start: out_end]['canceled_label'].tolist())
                y_hat.append(df.iloc[out_start: out_end]['yhat'].tolist())

            in_start += day_increment
        else:
            break

    # reshape output into [samples, timesteps, features]
    if not prediction:
        shape_0, shape_1 = np.array(y_label).shape
        y = np.array(y).reshape(shape_0, shape_1, 1)
        y_label = np.array(y_label).reshape(shape_0, shape_1, 1)
        y_hat = np.array(y_hat).reshape(shape_0, shape_1, 1)

    results = {'encoder_X_num': np.array(encoder_X_num)}

    if not prediction:
        results['y'] = y  # 原始值
        results['y_label'] = y_label
        results['yhat'] = y_hat

    return results


def parse_tf_input(results: Dict, prediction: bool = False):

    X = {'encoder_X_num': results['encoder_X_num']}

    if prediction:
        return X
    else:
        y = {'outputs': tf.convert_to_tensor(results['y_label'], dtype=tf.float32),
             'true': results['y'],
             'yhat': results['yhat']}
        return X, y