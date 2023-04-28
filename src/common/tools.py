import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def timeseries_train_test_split(df: pd.DataFrame, test_size: float = 0.1):

    train_time, test_time = train_test_split(np.unique(df['check_in']), test_size=test_size, shuffle=False,
                                             random_state=0)
    train_dataset = df[df['check_in'].isin(train_time)]
    eval_dataset = df[df['check_in'].isin(test_time)]
    train_target = train_dataset['label']
    eval_target = eval_dataset['label']

    return train_dataset, eval_dataset, train_target, eval_target


def prediction_postprocessing(y: np.ndarray, scaler) -> np.ndarray:

    y_extend = np.repeat(y.reshape(-1, 1), len(scaler.scale_), axis=1)
    y_reshape = np.round(scaler.inverse_transform(y_extend)[:, 0].reshape(y.shape))

    return np.squeeze(y_reshape, axis=2)


def timeseries_prediction_postprocessing(y: np.ndarray) -> np.ndarray:

    y_exp = np.empty((len(y), len(y) + config.prediction_time - 1), dtype=object)

    for ix, pred in enumerate(y):
        y_exp[ix, ix: ix + config.prediction_time] = pred

    y_exp = np.array(y_exp, dtype=np.float)

    return np.nanmean(y_exp, axis=0)
