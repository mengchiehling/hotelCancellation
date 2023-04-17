import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def timeseries_train_test_split(df: pd.DataFrame, test_size: float):

    train_time, test_time = train_test_split(np.unique(df['check_in']), test_size=test_size, shuffle=False,
                                             random_state=0)
    train_dataset = df[df['check_in'].isin(train_time)]
    eval_dataset = df[df['check_in'].isin(test_time)]
    train_target = train_dataset['label']
    eval_target = eval_dataset['label']

    return train_dataset, eval_dataset, train_target, eval_target
