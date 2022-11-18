import os
from datetime import datetime

import numpy as np
import pandas as pd

from src.io.path_definition import get_file



def data_preparation(hotel_id: int, booking_feature: pd.DataFrame, cancel_target: pd.DataFrame):

    column = f"hotel_{hotel_id}_canceled"

    cancel_target['date'] = cancel_target['date'].apply(lambda x: datetime.strptime(x, '%Y/%m/%d').strftime("%Y/%m/%d"))
    cancel_target.set_index('date', inplace=True)
    hotel_cancel = cancel_target[column].replace("-", np.nan).dropna().astype(int)

    booking_feature['date'] = booking_feature['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').strftime("%Y/%m/%d"))
    booking_feature.set_index('date', inplace=True)
    # date_feature = date_feature.loc[hotel_cancel.index]
    booking_feature = booking_feature.join(hotel_cancel)

    date_feature = pd.read_csv(get_file(os.path.join('data', f'cancel_dataset_date_feature.csv')))
    date_feature['date'] = date_feature['date'].apply(lambda x: datetime.strptime(x, '%Y/%m/%d').strftime("%Y/%m/%d"))
    date_feature.set_index('date', inplace=True)
    booking_feature = booking_feature.join(date_feature[['Precp', 'PrecpHour', 'SunShine', 'Temperature']])

    booking_feature.dropna(inplace=True)
    booking_feature['canceled'] = hotel_cancel   # 原始值

    covid_features_num = []

    num_feature_columns = ['canceled', 'booking', 'days2vecation', 'vecation_days', 'Precp',
                           'PrecpHour','SunShine','Temperature'] + covid_features_num

    return num_feature_columns, booking_feature