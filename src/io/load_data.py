import os
from datetime import datetime

import numpy as np
import pandas as pd

from src.io.path_definition import get_file


# covid_data = pd.read_excel(get_file(os.path.join('data', 'owid-covid-data.xlsx')),
#                            #engine='openpyxl'
#                            )


def data_preparation(hotel_id: int, date_feature: pd.DataFrame, cancel_target: pd.DataFrame):

    column = f"hotel_{hotel_id}_canceled"

    cancel_target['date'] = cancel_target['date'].apply(lambda x: datetime.strptime(x, '%Y/%m/%d').strftime("%Y/%m/%d"))
    cancel_target.set_index('date', inplace=True)
    hotel_cancel = cancel_target[column].replace("-", np.nan).dropna().astype(int)

    date_feature['date'] = date_feature['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').strftime("%Y/%m/%d"))
    date_feature.set_index('date', inplace=True)
    # date_feature = date_feature.loc[hotel_cancel.index]
    date_feature = date_feature.join(hotel_cancel)
    date_feature.dropna(inplace=True)

    date_feature['canceled'] = hotel_cancel   # 原始值

    # twn_covid_data = covid_data[covid_data['iso_code']=='TWN']
    # twn_covid_data['date'] = twn_covid_data['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').strftime("%Y/%m/%d"))
    # twn_covid_data.set_index('date', inplace=True)
    #
    # covid_features_num = ['new_cases', 'new_deaths']
    #
    # date_feature = date_feature.join(twn_covid_data[covid_features_num].fillna(0))

    covid_features_num = []

    num_feature_columns = ['canceled', 'booking'] + covid_features_num

    return num_feature_columns, date_feature