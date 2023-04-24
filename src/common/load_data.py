import os
from datetime import datetime
from typing import List, Tuple

import pandas as pd

from src.io.path_definition import get_datafetch, get_file


def remake_time_index(df: pd.DataFrame, hotel_id: int) -> List:

    date_list = df['check_in'].tolist()
    date_list.sort()
    date_end = date_list[-1]

    date_list = df[df['pms_hotel_id'] == hotel_id]['check_in'].tolist()
    date_list.sort()
    date_start = date_list[0]

    idx = pd.date_range(date_start, date_end)
    idx = [t.strftime("%Y-%m-%d") for t in idx]

    return idx


def load_data() -> pd.DataFrame:

    filename = os.path.join(get_datafetch(), '訂單資料_20221229.csv')
    booking_data = pd.read_csv(filename, index_col=0)
    booking_data.set_index('number', inplace=True)

    filename = os.path.join(get_datafetch(), '訂房資料_20221202.csv')
    room_data = pd.read_csv(filename, index_col=0)
    room_data = room_data.drop_duplicates(subset=['number'], keep='first').set_index('number')

    booking_data = booking_data.join(room_data[['lead_time', 'platform', 'holiday', 'weekday', 'pms_room_type_id', 'lead_time_range']], how='inner')

    filename = os.path.join(get_datafetch(), 'date_features.csv')
    date_features = pd.read_csv(filename)
    date_features['date'] = date_features['date'].apply(
     lambda x: datetime.strptime(x, '%Y/%m/%d').strftime("%Y-%m-%d"))
    booking_data = booking_data.merge(date_features, how='left', left_on='check_in', right_on='date')

    filename = os.path.join(get_datafetch(), '房型資料_20221229.csv')
    room_type_data = pd.read_csv(filename, index_col=0)
    room_type_data1 = room_type_data.set_index(["room_type_id"])['type'].to_dict()
    booking_data['type'] = booking_data['pms_room_type_id'].map(room_type_data1)

    # Attach the hotel information
    # Because not having hotel_info, comment out the following lines
    # filename = os.path.join(get_datafetch(), 'hotel_info.csv')
    # hotel_data = pd.read_csv(filename, index_col=0, encoding='utf-8')
    # # 這邊要寫pms hotel id ,還是hotel id
    # booking_data = booking_data.join(hotel_data, how='left', on='pms_hotel_id')
    # booking_data.dropna(subset=['所在縣市'], inplace=True)
    booking_data['date_filter'] = pd.to_datetime(booking_data['date'])
    booking_data = booking_data[booking_data['date_filter'] < pd.to_datetime('2022-12-29')]

    return booking_data


# 結合取消數量+訂房量+疫情數據
def load_training_data(hotel_id: int, remove_business_booking: bool = True) -> Tuple[pd.DataFrame, pd.Series]:

    df = load_data()

    idx = remake_time_index(df, hotel_id=hotel_id)

    df['adults'].fillna(0, inplace=True)
    df['children'].fillna(0, inplace=True)
    filter_ = (df.adults == 0) & (df.children == 0)
    df = df[~filter_]

    # specifcy the pms_hotel_id
    df = df[df['pms_hotel_id'] == hotel_id]

    if remove_business_booking:
        df = df[df["source"] != "BUSINESS_BOOKING"]

    df = df[~(df['status'] == 'UPCOMING')]

    df['label'] = 0
    df.loc[df['status'] == 'CHECKED_IN', 'label'] = 0
    df.loc[df['status'] == 'CHECKED_OUT', 'label'] = 0
    df.loc[df['status'] == 'NO_SHOW', 'label'] = 0
    df.loc[df['status'] == 'CANCELED', 'label'] = 1

    df.sort_values(by='check_in', inplace=True)

    # Simple time series, without extra features.
    # booking_feature = df.groupby(by="check_in").agg(canceled=('label', 'sum'),
    #                                                 booking=('label', 'count'))
    #
    # booking_feature = booking_feature.reindex(idx, fill_value=0)

    return df, idx
