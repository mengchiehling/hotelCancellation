import os
import joblib
import argparse
import pandas as pd
import numpy as np

from datetime import datetime
from src.io.path_definition import get_file, get_project_dir
from src.io.load_parameters import load_optimized_parameters
from train.logic.model_selection import model_training_pipeline
from sklearn.preprocessing import LabelEncoder

hotel_info = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_hotel_info.csv')))

cancel_target = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_target.csv')))
date_feature = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_date_feature.csv')))
hotel_meta = pd.read_csv(get_file(os.path.join('data', 'cancel_dataset_hotel_info.csv')), index_col=0)

covid_data = pd.read_excel(get_file(os.path.join('data', 'owid-covid-data.xlsx'))
                           #,engine='openpyxl'
                           )


def labelencoding(df, column: str):

    le = LabelEncoder()
    df.loc[:, column] = le.fit_transform(df[column])

    return df



def data_preparation(hotel_id: int, date_feature: pd.DataFrame, cancel_target: pd.DataFrame
                     # , smooth:bool=False, diff: Optional[List[int]]=None
                     ):

    column = f"hotel_{hotel_id}_canceled"

    cancel_target['date'] = cancel_target['date'].apply(lambda x: datetime.strptime(x, '%Y/%m/%d').strftime("%Y/%m/%d"))
    cancel_target.set_index('date', inplace=True)

    hotel_cancel = cancel_target[column].replace("-", np.nan).dropna().astype(int)

    date_feature['date'] = date_feature['date'].apply(lambda x: datetime.strptime(x, '%Y/%m/%d').strftime("%Y/%m/%d"))
    date_feature.set_index('date', inplace=True)
    date_feature = date_feature.loc[hotel_cancel.index]

    date_feature['canceled'] = hotel_cancel   # 原始值
    #date_feature['canceled_label'] = hotel_cancel  # hotel_cancel_smooth.mean()

    twn_covid_data = covid_data[covid_data['iso_code'] == 'TWN']
    twn_covid_data['date'] = twn_covid_data['date'].apply(
        lambda x: datetime.strptime(x, '%Y-%m-%d').strftime("%Y/%m/%d"))
    twn_covid_data.set_index('date', inplace=True)

    covid_features_num = []  # ['new_cases', 'new_deaths']
    covid_features_cat = []

    date_feature = date_feature.join(twn_covid_data[covid_features_num+covid_features_cat].fillna(0))

    num_feature_columns = ['canceled'] + covid_features_num
    #num_feature_columns = ['canceled', 'canceled_label','days2vecation','vecation_days','Precp','PrecpHour','SunShine','Temperature']

    #if smooth:
        # Smoothed features for input

        #for window in [3, 14]:
            #c  = f'canceled_{window}_roll'
            #roll = hotel_cancel.rolling(window=window, min_periods=1)
            #date_feature[c] = roll.mean()
            #num_feature_columns.extend([c])
        #date_feature['MACD'] = date_feature[f'canceled_3_roll'] - date_feature[f'canceled_14_roll']
        #num_feature_columns.append('MACD')

    #if diff is not None:
        #for time_diff in diff:
            #c = f'diff_{time_diff}'
            #date_feature[c] = (date_feature['canceled'] - date_feature['canceled'].shift(time_diff)).fillna(0)
            #num_feature_columns.append(c)

    return num_feature_columns, covid_features_cat, date_feature



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_range', type=int, help='length of input time series')
    parser.add_argument('--prediction_time', type=int, help='length of output time series')
    parser.add_argument('--hotel_id', type=int, help='id of hotel, should exists in cancel_dataset_target.csv')
    #parser.add_argument('--diff', type=int,  nargs='+', help='差分', default=[])
    #parser.add_argument('--smooth', action='store_true')


    args = parser.parse_args()

    input_range = args.input_range
    prediction_time = args.prediction_time
    hotel_id = args.hotel_id
    model_type = 'LSTM2LSTM'
    #diff = args.diff
    #smooth=args.smooth

    max_train_size = 365
    test_size = 30

    categorical_features = []  # ['vecation', 'weekdate','season','midd','sallery', 'is_rest_day','s_vecation', 'w_vecation','workingday','is_event','cov_policy']  # encoded_columns + nonencoded_columns

    # 之後有類別型特徵時需要做labelencoding
    # for encoded_column in categorical_features:
    # date_feature = labelencoding(date_feature, encoded_column)


    numerical_features, covid_features_cat, date_feature = data_preparation(hotel_id, date_feature, cancel_target
                                                        # , smooth=smooth, diff=diff
                                                        )

    categorical_features = categorical_features + covid_features_cat
    date_feature = date_feature[numerical_features + categorical_features]

    params, _ = load_optimized_parameters(f"{hotel_id}_{model_type}" + "_logs_[\d]{8}-[\d]{2}.json")

    params['batch_size'] = int(params['batch_size'])
    params['decoder_dense_units'] = int(params['batch_size'])
    params['encoder_lstm_units'] = int(params['encoder_lstm_units'])


    model, scaler = model_training_pipeline(date_feature=date_feature.iloc[-max_train_size-test_size:], test_size=test_size, input_range=input_range,
                                            prediction_time=prediction_time, numerical_features=numerical_features,
                                            model_type=model_type, **params)


    dir = os.path.join(get_project_dir(), 'data', 'model', model_type)

    if not os.path.isdir(dir):
        os.makedirs(dir)
    model.save(os.path.join(dir, 'model'))

    with open(os.path.join(dir, 'scaler'), mode='wb') as f:
        joblib.dump(scaler, f)