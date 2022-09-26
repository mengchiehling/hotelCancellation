import os
import joblib
import argparse

from src.io.path_definition import get_file, get_project_dir
from src.io.load_parameters import optimized_parameters
from train.logic.model_selection import model_training_pipeline


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_range', type=int, help='length of input time series')
    parser.add_argument('--prediction_time', type=int, help='length of output time series')
    parser.add_argument('--hotel_id', type=int, help='id of hotel, should exists in cancel_dataset_target.csv')


    args = parser.parse_args()

    input_range = args.input_range
    prediction_time = args.prediction_time
    hotel_id = args.hotel_id
    model_type = 'LSTM2LSTM'


    test_size = 30

    params, _ = optimized_parameters(f"{model_type}" + "_logs_[\d]{8}-[\d]{2}.json")

    params['batch_size'] = int(params['batch_size'])
    params['decoder_dense_units'] = int(params['batch_size'])
    params['encoder_lstm_units'] = int(params['encoder_lstm_units'])

    #不需要max_train_size嗎 ?
    model, scaler = model_training_pipeline(date_feature=date_feature, test_size=test_size, input_range=input_range,
                                            prediction_time=prediction_time, numerical_features=numerical_features,
                                            categorical_features=categorical_features,model_type=model_type, **params)


    dir = os.path.join(get_project_dir(), 'data', 'model', model_type)

    if not os.path.isdir(dir):
        os.makedirs(dir)
    model.save(os.path.join(dir, 'model'))

    with open(os.path.join(dir, 'scaler'), mode='wb') as f:
        joblib.dump(scaler, f)