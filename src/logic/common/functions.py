import re
from typing import Tuple, Optional, Dict
import os
import numpy as np
from src.io.path_definition import get_file, _load_yaml
from train.src import config

def parenthesis_striped(ingredient: str) -> Tuple[str, Optional[str]]:
    '''
    strip parenthesis from the ingredient name
    Args:
        ingredient:
    Returns:
    '''

    name_1 = ingredient

    parenthesis_remove_regex = re.compile('[\w\s_]+(\([^)]+\))')
    '''
    \( : match an opening parentheses
    ( : begin capturing group
    [^)]+: match one or more non ) characters
    ) : end capturing group
    \) : match closing parentheses
    '''
    group = parenthesis_remove_regex.match(ingredient)
    if group:
        chemical_list = [e for e in ingredient.replace(group[1], "").split(" ") if e != '']
        name_1 = " ".join(chemical_list)

    return name_1

# input range的28天，將每週取平均(例如:每週一) 作為decoder的輸入
def generate_weekly_inputs(X: Dict, y: Dict):


    encoder_X_num = X['encoder_X_num']
    # encoder_X_num_canceled = encoder_X_num[:, :, 1]
    # Use the average of 4 weeks average as input
    encoder_X_num_canceled = encoder_X_num[:, :, 0]
    data_size, n_days = encoder_X_num_canceled.shape
    rearranged_encoder_X_num_canceled = encoder_X_num_canceled.reshape((data_size, y['outputs'].shape[1], -1))
    encoder_X_num_canceled_weekly = rearranged_encoder_X_num_canceled.mean(axis=2)
    encoder_X_num_canceled_weekly = np.expand_dims(encoder_X_num_canceled_weekly, axis=2)
    X['previous_weekly_average_cancelled_inputs'] = encoder_X_num_canceled_weekly

    encoder_X_num_canceled = encoder_X_num[:, :, 1]
    data_size, n_days = encoder_X_num_canceled.shape
    rearranged_encoder_X_num_canceled = encoder_X_num_canceled.reshape((data_size, y['outputs'].shape[1], -1))
    #encoder_X_num_canceled_weekly = rearranged_encoder_X_num_canceled.mean(axis=2)
    #encoder_X_num_canceled_weekly = np.expand_dims(encoder_X_num_canceled_weekly, axis=2)
    #X['previous_weekly_average_booking_inputs'] = encoder_X_num_canceled_weekly

    #X['future_booking_inputs'] = np.expand_dims(X['decoder_X_num'], axis=2)

    model_metadata = _load_yaml(get_file(os.path.join('config', 'training_config.yml')))
    # basic_parameters = model_metadata['basic_parameters']
    numerical_features = model_metadata['features_configuration'][config.configuration]['numerical']
    if 'booking' in numerical_features:
        X['future_booking_inputs'] = np.expand_dims(X['decoder_X_num'], axis=2)

    return X