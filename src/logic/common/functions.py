import re
import os
from typing import Tuple, Optional, Dict
import os
import numpy as np

from src import config
from src.io.path_definition import get_file, load_yaml_file


def parenthesis_striped(ingredient: str) -> Tuple[str, Optional[str]]:

    """
    strip parenthesis from the ingredient name
    Args:
        ingredient:
    Returns:
    """

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


def generate_weekly_inputs(x: Dict, y: Dict) -> Dict:

    """
    input range的28天，將每週取平均(例如:每週一) 作為decoder的輸入
    :param x:
    :param y:
    :return:
    """

    encoder_X_num = x['encoder_X_num']
    # encoder_X_num_canceled = encoder_X_num[:, :, 1]
    # Use the average of 4 weeks average as input
    encoder_X_num_canceled = encoder_X_num[:, :, 0]
    data_size, n_days = encoder_X_num_canceled.shape
    rearranged_encoder_X_num_canceled = encoder_X_num_canceled.reshape((data_size, y['outputs'].shape[1], -1))
    encoder_X_num_canceled_weekly = rearranged_encoder_X_num_canceled.mean(axis=2)
    encoder_X_num_canceled_weekly = np.expand_dims(encoder_X_num_canceled_weekly, axis=2)
    x['previous_weekly_average_cancelled_inputs'] = encoder_X_num_canceled_weekly

    # encoder_X_num_canceled = encoder_X_num[:, :, 1]
    # data_size, n_days = encoder_X_num_canceled.shape
    # rearranged_encoder_X_num_canceled = encoder_X_num_canceled.reshape((data_size, y['outputs'].shape[1], -1))
    # encoder_X_num_canceled_weekly = rearranged_encoder_X_num_canceled.mean(axis=2)
    # encoder_X_num_canceled_weekly = np.expand_dims(encoder_X_num_canceled_weekly, axis=2)
    # X['previous_weekly_average_booking_inputs'] = encoder_X_num_canceled_weekly
    # basic_parameters = model_metadata['basic_parameters']
    # numerical_features = config.numerical_features
    # if 'booking' in numerical_features:
    #     x['future_booking_inputs'] = np.expand_dims(x['decoder_X_num'], axis=2)

    return x
