import os
import re
import numpy as np
from datetime import datetime

from src import config
from src.io.path_definition import get_project_dir


def retrieve_hyperparameter_files():

    """

    :return:
    """
    dir_ = os.path.join(get_project_dir(), 'data', 'optimization')

    algorithm = config.algorithm
    configuration = config.configuration
    hotel_id = config.hotel_id

    search_pattern = 'logs_' + algorithm + f"_{configuration}" + f"_{hotel_id}" + "_[\d]{8}-[\d]{4}.json"

    res = [f for f in os.listdir(dir_) if re.search(search_pattern, f)]
    files = [os.path.join(dir_, f) for f in res]

    files_with_time = [(file, datetime.fromtimestamp(os.path.getmtime(file))) for file in files]

    files_with_time.sort(key=lambda x: x[1])

    files = [f[0] for f in files_with_time]

    return files


def load_optimized_parameters():

    files = retrieve_hyperparameter_files()

    target_max = -np.inf

    for f in files:
        with open(f, 'rb') as f:
            while True:
                data = f.readline()
                if not data:
                    break
                data = eval(data)
                target = data['target']
                if target > target_max:
                    target_max = target
                    params = data['params']

    return params, target_max
