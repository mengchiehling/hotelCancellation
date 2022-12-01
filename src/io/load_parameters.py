import os
import re
from datetime import datetime

from src.io.path_definition import get_project_dir


def retrieve_hyperparameter_files(search_pattern: str):

    '''
    search_pattern = "LDA_logs_[\d]{8}-[\d]{2}.json"
    "CNN2LSTM_logs_[\d]{8}-[\d]{2}.json"
    :param search_pattern:
    :return:
    '''


    dir_ = os.path.join(get_project_dir(), 'data', 'optimization', 'without_category')


    res = [f for f in os.listdir(dir_) if re.search(search_pattern, f)]
    files = [os.path.join(dir_, f) for f in res]

    files_with_time = [(file, datetime.fromtimestamp(os.path.getmtime(file))) for file in files]

    files_with_time.sort(key=lambda x: x[1])

    files = [f[0] for f in files_with_time]

    return files


def optimized_parameters(search_pattern: str):

    files = retrieve_hyperparameter_files(search_pattern)

    target_max = -1

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