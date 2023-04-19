import os
from typing import Dict, Tuple
from datetime import datetime
from glob import glob

import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

from src import config
from src.io.path_definition import get_project_dir, get_file, load_yaml_file


def optimization_process(fn, pbounds: Dict) -> Tuple[Dict, np.ndarray]:

    """
    Bayesian optimization process interface. Returns hyperparameters of machine learning algorithms and the
    corresponding out-of-fold (oof) predictions. The progress will be saved into a json file.
    Args:
        fn: functional that will be optimized
        pbounds: a dictionary having the boundary of parameters of fn
    Returns:
        A tuple of dictionary containing optimized hyperparameters and oof-predictions
    """

    # So you do not have to change the hyperparameters explicitly to github everytime you change the code.
    path_name = os.path.join('config', 'training_config.yml')
    bayesianOptimization = load_yaml_file(get_file(path_name))['bayesianOptimization']

    optimizer = BayesianOptimization(
        f=fn,
        pbounds=pbounds,
        random_state=1)

    export_form = datetime.now().strftime("%Y%m%d-%H%M")

    dir_ = os.path.join(get_project_dir(), 'data', 'optimization') #, 'without_category')
    if not os.path.isdir(dir_):
        os.makedirs(dir_)

    hotel_id = config.hotel_id
    algorithm = config.algorithm
    configuration = config.configuration

    logs = f"{dir_}/logs_{algorithm}_{configuration}_{hotel_id}_{export_form}.json"
    previous_logs = glob(f"{dir_}/logs_{algorithm}_{configuration}_{hotel_id}_*.json")

    if previous_logs:
        load_logs(optimizer, logs=previous_logs)

    logger = JSONLogger(path=logs)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    if logs:
        bayesianOptimization['init_points'] = 0

    optimizer.maximize(
        **bayesianOptimization
    )
    optimized_parameters = optimizer.max['params']

    return optimized_parameters