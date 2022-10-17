import os
from typing import Dict, Tuple
from datetime import datetime
from glob import glob

import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

from src.io.path_definition import get_project_dir


def optimization_process(fn, pbounds: Dict, model_type: str, hotel_id: int) -> Tuple[Dict, np.ndarray]:

    """
    Bayesian optimization process interface. Returns hyperparameters of machine learning algorithms and the
    corresponding out-of-fold (oof) predictions. The progress will be saved into a json file.

    Args:
        fn: functional that will be optimized
        pbounds: a dictionary having the boundary of parameters of fn

    Returns:
        A tuple of dictionary containing optimized hyperparameters and oof-predictions
    """

    bayesianOptimization = {'init_points': 5, #init_points:8
                            'n_iter': 25,  #n_iter:32
                            'acq': 'ucb'}

    optimizer = BayesianOptimization(
        f=fn,
        pbounds=pbounds,
        random_state=1)

    export_form = datetime.now().strftime("%Y%m%d-%H")

    dir = os.path.join(get_project_dir(), 'data', 'optimization', 'without_category')
    if not os.path.isdir(dir):
        os.makedirs(dir)

    logs = f"{dir}/{hotel_id}_{model_type}_logs_{export_form}.json"
    previous_logs = glob(f"{dir}/{hotel_id}_{model_type}_logs_*.json")

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