"""
LSTM2LSTM parameters:

1. encoder:

    Must: encoder_lstm_units

2. decoder:
    Optional:
        decoder_dense_units
"""


import os
import random
from copy import copy
from typing import Dict, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

from src import config
from train.logic.model.encoder import LSTM_encoder
from train.logic.model.decoder import LSTM_decoder


def build_model(n_inputs, n_features, encoder_cat_dict: Optional[Dict], decoder_cat_dict: Optional[Dict],
                n_outputs: int, dropout: float = 0, recurrent_dropout: float = 0, **kwargs):

    tf.random.set_seed(42)
    os.environ['PYTHONHASHSEED'] = '42'
    random.seed(42)
    np.random.seed(42)

    encoder_lstm_units = int(kwargs.get('encoder_lstm_units'))
    decoder_dense_units = int(kwargs.get('decoder_dense_units'))

    encoder_inputs_layers, _, state_h, state_c = LSTM_encoder(n_inputs, n_features, lstm_units=encoder_lstm_units,
                                                              recurrent_dropout=recurrent_dropout,
                                                              dropout=dropout, encoder_cat_dict=encoder_cat_dict)

    outputs, decoder_input_layers = LSTM_decoder(state_h, dense_units=decoder_dense_units,
                                                 lstm_units=encoder_lstm_units, decoder_cat_dict=decoder_cat_dict,
                                                 dropout=dropout, recurrent_dropout=recurrent_dropout, state_c=state_c,
                                                 n_outputs=n_outputs, weekly_inputs=config.weekly_inputs)

    inputs = copy(encoder_inputs_layers)

    for _, layer in decoder_input_layers.items():
        inputs.append(layer)

    model = Model(inputs=inputs, outputs=outputs)

    return model
