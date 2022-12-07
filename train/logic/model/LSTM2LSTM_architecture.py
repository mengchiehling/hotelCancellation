'''
LSTM2LSTM parameters:

1. encoder:

    Must: encoder_lstm_units

2. decoder:
    Optional:
        decoder_dense_units
'''

import tensorflow as tf
import os
import random
import numpy as np

from typing import Dict
from tensorflow.keras.models import Model
from train.logic.model.encoder import LSTM_encoder
from train.logic.model.decoder import LSTM_decoder


def build_model(n_inputs, n_features, encoder_cat_dict: Dict, decoder_cat_dict: Dict, n_outputs: int,
                dropout: float=0, recurrent_dropout: float=0,
                weekly_inputs: bool=False , **kwargs):


    tf.random.set_seed(42)
    os.environ['PYTHONHASHSEED']='42'
    random.seed(42)
    np.random.seed(42)


    encoder_lstm_units = int(kwargs.get('encoder_lstm_units'))
    decoder_dense_units = int(kwargs.get('decoder_dense_units'))

    encoder_inputs_layers, _, state_h, state_c = LSTM_encoder(n_inputs, n_features, lstm_units=encoder_lstm_units,
                                                              recurrent_dropout=recurrent_dropout,
                                                              dropout=dropout, encoder_cat_dict=encoder_cat_dict)

    decoder_inputs_layers, outputs = LSTM_decoder(state_h, dense_units=decoder_dense_units,
                                                  lstm_units=encoder_lstm_units, decoder_cat_dict=decoder_cat_dict,
                                                  dropout=dropout, recurrent_dropout=recurrent_dropout,state_c=state_c,
                                                  n_outputs=n_outputs,weekly_inputs=weekly_inputs)

    for _, layer in decoder_inputs_layers.items():
        encoder_inputs_layers.append(layer)

    model = Model(inputs=encoder_inputs_layers, outputs=outputs)

    return model

