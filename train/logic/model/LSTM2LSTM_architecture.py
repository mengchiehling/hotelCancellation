'''
LSTM2LSTM parameters:

1. encoder:

    Must: encoder_lstm_units

2. decoder:
    Optional:
        decoder_dense_units
'''

import os
import random
from typing import Dict

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

from train.logic.model.encoder import LSTM_encoder
from train.logic.model.decoder import LSTM_decoder


def build_model(n_inputs, n_features, n_outputs: int,
                dropout: float=0, recurrent_dropout: float=0,
                weekly_inputs: bool=False, **kwargs):

    tf.random.set_seed(42)
    os.environ['PYTHONHASHSEED'] = '42'
    random.seed(42)
    np.random.seed(42)

    encoder_lstm_units = kwargs.get('encoder_lstm_units')
    decoder_dense_units = kwargs.get('decoder_dense_units')
    l2 = kwargs.get('l2')

    encoder_inputs_layers, _, state_h, state_c = LSTM_encoder(n_inputs, n_features, lstm_units=encoder_lstm_units,
                                                              recurrent_dropout=recurrent_dropout,
                                                              dropout=dropout, l2=l2)

    outputs, decoder_input_layers = LSTM_decoder(state_h, dense_units=decoder_dense_units, n_outputs=n_outputs,
                                                 lstm_units=encoder_lstm_units, dropout=dropout,
                                                 recurrent_dropout=recurrent_dropout, state_c=state_c,
                                                 weekly_inputs=weekly_inputs)

    if weekly_inputs:
        encoder_inputs_layers.append(decoder_input_layers)

    model = Model(inputs=encoder_inputs_layers, outputs=outputs)

    return model

