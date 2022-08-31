'''
LSTM2LSTM parameters:

1. encoder:

    Must: encoder_lstm_units

2. decoder:
    Optional:
        decoder_dense_units
'''

from typing import Dict

from tensorflow.keras.models import Model

from train.logic.model.encoder import LSTM_encoder
from train.logic.model.decoder import LSTM_decoder


def build_model(n_inputs, n_features, decoder_cat_dict: Dict, n_outputs: int,
                dropout: float=0, recurrent_dropout: float=0, **kwargs):

    encoder_lstm_units = kwargs.get('encoder_lstm_units')

    decoder_dense_units = kwargs.get('decoder_dense_units')

    encoder_inputs_layers, _, state_h, state_c = LSTM_encoder(n_inputs, n_features, lstm_units=encoder_lstm_units,
                                                              recurrent_dropout=recurrent_dropout,
                                                              dropout=dropout)

    decoder_inputs_layers, outputs = LSTM_decoder(state_h, dense_units=decoder_dense_units,
                                                  lstm_units=encoder_lstm_units, decoder_cat_dict=decoder_cat_dict,
                                                  dropout=dropout, recurrent_dropout=recurrent_dropout,state_c=state_c)


    model = Model(inputs=encoder_inputs_layers + decoder_inputs_layers, outputs=outputs)

    return model

