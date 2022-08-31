'''
CNN2LSTM parameters:

1. encoder:

    Must: encoder_filters_0
    Optional:
        encoder_filters_1
        encoder_filters_2

2. decoder:
    Optional:
        decoder_dense_units
'''
from typing import Dict

from tensorflow.keras.models import Model

from train.logic.model.encoder import CNN_encoder
from train.logic.model.decoder import LSTM_decoder


def build_model(n_inputs, n_features, decoder_cat_dict: Dict, n_outputs: int,
                dropout: float=0, recurrent_dropout: float=0, **kwargs):

    encoder_filters = kwargs.get('encoder_filters')

    decoder_dense_units = kwargs.get('decoder_dense_units')

    encoder_inputs_layers, embedding = CNN_encoder(n_inputs, n_features, filters=encoder_filters, dropout=dropout)

    decoder_inputs_layers, outputs = LSTM_decoder(state_h=embedding, dense_units=decoder_dense_units,
                                                  n_outputs=n_outputs,
                                                  lstm_units=encoder_filters, decoder_cat_dict=decoder_cat_dict,
                                                  dropout=dropout, recurrent_dropout=recurrent_dropout)

    model = Model(inputs=encoder_inputs_layers + decoder_inputs_layers, outputs=outputs)

    return model

