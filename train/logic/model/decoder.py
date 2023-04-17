from typing import Optional
import os
from tensorflow.keras.layers import Input, LSTM, Concatenate, TimeDistributed, GlobalAveragePooling1D, Embedding, \
    Dense, RepeatVector, Dropout, Add

from src import config
from train.common.data_preparation import generate_categorical_embeddings
from src.io.path_definition import get_file, load_yaml_file


def LSTM_block(x, lstm_units: int, dropout: float, recurrent_dropout: float, idx: int, initial_state: Optional = None):

    x, state_h, state_c = LSTM(lstm_units, activation='relu', return_state=True, return_sequences=True,
                               dropout=dropout,
                               recurrent_dropout=recurrent_dropout,
                               name=f'decoder_LSTM_{idx}')(x, initial_state=initial_state)

    return x, state_h, state_c


def LSTMRes_layer(x, lstm_units: int, dropout: float, recurrent_dropout: float, idx: int,
                  initial_state: Optional = None):

    x_b = LSTM_block(x, lstm_units, dropout, recurrent_dropout, idx, initial_state=initial_state)

    x = Add()([x, x_b])

    return x


def LSTM_decoder(state_h, lstm_units, dense_units, n_outputs: int, decoder_cat_dict,
                 dropout: float = 0, recurrent_dropout: float = 0, state_c=None,
                 weekly_inputs: bool = False):

    numerical_features = config.numerical_features

    decoder_input_layers = {}  # layers for raw input
    decoder_first_layers = []  # layers for process data before the CNN/LSTM

    categorical_inputs_layer, embedding_layers = generate_categorical_embeddings(section='decoder',
                                                                                 cat_dict=decoder_cat_dict)

    if len(categorical_inputs_layer) != 0:
        decoder_input_layers['categorical_inputs'] = categorical_inputs_layer
        decoder_first_layers.extend(embedding_layers)

    if weekly_inputs:
        previous_weekly_average_cancelled_inputs = Input(shape=(n_outputs, 1),
                                                         name='previous_weekly_average_cancelled_inputs')
        decoder_input_layers['weekly_inputs'] = previous_weekly_average_cancelled_inputs
        decoder_first_layers.append(previous_weekly_average_cancelled_inputs)

    if 'booking' in numerical_features:
        future_booking_inputs = Input(shape=(n_outputs, 1), name='future_booking_inputs')
        decoder_input_layers['booking'] = future_booking_inputs
        decoder_first_layers.append(future_booking_inputs)
            # decoder_inputs = Concatenate(axis=2)([previous_weekly_average_cancelled_inputs,
            #                                       future_booking_inputs, embedding_layers])
        # else:
        #     decoder_inputs = Concatenate(axis=2)([previous_weekly_average_cancelled_inputs,
        #                                           embedding_layers])
    # else:
    #     if 'booking' in numerical_features:
    #         future_booking_inputs = Input(shape=(n_outputs, 1), name='future_booking_inputs')
    #         decoder_input_layers['booking'] = future_booking_inputs
    #         decoder_inputs = Concatenate(axis=2)([future_booking_inputs, embedding_layers])
    #     else:
    #         if len(categorical_inputs_layer) == 0:
    #             decoder_inputs = RepeatVector(n_outputs)(state_h)
    #         else:
    #             decoder_inputs = embedding_layers

    if len(decoder_first_layers) == 0:
        y = RepeatVector(n_outputs)(state_h)
    else:
        y = Concatenate(axis=2)(decoder_first_layers)

    idx = 0
    if state_c is None:
        y, state_h, state_c = LSTM_block(y, lstm_units=lstm_units, dropout=dropout,
                                         recurrent_dropout=recurrent_dropout, idx=idx)
    else:
        y, state_h, state_c = LSTM_block(y, lstm_units=lstm_units, dropout=dropout,
                                         recurrent_dropout=recurrent_dropout, idx=idx,
                                         initial_state=[state_h, state_c])

    idx += 1
    y, state_h, state_c = LSTM_block(y, lstm_units=lstm_units, dropout=dropout,
                                     recurrent_dropout=recurrent_dropout, idx=idx,)
                                     # initial_state=[state_h, state_c])

    if dense_units:
        dense_units = int(dense_units)
        y = TimeDistributed(Dense(units=dense_units, activation='relu', name='decoder_dense_layer'))(y)
        if dropout > 0:
            y = TimeDistributed(Dropout(dropout))(y)

    outputs = TimeDistributed(Dense(1), name='outputs')(y)

    return outputs, decoder_input_layers


# def LSTM_decoder(state_h, lstm_units, dense_units, n_outputs: int,
#                  decoder_cat_dict, dropout: float=0, recurrent_dropout: float=0, state_c=None):
#
#     inputs_layers, categorical_inputs = generate_categorical_embeddings(section='decoder', cat_dict=decoder_cat_dict)
#
#     categorical_inputs.append(RepeatVector(n_outputs)(state_h))
#
#     decoder_inputs = Concatenate(axis=2)(categorical_inputs)
#
#     idx = 0
#     if state_c is None:
#         y, state_h, state_c = LSTM_block(decoder_inputs, lstm_units=lstm_units, dropout=dropout,
#                                          recurrent_dropout=recurrent_dropout, idx=idx)
#     else:
#         y, state_h, state_c = LSTM_block(decoder_inputs, lstm_units=lstm_units, dropout=dropout,
#                                          recurrent_dropout=recurrent_dropout, idx=idx,
#                                          initial_state=[state_h, state_c])
#
#     idx += 1
#     y, state_h, state_c = LSTM_block(y, lstm_units=lstm_units, dropout=dropout,
#                                      recurrent_dropout=recurrent_dropout, idx=idx,
#                                      initial_state=[state_h, state_c])
#
#     if dense_units:
#         dense_units = int(dense_units)
#         y = TimeDistributed(Dense(units=dense_units, activation='relu', name='decoder_dense_layer'))(y)
#         if dropout > 0:
#             y = TimeDistributed(Dropout(dropout))(y)
#
#     outputs = TimeDistributed(Dense(1), name='outputs')(y)
#
#     return inputs_layers, outputs



# def LSTMRes_decoder(state_h, lstm_units, dense_units, decoder_cat_dict, dropout: float=0, recurrent_dropout: float=0, state_c=None):
#
#     inputs_layers, categorical_inputs = generate_categorical_embeddings(state_h, decoder_cat_dict)
#
#     categorical_inputs.append(RepeatVector(state_h.shape[1])(state_h))
#
#     decoder_inputs = Concatenate(axis=2)(categorical_inputs)
#
#     idx = 0
#     if state_c is None:
#         y, state_h, state_c = LSTM_block(decoder_inputs, lstm_units=lstm_units, dropout=dropout,
#                                          recurrent_dropout=recurrent_dropout, idx=idx)
#     else:
#         y, state_h, state_c = LSTM_block(decoder_inputs, lstm_units=lstm_units, dropout=dropout,
#                                          recurrent_dropout=recurrent_dropout, idx=idx,
#                                          initial_state=[state_h, state_c])
#
#     idx += 1
#     y, state_h, state_c = LSTMRes_layer(y, lstm_units=lstm_units, dropout=dropout, recurrent_dropout=recurrent_dropout,
#                                         idx=idx, initial_state=[state_h, state_c])
#
#     if dense_units:
#         dense_units = int(dense_units)
#         y = TimeDistributed(Dense(units=dense_units, activation='relu', name='decoder_dense_layer'))(y)
#         if dropout > 0:
#             y = TimeDistributed(Dropout(dropout))(y)
#
#     outputs = TimeDistributed(Dense(1), name='outputs')(y)
#
#     return inputs_layers, outputs