from typing import Optional

from tensorflow.keras.layers import Input, LSTM, Concatenate, TimeDistributed, GlobalAveragePooling1D, Embedding, \
    Dense, RepeatVector, Dropout, Add

from train.logic.data_preparation import generate_categorical_embeddings


def LSTM_block(x, lstm_units: int, dropout: float, recurrent_dropout: float, idx: int, initial_state: Optional=None):

    x, state_h, state_c = LSTM(lstm_units, activation='relu', return_state=True, return_sequences=True,
                               dropout=dropout,
                               recurrent_dropout=recurrent_dropout, name=f'decoder_LSTM_{idx}')(x, initial_state=initial_state)

    return x, state_h, state_c


def LSTMRes_layer(x, lstm_units: int, dropout: float, recurrent_dropout: float, idx: int, initial_state: Optional=None):

    x_b = LSTM_block(x, lstm_units, dropout, recurrent_dropout, idx, initial_state=initial_state)

    x = Add()([x, x_b])

    return x


def LSTM_decoder(state_h, lstm_units, dense_units, n_outputs: int,
                 decoder_cat_dict, dropout: float=0, recurrent_dropout: float=0, state_c=None):

    inputs_layers, categorical_inputs = generate_categorical_embeddings(state_h, decoder_cat_dict)

    categorical_inputs.append(RepeatVector(n_outputs)(state_h))

    decoder_inputs = Concatenate(axis=2)(categorical_inputs)

    idx = 0
    if state_c is None:
        y, state_h, state_c = LSTM_block(decoder_inputs, lstm_units=lstm_units, dropout=dropout,
                                         recurrent_dropout=recurrent_dropout, idx=idx)
    else:
        y, state_h, state_c = LSTM_block(decoder_inputs, lstm_units=lstm_units, dropout=dropout,
                                         recurrent_dropout=recurrent_dropout, idx=idx,
                                         initial_state=[state_h, state_c])

    idx += 1
    y, state_h, state_c = LSTM_block(y, lstm_units=lstm_units, dropout=dropout,
                                     recurrent_dropout=recurrent_dropout, idx=idx,
                                     initial_state=[state_h, state_c])

    if dense_units:
        dense_units = int(dense_units)
        y = TimeDistributed(Dense(units=dense_units, activation='relu', name='decoder_dense_layer'))(y)
        if dropout > 0:
            y = TimeDistributed(Dropout(dropout))(y)

    outputs = TimeDistributed(Dense(1), name='outputs')(y)

    return inputs_layers, outputs

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