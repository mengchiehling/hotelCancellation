from typing import Optional

from tensorflow.keras.layers import LSTM, TimeDistributed, Dense, RepeatVector, Dropout, Add, Input, Concatenate


def LSTM_block(x, lstm_units: int, dropout: float, recurrent_dropout: float, idx: int, initial_state: Optional=None):

    if initial_state is not None:
        x, state_h, state_c = LSTM(lstm_units, activation='relu', return_state=True, return_sequences=True,
                                   dropout=dropout, recurrent_dropout=recurrent_dropout,
                                   name=f'decoder_LSTM_{idx}')(x, initial_state=initial_state)
    else:
        x, state_h, state_c = LSTM(lstm_units, activation='relu', return_state=True, return_sequences=True,
                                   dropout=dropout, recurrent_dropout=recurrent_dropout,
                                   name=f'decoder_LSTM_{idx}')(x)

    return x, state_h, state_c


def LSTMRes_layer(x, lstm_units: int, dropout: float, recurrent_dropout: float, idx: int, initial_state: Optional=None):

    x_b = LSTM_block(x, lstm_units, dropout, recurrent_dropout, idx, initial_state=initial_state)

    x = Add()([x, x_b])

    return x


def LSTM_decoder(state_h, lstm_units, dense_units, n_outputs: int,
                 dropout: float=0, recurrent_dropout: float=0, state_c=None,
                 weekly_inputs: bool=False):

    if weekly_inputs:
        previous_weekly_average_cancelled_inputs = Input(shape=(n_outputs, 1), name='previous_weekly_average_cancelled_inputs')
        # previous_weekly_average_booking_inputs = Input(shape=(n_outputs, 1), name='previous_weekly_average_booking_inputs')
        future_booking_inputs = Input(shape=(n_outputs, 1), name='future_booking_inputs')
        decoder_inputs = Concatenate(axis=2)([previous_weekly_average_cancelled_inputs,
                                              # previous_weekly_average_booking_inputs,
                                              future_booking_inputs])
    else:
        decoder_inputs = RepeatVector(n_outputs)(state_h)
        previous_weekly_average_cancelled_inputs = None
        previous_weekly_average_booking_inputs = None
        future_booking_inputs = None

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
                                     recurrent_dropout=recurrent_dropout, idx=idx)

    if dense_units:
        dense_units = int(dense_units)
        y = TimeDistributed(Dense(units=dense_units, activation='relu', name='decoder_dense_layer'))(y)
        if dropout > 0:
            y = TimeDistributed(Dropout(dropout))(y)

    outputs = TimeDistributed(Dense(1), name='outputs')(y)

    return outputs, [previous_weekly_average_cancelled_inputs,
                     # previous_weekly_average_booking_inputs,
                     future_booking_inputs]