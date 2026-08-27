"""
STUB FILE — NOT MY WORK, INCLUDED ONLY FOR INTERFACE REFERENCE.

The standalone forecasting models below (Sliding-Window ARIMA, BiLSTM, plain LSTM)
were provided as course material for this project and were NOT implemented by me.
Their real implementations are intentionally NOT included in this repository.

Everything in the other files (weighted_averaging.py, stacking.py, bagging.py,
boosting.py, transfer_learning.py) is my own hybridization work, built on top of
these base models. The functions below are empty stand-ins so the hybridization
code stays readable and shows exactly what inputs/outputs each base model is
expected to produce — they will raise NotImplementedError if actually called.
"""


def train_arima(data):
    """
    Provided base model. Fits a sliding-window ARIMA model and returns predictions.
    Not included here — see note above.
    """
    raise NotImplementedError("Base ARIMA implementation was provided course material, not included.")


def train_lstm(x_train, y_train, look_back=99):
    """
    Provided base model. Trains a Bidirectional LSTM and returns predictions.
    Not included here — see note above.
    """
    raise NotImplementedError("Base BiLSTM implementation was provided course material, not included.")


def encoder_decoder_model(look_back=99):
    """
    Provided base model architecture (Bidirectional LSTM encoder-decoder).
    Not included here — see note above.
    """
    raise NotImplementedError("Base encoder-decoder architecture was provided course material, not included.")
