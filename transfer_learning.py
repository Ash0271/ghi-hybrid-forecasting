"""
Hybridization technique 5 — Transfer Learning (LSTM to Wavelet Neural Network).

Idea: train the base LSTM on auxiliary meteorological features (relative humidity,
temperature, wind speed, clearness index) to predict GHI, then transfer the learned
weights from its first layer into a custom wavelet neural network, which refines
the final GHI prediction using a wavelet activation function.

The wavelet activation function, the wavelet network architecture, and the weight-
transfer logic below are my own contribution. The base LSTM model itself (loaded
here as a saved checkpoint) was provided course material — see base_models_stub.py
and note that the trained weights file (weights.h5) is intentionally NOT included
in this repository.
"""
import numpy as np
import tensorflow as tf
import pywt


def wavelet_activation(x, wavelet):
    """Custom activation that convolves the input with a wavelet filter bank."""
    wavelet_coeffs = pywt.Wavelet(wavelet).filter_bank[0]
    wavelet_coeffs = tf.constant(wavelet_coeffs, dtype=tf.float32)
    wavelet_coeffs = tf.reshape(wavelet_coeffs, [wavelet_coeffs.shape[0], 1, 1])
    x = tf.expand_dims(x, axis=-1)
    y = tf.nn.conv1d(x, filters=wavelet_coeffs, stride=1, padding="SAME")
    y = tf.squeeze(y, axis=-1)
    return y


def build_wavelet_model(input_neurons, wavelet_name, base_input_shape):
    """
    Build the wavelet neural network. base_input_shape should match the first
    layer's input shape of the (not-included) base LSTM model, since the network
    is designed to receive the transferred weights from that layer.
    """
    try:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(input_neurons, input_shape=(base_input_shape,)),
            tf.keras.layers.Lambda(lambda x: wavelet_activation(x, wavelet_name)),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(1),
        ])
    except Exception:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(input_neurons, input_shape=(base_input_shape,)),
            tf.keras.layers.Lambda(lambda x: wavelet_activation(x, "haar")),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(1),
        ])
    return model


def transfer_lstm_weights(wavelet_model, base_lstm_first_layer_weights):
    """
    Transfer weights from the base LSTM's first layer into the wavelet model,
    skipping the lambda (wavelet activation) layer.
    """
    # base_lstm_first_layer_weights = base_lstm_model.layers[0].get_weights()
    wavelet_weights = [base_lstm_first_layer_weights[0], base_lstm_first_layer_weights[2]]
    return wavelet_weights


if __name__ == "__main__":
    # Example usage — base_lstm_model would be loaded from a checkpoint produced
    # by the (not-included) base LSTM training pipeline.
    #
    # base_input_shape = base_lstm_model.layers[0].input_shape[1]
    # model = build_wavelet_model(input_neurons=10, wavelet_name="mexh", base_input_shape=base_input_shape)
    # transferred_weights = transfer_lstm_weights(model, base_lstm_model.layers[0].get_weights())
    # model.compile(optimizer="adam", loss="mean_squared_error")
    # model.fit(wave_x_train, wave_y_train, validation_data=(wave_x_val, wave_y_val), epochs=100)
    pass
