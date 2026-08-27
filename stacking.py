"""
Hybridization technique 2 — Stacking.

Idea: use the ARIMA base model's predictions as an additional engineered feature,
concatenated with the raw GHI sequence, and feed that combined feature set into
the BiLSTM (acting here as the meta-model) to produce a final, improved prediction.

The feature-construction logic below (aligning ARIMA predictions to each time step,
concatenating with the GHI sequence, and scaling) is my own contribution. It assumes
the base ARIMA and BiLSTM models (not included — see base_models_stub.py) have
already produced x_train/x_val/x_test (GHI sequences) and arima_prediction_train/
val/test (ARIMA outputs) beforehand.
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def build_stacked_features(x_train, x_val, x_test,
                            arima_prediction_train, arima_prediction_val, arima_prediction_test):
    """
    Combine base-model outputs into a single stacked feature set for the meta-model.
    """
    # Reshape ARIMA predictions and repeat across the sequence length so they can
    # be concatenated with the GHI sequence features.
    arima_prediction_train = arima_prediction_train.reshape(-1, 1, 1)
    arima_prediction_test = arima_prediction_test.reshape(-1, 1, 1)
    arima_prediction_val = arima_prediction_val.reshape(-1, 1, 1)

    arima_prediction_train = np.repeat(arima_prediction_train, x_train.shape[1], axis=1)
    arima_prediction_test = np.repeat(arima_prediction_test, x_test.shape[1], axis=1)
    arima_prediction_val = np.repeat(arima_prediction_val, x_val.shape[1], axis=1)

    combined_features_train = np.concatenate((x_train, arima_prediction_train), axis=1)
    combined_features_test = np.concatenate((x_test, arima_prediction_test), axis=1)
    combined_features_val = np.concatenate((x_val, arima_prediction_val), axis=1)

    # Scale the combined feature set.
    flat_train = combined_features_train.reshape(-1, combined_features_train.shape[-1])
    flat_test = combined_features_test.reshape(-1, combined_features_test.shape[-1])
    flat_val = combined_features_val.reshape(-1, combined_features_val.shape[-1])

    scaler = MinMaxScaler()
    scaler.fit(flat_train)

    combined_features_train_scaled = scaler.transform(flat_train).reshape(combined_features_train.shape)
    combined_features_test_scaled = scaler.transform(flat_test).reshape(combined_features_test.shape)
    combined_features_val_scaled = scaler.transform(flat_val).reshape(combined_features_val.shape)

    return combined_features_train_scaled, combined_features_val_scaled, combined_features_test_scaled, scaler


if __name__ == "__main__":
    # Example usage — the meta-model itself (encoder_decoder_model) is the
    # provided BiLSTM architecture (see base_models_stub.py), trained here on
    # the stacked features instead of the raw GHI sequence alone.
    #
    # train_feats, val_feats, test_feats, scaler = build_stacked_features(...)
    # meta_model = encoder_decoder_model()
    # meta_model.compile(optimizer='adam', loss='mean_squared_error')
    # meta_model.fit(train_feats, y_train, validation_data=(val_feats, y_val), epochs=120)
    # prediction = meta_model.predict(test_feats)
    pass
