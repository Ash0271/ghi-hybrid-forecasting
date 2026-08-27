"""
Hybridization technique 3 — Bagging.

Wraps the two base models (ARIMA, BiLSTM) in sklearn's BaggingRegressor so multiple
instances of each are trained on resampled subsets of the data, then combines the
two bagged base-model predictions via a weighted average.

The wrapper functions and combination logic below are my own contribution. They
assume the base ARIMA/BiLSTM models (not included — see base_models_stub.py) are
available as train_arima / train_lstm.
"""
from sklearn.ensemble import BaggingRegressor


def train_lstm_bagging(x_train, y_train, base_model, n_estimators=10):
    """Bagging ensemble wrapper around the base BiLSTM model."""
    x_train_flattened = x_train.reshape(x_train.shape[0], -1)
    bagging = BaggingRegressor(base_estimator=base_model, n_estimators=n_estimators, random_state=0)
    bagging.fit(x_train_flattened, y_train.ravel())
    return bagging


def train_arima_bagging(train_data, base_model, n_estimators=10):
    """Bagging ensemble wrapper around the base ARIMA model."""
    bagging = BaggingRegressor(base_estimator=base_model, n_estimators=n_estimators, random_state=0)
    bagging.fit(train_data, train_data)
    return bagging


def ensemble_pred(arima_predictions, lstm_predictions, alpha=0.5):
    """Weighted combination of the two bagged base-model predictions."""
    return alpha * arima_predictions + (1 - alpha) * lstm_predictions


if __name__ == "__main__":
    # Example usage — x_train/y_train/test_data would come from the (not-included)
    # base model preprocessing pipeline.
    #
    # bilstm_bagging = train_lstm_bagging(x_train, y_train, base_model=None, n_estimators=5)
    # arima_bagging = train_arima_bagging(test_data, base_model=None, n_estimators=5)
    # bilstm_predictions = bilstm_bagging.predict(x_test.reshape(x_test.shape[0], -1))
    # arima_predictions = arima_bagging.predict(test_data)
    # final_predictions = ensemble_pred(arima_predictions, bilstm_predictions, alpha=0.5)
    pass
