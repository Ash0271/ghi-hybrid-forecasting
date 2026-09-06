# Formulation of Hybrid Models for GHI Data Prediction

Course study project (MATH F266, BITS Pilani) on combining forecasting models to
improve Global Horizontal Irradiance (GHI) prediction for solar energy planning.

## What's mine vs. what isn't

The standalone base forecasting models — Sliding-Window ARIMA, BiLSTM, plain LSTM,
and the base wavelet-network scaffold — were provided as course material and are
**not included** in this repository. `base_models_stub.py` documents their expected
interfaces only, so the code below reads cleanly; it is not a working implementation
of those models.

**My own contribution** is the hybridization layer built on top of those base
models — combining, weighting, and stacking their outputs to produce better
predictions than any single model alone:

- `weighted_averaging.py` — combines ARIMA and BiLSTM predictions via a weighted
  average, with the optimal blend weight found using NSGA-II multi-objective
  optimization rather than a fixed value.
- `stacking.py` — feeds ARIMA's predictions as an engineered feature into the
  BiLSTM (acting as a meta-model) alongside the raw GHI sequence.
- `bagging.py` — wraps both base models in bagging ensembles (sklearn
  `BaggingRegressor`) and combines their outputs.
- `boosting.py` — trains a Gradient Boosting Regressor on top of the combined
  base-model predictions to iteratively correct errors.
- `evaluation.py` — shared RMSE / MAPE / MAE / R2 evaluation used across all of the
  above.

## Data

The GHI datasets (Bhopal, Hamirpur, Jafarabad, Thiruvananthapuram) used for
training and evaluation are **not included** in this repository.

## Results

Full methodology, results tables, and discussion are in the project report
(submitted separately as coursework, not included here). Boosting was the
best-performing ensemble technique across all four locations in testing.

## Note

These files are extracted from Jupyter notebooks and lightly cleaned up for
readability; they're meant to document the approach and are not a plug-and-play
package (running them requires the base models above, which aren't included).
