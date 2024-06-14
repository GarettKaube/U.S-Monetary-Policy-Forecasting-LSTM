# Monetary-Policy-Forecasting-LSTM

LSTM model and web endpoint built with PyTorch and FastAPI for multivariate forecasting inflation, unemployment, and the Federal Funds effective rate.
The model was compared to a baseline vector autoregression which LSTM performed better on the validation set by around 8-9%.
Parameter tuning was done using Optuna's Bayesian Optimization.

Model development is contained in the notebook while "src/" folder contains code for the FastAPI endpoint.

A Dockerfile is also available so the FastAPI endpoint can be run in a container.


**Notes**: 

- Uses MLflow as the model store so requires a MLflow server.
- Requires a FRED api key in "configs/fred_key.txt".
- Post request body schema: {
  "n_periods": int}
  . Where n_periods is the number of forecasts to return.
